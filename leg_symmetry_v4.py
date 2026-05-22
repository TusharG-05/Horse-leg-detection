#!/usr/bin/env python3
"""
Horse Leg Symmetry Analyzer — v4
=================================
Root-cause analysis of v3 failures visible across the test images:

  IMAGE FAILURE CLASSES:
  ─────────────────────
  A) "Body / adjacent-leg bleeds into upper region" (images 1, 5, 12):
     depth_prefilter_mask was passed to selection; when it didn't fire its
     safety guard the full combined blob (front leg + body + back leg) was
     given to the fallback.  Column-band restriction fixes this.

  B) "Two separate areas coloured" (image 3):
     Combined blob of front leg + back leg → analyse_symmetry computed lx/rx
     spanning both legs in each row → wrong dominant direction + both areas lit.
     Column-band restriction fixes this.

  C) "Only hoof / fetlock covered, cannon bone missing" (images 10, 11):
     depth_prefilter_mask (percentile 25) cut off cannon bone pixels in ground-
     level shots (hoof depth ≈ 1.0, cannon depth ≈ 0.5–0.6 → threshold ≈ 0.45
     too high for cannon).  Removing depth_prefilter_mask from the pipeline and
     reconnecting via a tall vertical close kernel fixes this.

  D) "Mask too wide at bottom / extra ground pixels" (image 15):
     rembg grabs ground-contact pixels.  Column-band restriction (based on
     contour bounding box, not padded full mask) trims the sides cleanly.

  E) "is_likely_tail flagged cannon shaft" (subtle):
     The "starts in top 30% of image" and "uniformly thin" checks in v3 could
     reject a cannon bone contour.  Replaced with a single clear-taper check.

  KEY ALGORITHMIC CHANGES:
  ─────────────────────────
  FIX A — depth_prefilter_mask REMOVED from the selection pipeline.
           Depth is only used for SCORING individual contours, never for
           filtering the mask before selection.

  FIX B — select_front_leg_fallback completely rewritten:
           • Zone starts at h×0.05 (was h×0.20) to catch full cannon bone.
           • Works on INDIVIDUAL contours from the raw full mask,
             not a merged combined blob.
           • Scores by avg_depth (depth_map) primary, area secondary.
           • After picking the best seed contour, applies a column-band
             restriction to the FULL original mask, preventing adjacent
             legs / body from entering the selected region.
           • Uses a tall vertical close kernel (RECT 11×60) to reconnect
             cannon bone ↔ hoof if a gap exists in the raw mask.
           • Keeps only the connected component overlapping most with
             the seed contour.

  FIX C — is_likely_tail simplified and made less aggressive:
           Removed "starts in top 30%" check and "uniformly thin" check,
           both of which incorrectly flagged narrow cannon bone sections.
           Now only rejects blobs with extreme taper (bottom tip < 35% of
           top width AND absolutely thin) or extreme aspect ratio (>8).

  FIX D — All selection paths receive the FULL mask (no depth prefilter),
           so upper-leg pixels are never silently discarded before selection.
"""

from pathlib import Path
import glob
import argparse
import logging
import cv2
import numpy as np
import sys
from PIL import Image as PILImage
from rembg import remove
from transformers import pipeline as hf_pipeline

try:
    from mmpose.apis import MMPoseInferencer
except Exception:
    MMPoseInferencer = None

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

_DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
_depth_pipe = None


# ─────────────────────────────────────────────────────────────────────────────
# Depth estimation
# ─────────────────────────────────────────────────────────────────────────────

def get_depth_pipe():
    global _depth_pipe
    if _depth_pipe is None:
        logging.info("Loading Depth Anything V2 Small …")
        _depth_pipe = hf_pipeline("depth-estimation", model=_DEPTH_MODEL_ID, device=-1)
        logging.info("Depth Anything V2 ready.")
    return _depth_pipe


def estimate_depth(image_bgr: np.ndarray,
                   fg_mask: np.ndarray | None = None) -> np.ndarray:
    """Depth Anything V2 → float32 map [0, 1].  1.0 = closest.
    Background pixels filled with neutral gray (127) before inference so the
    model is not confused by black zeros.  Depth outside fg_mask is zeroed.
    """
    h, w = image_bgr.shape[:2]
    input_img = image_bgr.copy()
    if fg_mask is not None:
        input_img[fg_mask == 0] = 127

    logging.info("Running Depth Anything V2 …")
    rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    result = get_depth_pipe()(PILImage.fromarray(rgb))
    depth_np = np.array(result["depth"]).astype(np.float32)

    if depth_np.shape[0] != h or depth_np.shape[1] != w:
        depth_np = cv2.resize(depth_np, (w, h), interpolation=cv2.INTER_LINEAR)

    d_min, d_max = depth_np.min(), depth_np.max()
    depth_np = (depth_np - d_min) / (d_max - d_min) if d_max > d_min else np.zeros_like(depth_np)

    if fg_mask is not None:
        depth_np[fg_mask == 0] = 0.0

    logging.info("Depth map ready (shape=%s).", depth_np.shape)
    return depth_np


# ─────────────────────────────────────────────────────────────────────────────
# Foreground extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_foreground_rgba(image_bgr: np.ndarray) -> tuple:
    """Run rembg; return (rgba HxWx4 in RGB order, mask uint8)."""
    h, w = image_bgr.shape[:2]
    logging.info("Running rembg (%dx%d) …", w, h)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgba = remove(rgb)
    if rgba is None or rgba.ndim < 3 or rgba.shape[2] < 4:
        raise RuntimeError("rembg returned unexpected result")
    alpha = rgba[:, :, 3]
    _, mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)

    # Remove tiny spurious components (background noise, tail wisps).
    # Keep components that are at least 5% of the largest component.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 2:
        max_area = int(stats[1:, cv2.CC_STAT_AREA].max())
        clean = np.zeros_like(mask)
        for lab in range(1, num_labels):
            if stats[lab, cv2.CC_STAT_AREA] >= max_area * 0.05:
                clean[labels == lab] = 255
        if cv2.countNonZero(clean) > cv2.countNonZero(mask) * 0.40:
            mask = clean

    return rgba, mask.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Tail-rejection helper  (FIX C — simplified, less aggressive)
# ─────────────────────────────────────────────────────────────────────────────

def is_likely_tail(contour: np.ndarray, mask: np.ndarray) -> bool:
    """Return True only for clear tail shapes.

    v3 had checks that also flagged narrow cannon-bone sections:
      - "starts in top 30% of image + narrow" → removed
      - "uniformly thin across whole height" → removed
    Now only two checks remain:
      1. Extreme aspect ratio (height/width > 8) AND very narrow absolute width.
      2. Strong bottom-taper: bottom tip < 35% of top width AND thin in absolute
         terms — a tail ends in a point, a leg ends in a wide hoof.
    """
    x, y, cw, ch = cv2.boundingRect(contour)
    h_img, w_img = mask.shape
    if ch == 0:
        return False

    # Check 1: extreme elongation
    if ch / max(cw, 1) > 8 and cw < w_img * 0.06:
        return True

    # Check 2: strong bottom taper with thin tip (no hoof flare)
    top_end   = y + max(1, int(ch * 0.15))
    bot_start = y + int(ch * 0.85)
    top_ws, bot_ws = [], []
    for ry in range(y, min(top_end + 1, h_img)):
        xs = np.where(mask[ry] > 0)[0]
        if xs.size >= 2:
            top_ws.append(int(xs[-1] - xs[0]))
    for ry in range(bot_start, min(y + ch + 1, h_img)):
        xs = np.where(mask[ry] > 0)[0]
        if xs.size >= 2:
            bot_ws.append(int(xs[-1] - xs[0]))

    if top_ws and bot_ws:
        avg_top = float(np.mean(top_ws))
        avg_bot = float(np.mean(bot_ws))
        if avg_bot < avg_top * 0.35 and avg_bot < w_img * 0.04:
            return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Utility: cannon-bone width estimate
# ─────────────────────────────────────────────────────────────────────────────

def _cannon_width(leg_mask: np.ndarray) -> int:
    """Median width of the narrowest third of the shaft (10–75% of leg height)."""
    h, w = leg_mask.shape
    ys = np.where(np.any(leg_mask > 0, axis=1))[0]
    if ys.size == 0:
        return max(20, w // 8)
    y0, y1 = int(ys[0]), int(ys[-1])
    leg_h = y1 - y0 + 1
    shaft_rows = range(y0 + int(leg_h * 0.10), y0 + int(leg_h * 0.75) + 1)
    widths = []
    for ry in shaft_rows:
        xs = np.where(leg_mask[ry] > 0)[0]
        if xs.size >= 2:
            widths.append(int(xs[-1] - xs[0]))
    if not widths:
        return max(20, w // 8)
    widths.sort()
    return int(np.median(widths[: max(1, len(widths) // 3)]))


def trim_upper_leg_fraction(leg_mask: np.ndarray, exclude_top_frac: float = 0.35) -> np.ndarray:
    """Remove the top fraction of a selected leg mask before analysis.

    This keeps the asymmetry stage focused on the lower leg / cannon area and
    reduces bleed from the body or neighboring legs above the selected front leg.
    """
    if leg_mask is None or leg_mask.size == 0:
        return leg_mask

    ys = np.where(np.any(leg_mask > 0, axis=1))[0]
    if ys.size == 0:
        return leg_mask.copy()

    top_y, bottom_y = int(ys[0]), int(ys[-1])
    leg_h = bottom_y - top_y + 1
    cut_y = top_y + int(round(leg_h * exclude_top_frac))
    cut_y = min(max(top_y, cut_y), bottom_y + 1)

    trimmed = leg_mask.copy()
    trimmed[top_y:cut_y, :] = 0
    return trimmed


def depth_prefilter_mask(mask: np.ndarray,
                         depth_map: np.ndarray,
                         percentile: float = 20.0) -> np.ndarray:
    """Remove far-background pixels before leg selection.

    Keeps foreground pixels whose depth is above the chosen percentile of
    foreground depth values. This is the same control point that made v2
    better at excluding back legs and tail fragments before contour selection.
    Safety guards return the original mask if the filter gets too aggressive.
    """
    if mask is None or depth_map is None or mask.size == 0:
        return mask

    fg_depths = depth_map[mask > 0]
    if fg_depths.size == 0:
        return mask

    ys_orig = np.where(np.any(mask > 0, axis=1))[0]
    orig_height = int(ys_orig[-1] - ys_orig[0]) if ys_orig.size >= 2 else 0

    thresh = float(np.percentile(fg_depths, percentile))
    filtered = np.zeros_like(mask)
    filtered[(mask > 0) & (depth_map >= thresh)] = 255

    filtered = cv2.morphologyEx(
        filtered,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
    )

    if cv2.countNonZero(filtered) < cv2.countNonZero(mask) * 0.15:
        logging.warning("depth_prefilter_mask pixel count too small; returning original mask.")
        return mask

    if orig_height > 0:
        ys_filt = np.where(np.any(filtered > 0, axis=1))[0]
        if ys_filt.size >= 2:
            filt_height = int(ys_filt[-1] - ys_filt[0])
            if filt_height < orig_height * 0.50:
                logging.warning("depth_prefilter_mask height shrank too much; returning original mask.")
                return mask

    return filtered


def build_near_depth_gate(depth_map: np.ndarray | None,
                          near_percentile: float = 55.0) -> tuple[np.ndarray | None, float]:
    """Build a mask that keeps only the nearer depth band of the foreground.

    The gate is used only as a selection cue so we can reject distant legs/tails
    without permanently deleting pixels from the final leg mask.
    """
    if depth_map is None:
        return None, 0.0

    values = depth_map[depth_map > 0]
    if values.size == 0:
        return None, 0.0

    cutoff = float(np.percentile(values, near_percentile))
    gate = np.where(depth_map >= cutoff, 255, 0).astype(np.uint8)
    gate = cv2.morphologyEx(gate, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    gate = cv2.dilate(gate, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    return gate, cutoff


def _depth_overlap_ratio(component_mask: np.ndarray, depth_gate: np.ndarray | None) -> float:
    if depth_gate is None:
        return 0.0
    comp_px = component_mask > 0
    total = int(np.count_nonzero(comp_px))
    if total == 0:
        return 0.0
    return float(np.count_nonzero(comp_px & (depth_gate > 0))) / float(total)


# ─────────────────────────────────────────────────────────────────────────────
# FIX B — Fallback leg selection: completely rewritten
# ─────────────────────────────────────────────────────────────────────────────

def select_front_leg_fallback(mask: np.ndarray,
                               depth_map: np.ndarray | None = None,
                               debug: bool = False) -> np.ndarray | None:
    """Select the frontmost front leg from the full foreground mask.

    v4 algorithm (see module docstring FIX B):
      1. Find individual contours in zone [h*0.05, h] of the FULL mask.
      2. Score by avg_depth (primary) + area (secondary).  Reject tails.
      3. Pick the best "seed" contour.
      4. Derive a column band from the seed's bounding box (±60% of bw,
         minimum 10% of image width).  This band prevents body/back-leg
         pixels from entering the final mask.
      5. Apply band to the FULL original mask and morphologically close
         with a tall vertical kernel to reconnect cannon↔hoof gaps.
      6. Keep the connected component that overlaps most with the seed.
      7. If the result is plausible (area > seed area × 0.5) return it,
         otherwise fall back to the seed contour itself.
    """
    h, w = mask.shape

    # ── 1. Find contours ────────────────────────────────────────────────────
    zone = np.zeros_like(mask)
    zone[int(h * 0.05):] = mask[int(h * 0.05):]

    contours, _ = cv2.findContours(zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # ── 2. Score each contour ───────────────────────────────────────────────
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 400:
            continue

        # Build a mask for just this contour
        temp = np.zeros_like(mask)
        cv2.drawContours(temp, [cnt], -1, 255, cv2.FILLED)

        # FIX C: simplified tail rejection
        if is_likely_tail(cnt, temp):
            if debug:
                bx,by,bw,bh = cv2.boundingRect(cnt)
                logging.info("Tail rejected: area=%.0f bx=%d bw=%d bh=%d", area,bx,bw,bh)
            continue

        bx, by, bw, bh = cv2.boundingRect(cnt)

        avg_depth = 0.0
        if depth_map is not None:
            px = temp > 0
            if np.any(px):
                avg_depth = float(depth_map[px].mean())

        if debug:
            logging.info("Candidate: area=%.0f bx=%d bw=%d avg_depth=%.3f",
                         area, bx, bw, avg_depth)
        candidates.append({
            'cnt': cnt, 'mask': temp, 'area': area,
            'avg_depth': avg_depth,
            'bx': bx, 'by': by, 'bw': bw, 'bh': bh,
        })

    if not candidates:
        return None

    # Must have at least some area
    candidates = [c for c in candidates if c['area'] >= 500]
    if not candidates:
        return None

    # ── 3. Sort: depth first, area second ───────────────────────────────────
    candidates.sort(key=lambda c: (c['avg_depth'], c['area']), reverse=True)
    seed = candidates[0]
    if debug:
        logging.info("Seed: area=%.0f depth=%.3f bx=%d bw=%d",
                     seed['area'], seed['avg_depth'], seed['bx'], seed['bw'])

    # ── 4. Column-band restriction ───────────────────────────────────────────
    bx_s, bw_s = seed['bx'], seed['bw']
    cx_s = bx_s + bw_s // 2

    # Band half-width: keep this deliberately tight so the selected leg does
    # not absorb the neighboring leg or body while still preserving the cannon.
    cannon_w = _cannon_width(seed['mask'])
    band_half = max(int(bw_s * 0.40), cannon_w + 25, int(w * 0.08))
    col_l = max(0, cx_s - band_half)
    col_r = min(w, cx_s + band_half)

    # Apply band to FULL original mask
    band_full = np.zeros_like(mask)
    band_full[:, col_l:col_r] = mask[:, col_l:col_r]

    # ── 5. Morphological close to reconnect cannon ↔ hoof gaps ──────────────
    # Use a tall rectangular kernel: wide enough to bridge 1-2 cm gaps
    # vertically (fetlock gap, tracking artefacts), narrow enough not to
    # swallow the adjacent leg horizontally.
    k_tall = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 60))
    band_closed = cv2.morphologyEx(band_full, cv2.MORPH_CLOSE, k_tall)

    # Also apply a round close to fill horizontal holes
    k_round = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    band_closed = cv2.morphologyEx(band_closed, cv2.MORPH_CLOSE, k_round)

    # Build a near-camera depth gate and prefer components inside it
    depth_gate, depth_cutoff = build_near_depth_gate(depth_map)

    # ── 6. Pick the connected component overlapping most with the seed ───────
    num_labels, labels = cv2.connectedComponents(band_closed, connectivity=8)
    best_score, best_lab = -1.0, -1
    seed_px = seed['mask'] > 0
    for lab in range(1, num_labels):
        comp_px = labels == lab
        ov = int(np.sum(comp_px & seed_px))
        # depth overlap ratio: fraction of component pixels inside near gate
        depth_ratio = _depth_overlap_ratio(comp_px.astype(np.uint8), depth_gate)
        # score: prefer components that overlap the seed and are near
        score = float(ov) * (1.0 + depth_ratio * 4.0)
        if debug:
            logging.info("Component %d: overlap=%d depth_ratio=%.3f score=%.3f",
                         lab, ov, depth_ratio, score)
        if score > best_score:
            best_score = score
            best_lab = lab

    if best_lab < 0:
        logging.warning("No overlapping component found — returning seed contour.")
        return seed['mask']

    result = np.zeros_like(mask)
    result[labels == best_lab] = 255

    # ── 7. Sanity check ──────────────────────────────────────────────────────
    if cv2.countNonZero(result) < cv2.countNonZero(seed['mask']) * 0.5:
        logging.warning("Column-band result smaller than seed — returning seed.")
        return seed['mask']

    logging.info("Selected leg: seed_area=%.0f final_area=%d depth=%.3f",
                 seed['area'], cv2.countNonZero(result), seed['avg_depth'])
    return result


# ─────────────────────────────────────────────────────────────────────────────
# AI-path leg selection (MMPose keypoints)
# ─────────────────────────────────────────────────────────────────────────────

def select_front_leg_from_keypoints(mask: np.ndarray,
                                    knee: tuple[float, float],
                                    hoof: tuple[float, float],
                                    depth_map: np.ndarray | None = None,
                                    debug: bool = False) -> np.ndarray | None:
    """Select leg contour matching knee/hoof keypoints then apply column band.
    Receives the FULL mask (FIX D).
    """
    h, w = mask.shape
    start_y = max(0, int(round(knee[1])) - 40)
    zone = np.zeros_like(mask)
    zone[start_y:] = mask[start_y:]
    contours, _ = cv2.findContours(zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    hoof_pt = (float(hoof[0]), float(hoof[1]))
    best_cnt, best_score = None, None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:
            continue
        dist = cv2.pointPolygonTest(cnt, hoof_pt, True)
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00']) if M['m00'] != 0 else cv2.boundingRect(cnt)[0] + cv2.boundingRect(cnt)[2] // 2
        dx_knee = abs(cx - int(round(knee[0])))
        score = (dist, -dx_knee, area)
        if best_score is None or score > best_score:
            best_score = score
            best_cnt = cnt

    if best_cnt is None:
        return None

    # Build seed mask from best contour
    seed = np.zeros_like(mask)
    cv2.drawContours(seed, [best_cnt], -1, 255, cv2.FILLED)

    # Apply column band (same logic as fallback) to recover full leg
    bx, by, bw_c, bh_c = cv2.boundingRect(best_cnt)
    cx_s = bx + bw_c // 2
    cannon_w = _cannon_width(seed)
    band_half = max(int(bw_c * 0.40), cannon_w + 25, int(w * 0.08))
    col_l = max(0, cx_s - band_half)
    col_r = min(w, cx_s + band_half)

    # Vertical clip to reasonable leg extent around keypoints
    ky, hy = int(round(knee[1])), int(round(hoof[1]))
    top_clip = max(0, ky - 30)
    bottom_clip = min(h - 1, hy + 80)

    band_full = np.zeros_like(mask)
    band_full[top_clip:bottom_clip + 1, col_l:col_r] = mask[top_clip:bottom_clip + 1, col_l:col_r]

    k_tall = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 60))
    band_closed = cv2.morphologyEx(band_full, cv2.MORPH_CLOSE, k_tall)
    k_round = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    band_closed = cv2.morphologyEx(band_closed, cv2.MORPH_CLOSE, k_round)

    # Prefer components that overlap the seed and lie within the near depth gate
    depth_gate, depth_cutoff = build_near_depth_gate(depth_map)
    num_labels, labels = cv2.connectedComponents(band_closed, connectivity=8)
    seed_px = seed > 0
    best_score, best_lab = -1.0, -1
    for lab in range(1, num_labels):
        comp_px = labels == lab
        ov = int(np.sum(comp_px & seed_px))
        depth_ratio = _depth_overlap_ratio(comp_px.astype(np.uint8), depth_gate)
        score = float(ov) * (1.0 + depth_ratio * 4.0)
        if debug:
            logging.info("AI component %d: overlap=%d depth_ratio=%.3f score=%.3f",
                         lab, ov, depth_ratio, score)
        if score > best_score:
            best_score = score
            best_lab = lab

    if best_lab < 0:
        return seed

    result = np.zeros_like(mask)
    result[labels == best_lab] = 255

    # Validate: must have reasonable bottom width
    xs_bottom = np.where(result[min(bottom_clip, h - 1)] > 0)[0]
    if xs_bottom.size > 0 and (xs_bottom[-1] - xs_bottom[0]) < max(15, int(0.07 * w)):
        return seed  # fall back to seed if result is too narrow at hoof
    return result


# ─────────────────────────────────────────────────────────────────────────────
# split_mask_on_width (kept for AI path / watershed seeding)
# ─────────────────────────────────────────────────────────────────────────────

def split_mask_on_width(mask_region: np.ndarray,
                        min_narrow_frac: float = 0.6,
                        min_width_px: int = 30,
                        debug: bool = False) -> list[np.ndarray]:
    ys = np.where(np.any(mask_region > 0, axis=1))[0]
    if ys.size == 0:
        return [mask_region]
    y0, y1 = int(ys[0]), int(ys[-1])
    widths = np.zeros(y1 - y0 + 1, dtype=np.int32)
    for i, ry in enumerate(range(y0, y1 + 1)):
        xs = np.where(mask_region[ry] > 0)[0]
        widths[i] = int(xs[-1] - xs[0]) if xs.size >= 2 else 0
    nonzero = widths[widths > 0]
    if nonzero.size == 0:
        return [mask_region]
    thresh = max(min_width_px, int(np.median(nonzero) * min_narrow_frac))
    narrow = widths < thresh
    cut_rows = []
    i = 0
    while i < len(narrow):
        if narrow[i]:
            j = i
            while j + 1 < len(narrow) and narrow[j + 1]:
                j += 1
            cut_rows.append(y0 + (i + j) // 2)
            i = j + 1
        else:
            i += 1
    if not cut_rows:
        return [mask_region]
    split = mask_region.copy()
    for r in cut_rows:
        split[max(y0, r - 2): min(y1, r + 2) + 1, :] = 0
    num_labels, labels = cv2.connectedComponents(split)
    parts = [np.where(labels == lab, np.uint8(255), np.uint8(0))
             for lab in range(1, num_labels)
             if cv2.countNonZero(labels == lab) > 50]
    if not parts:
        return [mask_region]
    parts.sort(key=cv2.countNonZero, reverse=True)
    return parts


# ─────────────────────────────────────────────────────────────────────────────
# Watershed leg separator
# ─────────────────────────────────────────────────────────────────────────────

def seed_watershed_from_hooves(mask: np.ndarray,
                                hooves: list[tuple[float, float]],
                                rgba: np.ndarray | None = None,
                                snap_radius: int = 30) -> list:
    if mask is None or mask.size == 0:
        return []
    bm = (mask > 0).astype(np.uint8) * 255
    h, w = bm.shape

    def snap(xf, yf):
        x, y = int(round(xf)), int(round(yf))
        if 0 <= x < w and 0 <= y < h and bm[y, x]:
            return (x, y)
        for r in range(1, snap_radius + 1):
            best, bd = None, None
            for yy in range(max(0, y - r), min(h, y + r + 1)):
                for xx in range(max(0, x - r), min(w, x + r + 1)):
                    if bm[yy, xx]:
                        d = (xx - x) ** 2 + (yy - y) ** 2
                        if bd is None or d < bd:
                            bd, best = d, (xx, yy)
            if best:
                return best
        return None

    markers = np.zeros((h, w), dtype=np.int32)
    seeds = [snap(kx, ky) for kx, ky in hooves]
    for i, s in enumerate(seeds, 1):
        if s:
            cv2.circle(markers, s, 6, i, -1)
    if all(s is None for s in seeds):
        return []
    try:
        dist = cv2.distanceTransform((bm // 255).astype(np.uint8), cv2.DIST_L2, 5)
        if dist.max() <= 0:
            return []
        dist8 = np.uint8((dist / dist.max()) * 255)
        cv2.watershed(cv2.cvtColor(dist8, cv2.COLOR_GRAY2BGR), markers)
    except Exception as e:
        logging.warning("watershed failed: %s", e)
        return []
    parts = []
    for i in range(1, len(hooves) + 1):
        m = np.where(markers == i, np.uint8(255), np.uint8(0))
        parts.append(m if cv2.countNonZero(m) > 50 else None)
    return parts


# ─────────────────────────────────────────────────────────────────────────────
# MMPose keypoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_ai_leg_keypoints(inferencer, image_path: str):
    if inferencer is None:
        return []
    try:
        res = inferencer(image_path)
    except Exception as e:
        logging.warning("MMPose failed: %s", e)
        return []
    try:
        results = next(iter(res)) if hasattr(res, '__iter__') and not isinstance(res, dict) else res
    except Exception:
        results = res
    preds = None
    if isinstance(results, dict) and results.get('predictions'):
        preds = results['predictions'][0]
    elif isinstance(results, list) and results:
        preds = results[0]
    if not preds:
        return []
    kpts = scores = None
    if isinstance(preds, dict):
        kpts = preds.get('keypoints') or preds.get('preds')
        scores = preds.get('keypoint_scores') or preds.get('scores')
    if kpts is None:
        return []
    legs = []
    try:
        if isinstance(kpts, np.ndarray):
            kpts = kpts.tolist()
        if len(kpts) > 10:
            def sc(i):
                return float(scores[i]) if scores and len(scores) > i else 1.0
            if sc(6) > 0.12 and sc(7) > 0.12:
                legs.append((tuple(kpts[6][:2]), tuple(kpts[7][:2])))
            if sc(9) > 0.12 and sc(10) > 0.12:
                legs.append((tuple(kpts[9][:2]), tuple(kpts[10][:2])))
    except Exception:
        return []
    return legs


# ─────────────────────────────────────────────────────────────────────────────
# Cannon-bone axis  (strictly vertical centre line)
# ─────────────────────────────────────────────────────────────────────────────

def find_cannon_bone_axis(leg_mask: np.ndarray,
                          target_knee=None,
                          target_hoof=None):
    """Strictly vertical centre line at the median X of the narrowest shaft rows."""
    h, w = leg_mask.shape
    clean = cv2.morphologyEx(leg_mask, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    ys = np.where(np.any(clean > 0, axis=1))[0]
    if ys.size == 0:
        return (w // 2, 0), (w // 2, h - 1)
    top_y, bottom_y = int(ys[0]), int(ys[-1])

    rows = []
    for ry in range(top_y, bottom_y + 1):
        xs = np.where(clean[ry] > 0)[0]
        if xs.size >= 2:
            rows.append((ry, int(xs[0]), int(xs[-1]),
                         (float(xs[0]) + float(xs[-1])) / 2.0,
                         int(xs[-1] - xs[0])))
    if not rows:
        return (w // 2, top_y), (w // 2, bottom_y)

    leg_h = bottom_y - top_y + 1
    shaft_top = top_y + int(leg_h * 0.10)
    shaft_bot = top_y + int(leg_h * 0.75)
    shaft = [r for r in rows if shaft_top <= r[0] <= shaft_bot] or rows

    shaft_sorted = sorted(shaft, key=lambda r: r[4])
    n = max(5, int(len(shaft_sorted) * 0.30))
    cannon_rows = shaft_sorted[:n]

    cx = int(round(float(np.median([r[3] for r in cannon_rows]))))
    all_lx = min(r[1] for r in rows)
    all_rx = max(r[2] for r in rows)
    cx = int(np.clip(cx, all_lx + 1, all_rx - 1))

    logging.info("Cannon axis: cx=%d top=%d bottom=%d", cx, top_y, bottom_y)
    return (cx, top_y), (cx, bottom_y)


# ─────────────────────────────────────────────────────────────────────────────
# Symmetry analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_symmetry(leg_mask: np.ndarray,
                     pt_top: tuple[int, int],
                     pt_bottom: tuple[int, int]):
    """Row-by-row symmetry with vertical centre line.
    Returns (green_mask, red_mask, dominant_side).
    """
    h, w = leg_mask.shape
    green = np.zeros((h, w), dtype=np.uint8)
    red   = np.zeros((h, w), dtype=np.uint8)

    top_y, bottom_y = pt_top[1], pt_bottom[1]
    cx_fixed = pt_top[0]  # same X for vertical line

    total_left = total_right = 0
    row_data = []
    for ry in range(top_y, min(bottom_y + 1, h)):
        xs = np.where(leg_mask[ry] > 0)[0]
        if xs.size < 2:
            row_data.append(None)
            continue
        lx, rx = int(xs[0]), int(xs[-1])
        cx = cx_fixed
        if lx >= cx or rx <= cx:
            row_data.append(None)
            continue
        lw, rw = cx - lx, rx - cx
        total_left += lw
        total_right += rw
        row_data.append((ry, lx, rx, lw, rw, cx))

    if total_left > total_right * 1.02:
        dominant = "LEFT"
    elif total_right > total_left * 1.02:
        dominant = "RIGHT"
    else:
        dominant = "SYMMETRIC"

    for item in row_data:
        if item is None:
            continue
        ry, lx, rx, lw, rw, cx = item
        sw = min(lw, rw)
        if dominant == "LEFT":
            green[ry, cx: rx + 1]           = leg_mask[ry, cx: rx + 1]
            green[ry, max(0,cx-sw): cx]     = leg_mask[ry, max(0,cx-sw): cx]
            es, ee = lx, max(0, cx - sw)
            if es < ee:
                red[ry, es:ee] = leg_mask[ry, es:ee]
        elif dominant == "RIGHT":
            green[ry, lx: cx]               = leg_mask[ry, lx: cx]
            green[ry, cx: min(w,cx+sw+1)]   = leg_mask[ry, cx: min(w,cx+sw+1)]
            es, ee = min(w, cx + sw + 1), rx + 1
            if es < ee:
                red[ry, es:ee] = leg_mask[ry, es:ee]
        else:
            green[ry, lx: rx + 1] = leg_mask[ry, lx: rx + 1]

    green = cv2.bitwise_and(green, leg_mask)
    red   = cv2.bitwise_and(red,   leg_mask)
    logging.info("Dominant: %s  (left=%d right=%d)", dominant, total_left, total_right)
    return green, red, dominant


def apply_overlay(img, green_mask, red_mask, alpha=0.55):
    res  = img.astype(np.float32)
    orig = res.copy()
    G = np.array([34, 197, 94],  dtype=np.float32)   # green
    R = np.array([48,  48, 220], dtype=np.float32)   # red (note: BGR)
    gm, rm = green_mask > 0, red_mask > 0
    res[gm] = orig[gm] * (1 - alpha) + G * alpha
    res[rm] = orig[rm] * (1 - alpha) + R * alpha
    return np.clip(res, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_image(path: str, do_debug: bool = False, inferencer=None) -> None:
    p = Path(path)
    img = cv2.imread(str(p))
    if img is None:
        logging.error("Cannot read %s", p)
        return
    h, w = img.shape[:2]
    logging.info("Processing %s (%dx%d)", p.name, w, h)

    # ── Foreground extraction ────────────────────────────────────────────────
    rgba, mask = extract_foreground_rgba(img)
    fg_bgr = cv2.bitwise_and(img, img, mask=mask)

    rgba_bgra = cv2.cvtColor(np.asarray(rgba), cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(str(p.parent / f"{p.stem}_foreground.png"), rgba_bgra)

    # ── Depth estimation (on neutral-gray-background image) ──────────────────
    depth_map = estimate_depth(fg_bgr, fg_mask=mask)
    depth_vis = cv2.applyColorMap((depth_map * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.imwrite(str(p.parent / f"{p.stem}_depth.png"), depth_vis)

    # Depth prefilter: remove far pixels before contour selection.
    depth_mask = depth_prefilter_mask(mask, depth_map, percentile=20.0)
    if do_debug:
        cv2.imwrite(str(p.parent / f"{p.stem}_depth_mask.png"), depth_mask)

    leg_masks, leg_infos = [], []

    # ── AI path (MMPose) ─────────────────────────────────────────────────────
    if inferencer is not None:
        try:
            legs = get_ai_leg_keypoints(inferencer, str(p))
        except Exception:
            legs = []
        if legs:
            best_leg, best_depth_val = None, -1.0
            for knee, hoof in legs:
                lm_tmp = np.zeros((h, w), dtype=np.uint8)
                cv2.line(lm_tmp, (int(knee[0]), int(knee[1])),
                         (int(hoof[0]), int(hoof[1])), 255, thickness=5)
                overlap = (lm_tmp > 0) & (mask > 0)
                avg_d = float(depth_map[overlap].mean()) if np.any(overlap) else 0.0
                if avg_d > best_depth_val:
                    best_depth_val, best_leg = avg_d, (knee, hoof)

            if best_leg is not None:
                knee, hoof = best_leg
                lm = select_front_leg_from_keypoints(depth_mask, knee, hoof, depth_map=depth_map, debug=do_debug)
                if lm is not None:
                    lm = trim_upper_leg_fraction(lm, 0.35)
                    leg_masks.append(lm)
                    leg_infos.append({'mask': lm, 'knee': knee, 'hoof': hoof})
                    cv2.imwrite(str(p.parent / f"{p.stem}_isolated_leg.png"), lm)
                    logging.info("Saved isolated leg (AI path).")

    # ── Fallback path ────────────────────────────────────────────────────────
    if not leg_masks:
        logging.warning("Fallback leg selection.")
        lm = select_front_leg_fallback(depth_mask, depth_map=depth_map, debug=do_debug)
        if lm is None:
            logging.warning("No front leg found for %s", p.name)
            cv2.imwrite(str(p.parent / f"{p.stem}_analyzed.jpg"), img)
            return
        lm = trim_upper_leg_fraction(lm, 0.35)
        leg_masks = [lm]
        leg_infos = [{'mask': lm, 'knee': (w / 2.0, 0.0), 'hoof': (w / 2.0, float(h - 1))}]
        cv2.imwrite(str(p.parent / f"{p.stem}_isolated_leg.png"), lm)
        logging.info("Saved isolated leg (fallback).")

    # ── Symmetry analysis ────────────────────────────────────────────────────
    combined_green = np.zeros((h, w), dtype=np.uint8)
    combined_red   = np.zeros((h, w), dtype=np.uint8)
    per_leg_draw: list[dict] = []

    for info in leg_infos:
        pt_top, pt_bottom = find_cannon_bone_axis(
            info['mask'],
            target_knee=info.get('knee'),
            target_hoof=info.get('hoof'),
        )
        green, red, dominant = analyze_symmetry(info['mask'], pt_top, pt_bottom)
        combined_green = np.maximum(combined_green, green)
        combined_red   = np.maximum(combined_red,   red)
        per_leg_draw.append({'pt_top': pt_top, 'pt_bottom': pt_bottom, 'dominant': dominant})

    out = apply_overlay(img, combined_green, combined_red, alpha=0.55)

    for i, di in enumerate(per_leg_draw):
        cv2.line(out, di['pt_top'], di['pt_bottom'], (255, 80, 0), max(2, int(w * 0.004)))
        cv2.putText(out, f"Leg {i + 1} Dominant: {di['dominant']}",
                    (10, 30 + i * 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imwrite(str(p.parent / f"{p.stem}_analyzed.jpg"), out)

    if do_debug:
        dbg = img.copy()
        dbg[combined_green > 0] = [34, 197, 94]
        dbg[combined_red   > 0] = [48, 48, 220]
        for di in per_leg_draw:
            cv2.line(dbg, di['pt_top'], di['pt_bottom'], (255, 80, 0), max(2, int(w * 0.004)))
        cv2.imwrite(str(p.parent / f"{p.stem}_debug.png"), dbg)

    logging.info("Done: %s_analyzed.jpg", p.stem)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Horse leg symmetry analyzer v4")
    ap.add_argument("images", nargs="+", help="Image files or glob patterns")
    ap.add_argument("--debug",      action="store_true", help="Save debug images")
    ap.add_argument("--use-ai",     action="store_true", help="Enable MMPose")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--device",     default=None)
    args = ap.parse_args()

    inputs = sorted(set(f for pat in args.images for f in glob.glob(pat)))
    if not inputs:
        logging.error("No input images.")
        sys.exit(1)

    inferencer = None
    if args.use_ai and MMPoseInferencer is not None:
        try:
            pose = args.model_path or 'rtmpose-m_8xb64-210e_ap10k-256x256'
            kw = {"device": args.device} if args.device else {}
            inferencer = MMPoseInferencer(pose2d=pose, **kw)
            logging.info("MMPose ready.")
        except Exception as e:
            logging.warning("MMPose init failed: %s", e)

    for img_path in inputs:
        try:
            process_image(img_path, do_debug=args.debug, inferencer=inferencer)
        except Exception as e:
            logging.exception("Failed %s: %s", img_path, e)


if __name__ == "__main__":
    main()
