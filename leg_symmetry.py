#!/usr/bin/env python3
"""
Minimal Horse Leg Symmetry Analyzer
----------------------------------
This simplified script keeps only the essential pipeline needed for your
use-case:

- Remove background at full resolution (no downscaling).
- Heuristically select the front leg (largest lower-half contour near image center).
- Find the cannon-bone centre-line and extend it to include the hoof rim.
- Split the leg by the centre-line, mark the non-dominant side green, and
  mark asymmetric extra pixels on the dominant side red.

This version removes AI/model dependencies and extra assignment logic
so it's easier to tune and reason about.
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
# Optional AI-based keypoint detector (MMPose). Import if available.
try:
    from mmpose.apis import MMPoseInferencer
except Exception:
    MMPoseInferencer = None

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

_DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
_depth_pipe = None

def get_depth_pipe():
    global _depth_pipe
    if _depth_pipe is None:
        logging.info("Loading Depth Anything V2 Small …")
        _depth_pipe = hf_pipeline("depth-estimation", model=_DEPTH_MODEL_ID, device=-1)
        logging.info("Depth Anything V2 ready.")
    return _depth_pipe

def estimate_depth(image_bgr: np.ndarray) -> np.ndarray:
    """Depth Anything V2 → float32 depth map [0,1].  1.0 = closest."""
    logging.info("Running Depth Anything V2 …")
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(rgb)
    result = get_depth_pipe()(pil_img)
    depth_pil = result["depth"]
    depth_np = np.array(depth_pil).astype(np.float32)
    h, w = image_bgr.shape[:2]
    if depth_np.shape[0] != h or depth_np.shape[1] != w:
        depth_np = cv2.resize(depth_np, (w, h), interpolation=cv2.INTER_LINEAR)
    d_min, d_max = depth_np.min(), depth_np.max()
    if d_max > d_min:
        depth_np = (depth_np - d_min) / (d_max - d_min)
    else:
        depth_np = np.zeros_like(depth_np)
    logging.info("Depth map ready (shape=%s).", depth_np.shape)
    return depth_np

def extract_foreground_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Run rembg at full resolution and return a cleaned binary mask.

    Uses a permissive alpha threshold and morphological ops to keep thin
    structures like fetlock and hoof rim.
    """
    h, w = image_bgr.shape[:2]
    logging.info("Running rembg on image at full resolution (%dx%d)", w, h)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgba = remove(rgb)
    if rgba is None or rgba.ndim < 3 or rgba.shape[2] < 4:
        raise RuntimeError("rembg returned unexpected result")
    alpha = rgba[:, :, 3]
    _, mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)

    # close and dilate to preserve hoof/fetlock connections
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    return mask.astype(np.uint8)


def extract_foreground_rgba(image_bgr: np.ndarray) -> tuple:
    """Run rembg and return (rgba_rgb, mask_uint8).

    `rgba_rgb` is an HxWx4 numpy array in RGB order as returned by rembg.
    """
    h, w = image_bgr.shape[:2]
    logging.info("Running rembg on image at full resolution (%dx%d)", w, h)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgba = remove(rgb)
    if rgba is None or rgba.ndim < 3 or rgba.shape[2] < 4:
        raise RuntimeError("rembg returned unexpected result")
    alpha = rgba[:, :, 3]
    _, mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)

    # close and dilate to preserve hoof/fetlock connections
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    return rgba, mask.astype(np.uint8)


def split_mask_on_width(mask_region: np.ndarray, min_narrow_frac: float = 0.6,
                        min_width_px: int = 30, debug: bool = False) -> list:
    """Split a mask at rows where width narrows sharply (touching-leg separation). Bug 7 fix."""
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
    median_w = int(np.median(nonzero))
    thresh = max(min_width_px, int(median_w * min_narrow_frac))
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
    parts = []
    for lab in range(1, num_labels):
        part = np.zeros_like(mask_region)
        part[labels == lab] = 255
        if cv2.countNonZero(part) > 50:
            parts.append(part)
    if not parts:
        return [mask_region]
    parts.sort(key=lambda m: cv2.countNonZero(m), reverse=True)
    if debug:
        logging.info("split_mask_on_width: %d parts (median_w=%d thresh=%d cuts=%s)",
                     len(parts), median_w, thresh, cut_rows)
    return parts


def select_front_leg_fallback(mask: np.ndarray, depth_map: np.ndarray | None = None,
                              debug: bool = False) -> np.ndarray | None:
    """Select the front leg contour from the lower half of the mask.

    Bug 2 fix: scores candidates by avg depth when depth_map is provided.
    Bug 3 fix: rejects tails via aspect-ratio and ground-reach filters.
    Bug 7 fix: calls split_mask_on_width to separate touching legs.
    """
    h, w = mask.shape
    zone = np.zeros_like(mask)
    zone[int(h * 0.35):] = mask[int(h * 0.35):]
    contours, _ = cv2.findContours(zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_cx = w / 2.0
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        # Bug 2/3 adjustment: measure depth first to use it as an override for shape filters
        avg_depth = 0.0
        if depth_map is not None:
            cm = np.zeros_like(mask)
            cv2.drawContours(cm, [cnt], -1, 255, cv2.FILLED)
            ov = (cm > 0) & (mask > 0)
            if np.any(ov):
                avg_depth = float(depth_map[ov].mean())

        # Bug 3: reject blobs that don't reach low enough (tail/body filter)
        # Relax requirement (80% -> 50% height) if it's very close to camera (high depth)
        # Lowered depth threshold to 0.40 because red/orange legs can score around ~0.50
        req_bottom = 0.80 * h if avg_depth < 0.40 else 0.50 * h
        if (y + ch) < int(req_bottom):
            if debug:
                logging.info("fallback reject (tail/body): bottom=%d < req=%.0f (depth=%.3f)", y + ch, req_bottom, avg_depth)
            continue
        # Bug 3: reject blobs that are too wide relative to height (not leg-shaped)
        # Relax requirement for frontmost objects (cropped legs can have squatter bounding boxes)
        max_ratio = 0.65 if avg_depth < 0.40 else 0.90
        if ch == 0 or (cw / ch) > max_ratio:
            if debug:
                logging.info("fallback reject (aspect): cw/ch=%.2f > req=%.2f (depth=%.3f)", cw / ch if ch else 999, max_ratio, avg_depth)
            continue
        cx_cnt = x + cw / 2.0
        dx = abs(cx_cnt - img_cx)
        scan_start = y + int(ch * 0.75)
        scan_end = min(h - 1, y + ch - 1)
        bottom_width = 0
        for ry in range(scan_start, scan_end + 1):
            xs_row = np.where(mask[ry] > 0)[0]
            if xs_row.size > 0:
                bottom_width = max(bottom_width, int(xs_row[-1] - xs_row[0]))
        
        if debug:
            logging.info("fallback candidate: area=%d bottom_w=%d dx=%.1f depth=%.3f",
                         area, bottom_width, dx, avg_depth)
        candidates.append((avg_depth, area, bottom_width, -dx, cnt))

    if not candidates:
        return None
    # Bug 2: sort by depth first, then area, then bottom_width, then proximity to center
    candidates.sort(key=lambda s: (s[0], s[1], s[2], s[3]), reverse=True)
    avg_depth, area, bottom_width, _, best_cnt = candidates[0]
    if area < 800 or bottom_width < max(20, int(0.12 * w)):
        if debug:
            logging.info("fallback reject best: area=%d bottom=%d", area, bottom_width)
        return None
    lm = np.zeros_like(mask)
    cv2.drawContours(lm, [best_cnt], -1, 255, cv2.FILLED)
    # Bug 7: split touching legs within the selected contour
    parts = split_mask_on_width(lm, min_narrow_frac=0.6,
                                min_width_px=max(20, int(0.06 * w)), debug=debug)
    if len(parts) > 1:
        if depth_map is not None:
            best_part, best_d = None, -1.0
            for part in parts:
                ov = (part > 0) & (mask > 0)
                d = float(depth_map[ov].mean()) if np.any(ov) else 0.0
                if d > best_d:
                    best_d, best_part = d, part
            if best_part is not None:
                lm = best_part
        else:
            lm = parts[0]  # largest by area
    logging.info("Selected front leg (area=%.0f depth=%.3f)", area, avg_depth)
    return lm


def select_front_leg_from_keypoints(mask: np.ndarray, knee: tuple[float, float], hoof: tuple[float, float], debug: bool = False) -> np.ndarray | None:
    """Select a leg contour that best matches the provided knee/hoof keypoints."""
    h, w = mask.shape
    # focus search starting slightly above the knee downwards
    start_y = max(0, int(round(knee[1])) - 40)
    zone = np.zeros_like(mask)
    zone[start_y:] = mask[start_y:]
    contours, _ = cv2.findContours(zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    hoof_pt = (float(hoof[0]), float(hoof[1]))
    best_cnt = None
    best_cnt = None
    best_score = None
    img_w = mask.shape[1]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:
            continue
        # prefer contours that contain/are near the hoof, but also near the knee x
        dist = cv2.pointPolygonTest(cnt, hoof_pt, True)
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            cx = bx + bw // 2
            cy = by + bh // 2
        # horizontal distance from knee
        dx_knee = abs(cx - int(round(knee[0])))
        # small penalty for being far from knee; prefer contours with positive dist (inside)
        score = (dist, -dx_knee, area)
        if debug:
            logging.info("candidate: area=%d dist=%.2f dx_knee=%d", area, dist, int(dx_knee))
        if best_score is None or score > best_score:
            best_score = score
            best_cnt = cnt

    if best_cnt is None:
        return None

    lm = np.zeros_like(mask)
    cv2.drawContours(lm, [best_cnt], -1, 255, cv2.FILLED)

    # Bug 7: use module-level split_mask_on_width instead of nested copy
    if cv2.countNonZero(lm) > (mask.shape[0] * mask.shape[1] * 0.02):
        parts = split_mask_on_width(lm, min_narrow_frac=0.6,
                                    min_width_px=max(20, int(0.06 * mask.shape[1])), debug=debug)
        if len(parts) > 1:
            best_part, best_dist = None, None
            kx = int(round(knee[0]))
            for part in parts:
                ys_p, xs_p = np.where(part > 0)
                if ys_p.size == 0:
                    continue
                pcx = int(np.mean(xs_p))
                d = abs(pcx - kx)
                if best_dist is None or d < best_dist:
                    best_dist = d
                    best_part = part
            if best_part is not None:
                lm = best_part
    # Restrict to vertical band exactly around knee->hoof
    ky = int(round(knee[1]))
    hy = int(round(hoof[1]))
    top_clip = max(0, ky - 20)
    bottom_clip = min(mask.shape[0] - 1, hy + 60)
    band = np.zeros_like(mask)
    band[top_clip: bottom_clip + 1, :] = 1
    lm = cv2.bitwise_and(lm, lm, mask=band.astype(np.uint8))
    lm = cv2.morphologyEx(lm, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    lm = cv2.dilate(lm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    hy_clamped = min(bottom_clip, mask.shape[0] - 1)
    xs_h = np.where(lm[hy_clamped] > 0)[0]
    if xs_h.size == 0:
        if debug:
            logging.info("rejecting candidate: no pixels at hoof row after clipping")
        return None
    bottom_width = xs_h[-1] - xs_h[0]
    if bottom_width < max(20, int(0.08 * mask.shape[1])):
        if debug:
            logging.info("rejecting candidate: bottom_width=%d too small", bottom_width)
        return None
    return lm


def seed_watershed_from_hooves(mask: np.ndarray, hooves: list[tuple[float, float]], rgba: np.ndarray | None = None, snap_radius: int = 30) -> list:
    """Segment `mask` into regions seeded at each hoof point using watershed.

    Returns list of uint8 masks (0/255) corresponding to each hoof seed in order.
    """
    if mask is None or mask.size == 0:
        return []
    bm = (mask > 0).astype(np.uint8) * 255

    h, w = bm.shape
    # helper: snap seed to nearest mask pixel
    def snap_to_mask(xf, yf):
        x = int(round(xf)); y = int(round(yf))
        if x < 0 or x >= w or y < 0 or y >= h:
            return None
        if bm[y, x] > 0:
            return (x, y)
        # search small neighborhood
        for r in range(1, snap_radius + 1):
            ys = range(max(0, y - r), min(h, y + r + 1))
            xs = range(max(0, x - r), min(w, x + r + 1))
            best = None
            bestd = None
            for yy in ys:
                for xx in xs:
                    if bm[yy, xx] > 0:
                        d = (xx - x) * (xx - x) + (yy - y) * (yy - y)
                        if bestd is None or d < bestd:
                            bestd = d
                            best = (xx, yy)
            if best is not None:
                return best
        return None

    markers = np.zeros((h, w), dtype=np.int32)
    seed_points = []
    for i, (kx, ky) in enumerate(hooves, start=1):
        s = snap_to_mask(kx, ky)
        if s is None:
            seed_points.append(None)
            continue
        sx, sy = s
        seed_points.append((sx, sy))
        cv2.circle(markers, (sx, sy), 6, i, -1)

    if all(s is None for s in seed_points):
        return []

    # build a topography for watershed. Prefer a foreground-weighted topo using
    # the RGBA/foreground colors when available: darker/lower-contrast hair can
    # be given higher elevation to encourage separation.
    if rgba is not None:
        try:
            # rgba is RGB order from rembg
            rgb_fg = rgba[..., :3]
            gray = cv2.cvtColor(rgb_fg, cv2.COLOR_RGB2GRAY)
            # invert brightness so darker pixels produce higher topo
            inv = (255 - gray).astype(np.float32) / 255.0
            dist = cv2.distanceTransform((bm // 255).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
            if dist.max() <= 0:
                return []
            topo = dist * (1.0 + 0.7 * inv)
            topo8 = np.uint8((topo / topo.max()) * 255.0)
            img3 = cv2.cvtColor(topo8, cv2.COLOR_GRAY2BGR)
            cv2.watershed(img3, markers)
        except Exception:
            # fallback to simple dist-based watershed
            dist = cv2.distanceTransform((bm // 255).astype(np.uint8), cv2.DIST_L2, 5)
            if dist.max() <= 0:
                return []
            dist8 = np.uint8((dist / dist.max()) * 255.0)
            img3 = cv2.cvtColor(dist8, cv2.COLOR_GRAY2BGR)
            try:
                cv2.watershed(img3, markers)
            except Exception:
                return []
    else:
        dist = cv2.distanceTransform((bm // 255).astype(np.uint8), cv2.DIST_L2, 5)
        if dist.max() <= 0:
            return []
        dist8 = np.uint8((dist / dist.max()) * 255.0)
        img3 = cv2.cvtColor(dist8, cv2.COLOR_GRAY2BGR)
        try:
            cv2.watershed(img3, markers)
        except Exception:
            return []

    parts = []
    for i in range(1, len(hooves) + 1):
        m = np.zeros_like(bm)
        m[markers == i] = 255
        # remove small specks
        if cv2.countNonZero(m) > 50:
            parts.append(m)
        else:
            parts.append(None)
    return parts
def get_ai_leg_keypoints(inferencer, image_path: str) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Run MMPose (if available) to return a list of (knee, hoof) pairs for detected front legs.
    Returns an empty list if MMPose unavailable or no confident detections.
    """
    if inferencer is None:
        return []
    try:
        res = inferencer(image_path)
    except Exception as e:
        logging.warning("MMPose inference failed: %s", e)
        return []

    # normalise to dict-like
    try:
        if hasattr(res, '__iter__') and not isinstance(res, dict):
            results = next(iter(res))
        else:
            results = res
    except Exception:
        results = res

    preds = None
    if isinstance(results, dict) and 'predictions' in results:
        if results['predictions']:
            preds = results['predictions'][0]
    elif isinstance(results, list) and results:
        preds = results[0]

    if not preds:
        return []

    kpts = None
    scores = None
    if isinstance(preds, dict):
        kpts = preds.get('keypoints') or preds.get('preds')
        scores = preds.get('keypoint_scores') or preds.get('scores')

    if kpts is None:
        return []

    # AP-10K mapping guesses: 6=knee,7=hoof (left), 9=knee,10=hoof (right)
    legs = []
    try:
        # convert flat arrays to (x,y) if necessary
        if isinstance(kpts, np.ndarray):
            kpts = kpts.tolist()
        if len(kpts) > 10:
            l_knee, l_hoof = tuple(kpts[6][:2]), tuple(kpts[7][:2])
            r_knee, r_hoof = tuple(kpts[9][:2]), tuple(kpts[10][:2])
            # scores if available
            s6 = float(scores[6]) if scores and len(scores) > 6 else 1.0
            s7 = float(scores[7]) if scores and len(scores) > 7 else 1.0
            s9 = float(scores[9]) if scores and len(scores) > 9 else 1.0
            s10 = float(scores[10]) if scores and len(scores) > 10 else 1.0
            if s6 > 0.12 and s7 > 0.12:
                legs.append((l_knee, l_hoof))
            if s9 > 0.12 and s10 > 0.12:
                legs.append((r_knee, r_hoof))
    except Exception:
        return []

    return legs


def find_cannon_bone_axis(leg_mask: np.ndarray, target_knee: tuple | None = None,
                          target_hoof: tuple | None = None) -> tuple:
    """Estimate cannon axis via least-squares line fit through cannon-zone midpoints.

    Bug 4 fix: returns (cx_at_pivot, axis_top_y, axis_bottom_y, pivot_y, slope)
    where slope is dx/dy so the axis line is drawn diagonally for angled legs.
    """
    h, w = leg_mask.shape
    clean = cv2.morphologyEx(leg_mask, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    ys = np.where(np.any(clean > 0, axis=1))[0]
    if ys.size == 0:
        return w // 2, 0, h - 1, h // 2, 0.0
    top_y = int(ys[0])
    bottom_y = int(ys[-1])

    rows = []
    for ry in range(top_y, bottom_y + 1):
        xs = np.where(clean[ry] > 0)[0]
        if xs.size >= 2:
            lx, rx = int(xs[0]), int(xs[-1])
            rows.append((ry, lx, rx, (lx + rx) / 2.0))

    if not rows:
        return w // 2, top_y, bottom_y, (top_y + bottom_y) // 2, 0.0

    leg_h = bottom_y - top_y + 1
    if target_knee is not None and target_hoof is not None:
        try:
            ai_h = int(target_hoof[1] - target_knee[1])
            cs_y = int(target_knee[1]) + int(ai_h * 0.10)
            ce_y = int(target_knee[1]) + int(ai_h * 0.40)
            cannon_rows = [r for r in rows if cs_y <= r[0] <= ce_y]
        except Exception:
            cannon_rows = []
    else:
        cs_y = top_y + int(leg_h * 0.10)
        ce_y = top_y + int(leg_h * 0.40)
        cannon_rows = [r for r in rows if cs_y <= r[0] <= ce_y]
    if not cannon_rows:
        cannon_rows = rows

    ry_arr = np.array([r[0] for r in cannon_rows], dtype=np.float64)
    cx_arr = np.array([r[3] for r in cannon_rows], dtype=np.float64)
    # Bug 4 fix: fit a line (slope, intercept) through cannon midpoints
    if len(cannon_rows) >= 2:
        slope, intercept = np.polyfit(ry_arr, cx_arr, 1)
    else:
        slope = 0.0
        intercept = float(cx_arr[0]) if len(cx_arr) > 0 else w / 2.0
    pivot_y = int(round(float(np.mean(ry_arr))))
    cx = int(round(slope * pivot_y + intercept))

    axis_top_y = max(top_y, int(cs_y))
    axis_bottom_y = min(h - 1, bottom_y + 120)
    all_lx = min(r[1] for r in rows)
    all_rx = max(r[2] for r in rows)
    cx = int(np.clip(cx, all_lx + 1, all_rx - 1))
    logging.info("Cannon axis: cx=%d pivot_y=%d top=%d bottom=%d slope=%.4f",
                 cx, pivot_y, axis_top_y, axis_bottom_y, slope)
    return cx, axis_top_y, axis_bottom_y, pivot_y, slope


def analyze_symmetry(leg_mask: np.ndarray, center_x: int, top_y: int, bottom_y: int,
                     slope: float = 0.0, pivot_y: int | None = None):
    """Row-by-row symmetry with angled-axis support.

    Bug 4 fix: effective center_x shifts per-row using fitted slope.
    Returns (green_mask, red_mask, dominant_side).
    """
    h, w = leg_mask.shape
    if pivot_y is None:
        pivot_y = (top_y + bottom_y) // 2
    green = np.zeros((h, w), dtype=np.uint8)
    red = np.zeros((h, w), dtype=np.uint8)

    total_left = 0
    total_right = 0
    rows = []
    for ry in range(top_y, min(bottom_y + 1, h)):
        ecx = int(round(center_x + slope * (ry - pivot_y)))
        ecx = max(1, min(w - 2, ecx))
        xs = np.where(leg_mask[ry] > 0)[0]
        if xs.size < 2:
            rows.append(None)
            continue
        lx, rx = int(xs[0]), int(xs[-1])
        if lx >= ecx or rx <= ecx:
            rows.append(None)
            continue
        lw = ecx - lx
        rw = rx - ecx
        total_left += lw
        total_right += rw
        rows.append((ry, lx, rx, lw, rw, ecx))

    if total_left > total_right * 1.02:
        dominant = "LEFT"
    elif total_right > total_left * 1.02:
        dominant = "RIGHT"
    else:
        dominant = "SYMMETRIC"

    for item in rows:
        if item is None:
            continue
        ry, lx, rx, lw, rw, ecx = item
        sw = min(lw, rw)
        if dominant == "LEFT":
            if ecx < rx + 1:
                green[ry, ecx: rx + 1] = leg_mask[ry, ecx: rx + 1]
            green[ry, max(0, ecx - sw): ecx] = leg_mask[ry, max(0, ecx - sw): ecx]
            if lw > sw:
                es, ee = lx, max(0, ecx - sw)
                if es < ee:
                    red[ry, es:ee] = leg_mask[ry, es:ee]
        elif dominant == "RIGHT":
            if lx < ecx:
                green[ry, lx: ecx] = leg_mask[ry, lx: ecx]
            green[ry, ecx: min(w, ecx + sw + 1)] = leg_mask[ry, ecx: min(w, ecx + sw + 1)]
            if rw > sw:
                es, ee = min(w, ecx + sw + 1), rx + 1
                if es < ee:
                    red[ry, es:ee] = leg_mask[ry, es:ee]
        else:
            green[ry, lx: rx + 1] = leg_mask[ry, lx: rx + 1]

    green = cv2.bitwise_and(green, leg_mask)
    red = cv2.bitwise_and(red, leg_mask)
    logging.info("Dominant side: %s (left=%d right=%d)", dominant, total_left, total_right)
    return green, red, dominant


def apply_overlay(img: np.ndarray, green_mask: np.ndarray, red_mask: np.ndarray, alpha: float = 0.55):
    res = img.astype(np.float32)
    orig = img.astype(np.float32)
    COLOR_GREEN = np.array([34, 197, 94], dtype=np.float32)
    COLOR_RED = np.array([48, 48, 220], dtype=np.float32)
    gm = green_mask > 0
    rm = red_mask > 0
    res[gm] = orig[gm] * (1 - alpha) + COLOR_GREEN * alpha
    res[rm] = orig[rm] * (1 - alpha) + COLOR_RED * alpha
    return np.clip(res, 0, 255).astype(np.uint8)


def process_image(path: str, do_debug: bool = False, inferencer=None) -> None:
    p = Path(path)
    img = cv2.imread(str(p))
    if img is None:
        logging.error("Cannot read %s", p)
        return
    h, w = img.shape[:2]
    logging.info("Processing %s (%dx%d)", p.name, w, h)

    rgba, mask = extract_foreground_rgba(img)
    fg_bgr = cv2.bitwise_and(img, img, mask=mask)

    # 1. Save foreground image
    cv2.imwrite(str(p.parent / f"{p.stem}_foreground.png"), fg_bgr)
    logging.info("Saved %s_foreground.png", p.stem)

    # 2. Depth estimation — Bug 1 fix: fill background with neutral gray so the
    #    depth model isn't misled by black zeros at the masked-out region.
    depth_input = img.copy()
    depth_input[mask == 0] = 128
    depth_map = estimate_depth(depth_input)
    depth_color = cv2.applyColorMap((depth_map * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.imwrite(str(p.parent / f"{p.stem}_depth.png"), depth_color)
    logging.info("Saved %s_depth.png", p.stem)

    # 3 & 4. Select front leg using depth + AI or depth-aware fallback
    leg_infos = []
    if inferencer is not None:
        try:
            legs = get_ai_leg_keypoints(inferencer, str(p))
        except Exception:
            legs = []
        if legs:
            # Bug 6 fix: use watershed to separate touching legs when >= 2 detected
            if len(legs) >= 2:
                hooves = [hoof for (_, hoof) in legs]
                ws_parts = seed_watershed_from_hooves(mask, hooves, rgba=rgba)
            else:
                ws_parts = []

            if ws_parts and any(pt is not None for pt in ws_parts):
                # Score each watershed segment by avg depth; pick the frontmost
                best_leg, best_d = None, -1.0
                for (knee, hoof), part in zip(legs, ws_parts):
                    if part is None:
                        continue
                    ov = (part > 0) & (mask > 0)
                    d = float(depth_map[ov].mean()) if np.any(ov) else 0.0
                    logging.info("Watershed leg (%.0f,%.0f)->(%.0f,%.0f) depth=%.3f",
                                 knee[0], knee[1], hoof[0], hoof[1], d)
                    if d > best_d:
                        best_d, best_leg = d, (knee, hoof, part)
                if best_leg is not None:
                    knee, hoof, lm = best_leg
                    leg_infos.append({'mask': lm, 'knee': knee, 'hoof': hoof})
            else:
                # Score each keypoint pair by depth along knee->hoof segment
                best_leg, best_d = None, -1.0
                for knee, hoof in legs:
                    lm_line = np.zeros((h, w), dtype=np.uint8)
                    cv2.line(lm_line, (int(knee[0]), int(knee[1])),
                             (int(hoof[0]), int(hoof[1])), 255, 5)
                    ov = (lm_line > 0) & (mask > 0)
                    d = float(depth_map[ov].mean()) if np.any(ov) else 0.0
                    logging.info("Leg (%.0f,%.0f)->(%.0f,%.0f) avg depth=%.3f",
                                 knee[0], knee[1], hoof[0], hoof[1], d)
                    if d > best_d:
                        best_d, best_leg = d, (knee, hoof)
                if best_leg is not None:
                    knee, hoof = best_leg
                    lm = select_front_leg_from_keypoints(mask, knee, hoof, debug=do_debug)
                    if lm is not None:
                        leg_infos.append({'mask': lm, 'knee': knee, 'hoof': hoof})

    # Bug 2 fix: fallback now receives depth_map for depth-based scoring
    if not leg_infos:
        logging.warning("AI path produced no leg; using depth-aware fallback.")
        lm = select_front_leg_fallback(mask, depth_map=depth_map, debug=do_debug)
        if lm is None:
            logging.warning("No front leg found for %s", p.name)
            cv2.imwrite(str(p.parent / f"{p.stem}_analyzed.jpg"), img)
            return
        leg_infos = [{'mask': lm, 'knee': (int(w / 2), 0), 'hoof': (int(w / 2), int(h - 1))}]

    # 4. Save isolated leg
    cv2.imwrite(str(p.parent / f"{p.stem}_isolated_leg.png"), leg_infos[0]['mask'])
    logging.info("Saved %s_isolated_leg.png", p.stem)

    # 5. Symmetry analysis
    h, w = img.shape[:2]
    combined_green = np.zeros((h, w), dtype=np.uint8)
    combined_red = np.zeros((h, w), dtype=np.uint8)
    draw_info = []  # Bug 5 fix: store per-leg draw data instead of relying on loop vars

    for info in leg_infos:
        leg_mask_i = info['mask']
        cx, ty, by, piv_y, slope = find_cannon_bone_axis(leg_mask_i)
        green, red, dominant = analyze_symmetry(leg_mask_i, cx, ty, by,
                                                slope=slope, pivot_y=piv_y)
        combined_green = np.maximum(combined_green, green)
        combined_red = np.maximum(combined_red, red)
        draw_info.append((cx, ty, by, piv_y, slope, dominant))

    out = apply_overlay(img, combined_green, combined_red, alpha=0.55)
    thickness = max(2, int(w * 0.004))

    # Bug 5 fix: render each leg's axis and label independently
    for i, (cx, ty, by, piv_y, slope, dominant) in enumerate(draw_info):
        x_top = int(round(cx + slope * (ty - piv_y)))
        x_bot = int(round(cx + slope * (by - piv_y)))
        cv2.line(out, (x_top, ty), (x_bot, by), (255, 80, 0), thickness)
        cv2.putText(out, f"Leg {i + 1} Dominant: {dominant}",
                    (10, 30 + i * 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imwrite(str(p.parent / f"{p.stem}_analyzed.jpg"), out)
    if do_debug:
        dbg = img.copy()
        dbg[combined_green > 0] = [34, 197, 94]
        dbg[combined_red > 0] = [48, 48, 220]
        for cx, ty, by, piv_y, slope, _ in draw_info:
            x_top = int(round(cx + slope * (ty - piv_y)))
            x_bot = int(round(cx + slope * (by - piv_y)))
            cv2.line(dbg, (x_top, ty), (x_bot, by), (255, 80, 0), thickness)
        cv2.imwrite(str(p.parent / f"{p.stem}_debug.png"), dbg)
    logging.info("Saved %s_analyzed.jpg", p.stem)


def main():
    parser = argparse.ArgumentParser(description="Minimal leg symmetry analyzer")
    parser.add_argument("images", nargs="+", help="Image files or glob patterns")
    parser.add_argument("--debug", action="store_true", help="Save intermediate debug images")
    parser.add_argument("--use-ai", action="store_true", help="Enable MMPose AI keypoint detection if available")
    parser.add_argument("--model-path", type=str, default=None, help="Optional local model path for MMPoseInferencer")
    parser.add_argument("--device", type=str, default=None, help="Device for MMPose (e.g., cpu or cuda:0)")
    args = parser.parse_args()

    inputs = sorted(set(f for pat in args.images for f in glob.glob(pat)))
    if not inputs:
        logging.error("No input images")
        sys.exit(1)

    inferencer = None
    if args.use_ai and MMPoseInferencer is not None:
        try:
            if args.model_path:
                inferencer = MMPoseInferencer(pose2d=args.model_path, device=args.device) if args.device else MMPoseInferencer(pose2d=args.model_path)
            else:
                inferencer = MMPoseInferencer(pose2d='rtmpose-m_8xb64-210e_ap10k-256x256')
            logging.info("MMPose inferencer initialized")
        except Exception as e:
            logging.warning("Failed to initialize MMPoseInferencer: %s", e)
            inferencer = None

    for img in inputs:
        try:
            process_image(img, do_debug=args.debug, inferencer=inferencer)
        except Exception as e:
            logging.exception("Failed processing %s: %s", img, e)


if __name__ == "__main__":
    main()
