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
from rembg import remove
# Optional AI-based keypoint detector (MMPose). Import if available.
try:
    from mmpose.apis import MMPoseInferencer
except Exception:
    MMPoseInferencer = None

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


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


def select_front_leg_fallback(mask: np.ndarray, debug: bool = False) -> np.ndarray | None:
    """Select the front leg by choosing the largest lower-half contour near center.

    Returns a binary mask for the selected leg or None if not found.
    """
    h, w = mask.shape
    zone = np.zeros_like(mask)
    # search from 35% downwards (include more of fetlock/cannon region)
    start_row = int(h * 0.35)
    zone[start_row:] = mask[start_row:]
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
        cx = x + cw / 2.0
        dx = abs(cx - img_cx)
        # compute bottom width of contour to prefer hoof-like shapes
        by = y + ch - 1
        ys = np.where(mask[y: y + ch] > 0)
        # approximate bottom width by scanning last 10% of bounding box rows
        scan_start = y + int(ch * 0.75)
        scan_end = y + ch - 1
        bottom_width = 0
        for ry in range(scan_start, scan_end + 1):
            xs_row = np.where(mask[ry] > 0)[0]
            if xs_row.size > 0:
                bottom_width = max(bottom_width, int(xs_row[-1] - xs_row[0]))
        # score: prefer larger area, larger bottom width, then closeness to center
        if debug:
            logging.info("fallback candidate: area=%d bottom_width=%d dx=%.1f", area, bottom_width, dx)
        candidates.append(((area, bottom_width, -dx), cnt, area, bottom_width))

    if not candidates:
        return None
    candidates.sort(key=lambda s: (s[0][0], s[0][1], s[0][2]), reverse=True)
    best = candidates[0][1]
    if candidates[0][2] < 800 or candidates[0][3] < max(20, int(0.12 * w)):
        if debug:
            logging.info("fallback reject best area=%d bottom=%d", candidates[0][2], candidates[0][3])
        return None
    lm = np.zeros_like(mask)
    cv2.drawContours(lm, [best], -1, 255, cv2.FILLED)
    logging.info("Selected front leg (area=%.0f)", candidates[0][2])
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

    # If the contour is large, attempt to split touching legs by looking for
    # narrow "waist" rows (local minima of width) and cutting there.
    def split_mask_on_width(mask_region: np.ndarray, min_narrow_frac: float = 0.6, min_width_px: int = 30, debug_loc=False):
        # returns list of masks (same size as mask_region) for each component after splitting
        ys = np.where(np.any(mask_region > 0, axis=1))[0]
        if ys.size == 0:
            return [mask_region]
        y0, y1 = int(ys[0]), int(ys[-1])
        widths = np.zeros(y1 - y0 + 1, dtype=np.int32)
        lx_list = np.zeros_like(widths)
        rx_list = np.zeros_like(widths)
        for i, ry in enumerate(range(y0, y1 + 1)):
            xs = np.where(mask_region[ry] > 0)[0]
            if xs.size >= 2:
                lx_list[i] = xs[0]
                rx_list[i] = xs[-1]
                widths[i] = xs[-1] - xs[0]
            else:
                widths[i] = 0

        nonzero = widths[widths > 0]
        if nonzero.size == 0:
            return [mask_region]
        median_w = int(np.median(nonzero))
        thresh = max(min_width_px, int(median_w * min_narrow_frac))
        # find rows where width < thresh
        narrow = widths < thresh
        # identify contiguous narrow zones (at least 1 row)
        cut_rows = []
        i = 0
        while i < len(narrow):
            if narrow[i]:
                j = i
                while j + 1 < len(narrow) and narrow[j + 1]:
                    j += 1
                # choose middle row as cut
                cut_rows.append(y0 + (i + j) // 2)
                i = j + 1
            else:
                i += 1

        if not cut_rows:
            return [mask_region]

        split_mask = mask_region.copy()
        pad = 2
        for r in cut_rows:
            a = max(y0, r - pad)
            b = min(y1, r + pad)
            split_mask[a:b+1, :] = 0
        # connected components
        num_labels, labels = cv2.connectedComponents(split_mask)
        parts = []
        for lab in range(1, num_labels):
            part = np.zeros_like(mask_region)
            part[labels == lab] = 255
            if cv2.countNonZero(part) > 50:
                parts.append(part)
        if not parts:
            return [mask_region]
        # sort by area desc
        parts.sort(key=lambda m: cv2.countNonZero(m), reverse=True)
        if debug_loc:
            logging.info("split into %d parts (median_w=%d thresh=%d cuts=%s)", len(parts), median_w, thresh, cut_rows)
        return parts

    # attempt splitting if it's large
    if cv2.countNonZero(lm) > (mask.shape[0] * mask.shape[1] * 0.02):
        parts = split_mask_on_width(lm, min_narrow_frac=0.6, min_width_px=max(20, int(0.06 * mask.shape[1])), debug_loc=debug)
        if len(parts) > 1:
            # pick part whose centroid is nearest to knee x (prefer front)
            best_part = None
            best_dist = None
            kx = int(round(knee[0]))
            for part in parts:
                ys, xs = np.where(part > 0)
                if ys.size == 0:
                    continue
                cx = int(np.mean(xs))
                dist = abs(cx - kx)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_part = part
            if best_part is not None:
                lm = best_part
    # Restrict to vertical band around knee->hoof to avoid hair/background merging
    ky = int(round(knee[1]))
    hy = int(round(hoof[1]))
    # increase top padding to include fetlock/cannon region
    top_clip = max(0, ky - 80)
    bottom_clip = min(mask.shape[0] - 1, hy + 120)
    band = np.zeros_like(mask)
    band[top_clip: bottom_clip + 1, :] = 1
    lm = cv2.bitwise_and(lm, lm, mask=band.astype(np.uint8))
    # close small gaps and dilate to include hoof rim
    lm = cv2.morphologyEx(lm, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    lm = cv2.dilate(lm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    # ensure the selected mask has a reasonable bottom width (avoid hair-like shapes)
    hy_clamped = min(bottom_clip, mask.shape[0] - 1)
    xs = np.where(lm[hy_clamped] > 0)[0]
    if xs.size == 0:
        if debug:
            logging.info("rejecting candidate: no pixels at hoof row after clipping")
        return None
    bottom_width = xs[-1] - xs[0]
    if bottom_width < max(20, int(0.08 * mask.shape[1])):
        # too narrow at hoof level — likely hair or small background piece
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


def find_cannon_bone_axis(leg_mask: np.ndarray, target_knee: tuple[float, float] | None = None, target_hoof: tuple[float, float] | None = None) -> tuple[int, int, int]:
    """Estimate the cannon centre X and axis range from the leg mask.

    Returns (center_x, axis_top_y, axis_bottom_y).
    """
    h, w = leg_mask.shape
    # smooth small holes
    clean = cv2.morphologyEx(leg_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    ys = np.where(np.any(clean > 0, axis=1))[0]
    if ys.size == 0:
        return w // 2, 0, h - 1
    top_y = int(ys[0])
    bottom_y = int(ys[-1])

    # collect midpoints per row
    rows = []
    for ry in range(top_y, bottom_y + 1):
        xs = np.where(clean[ry] > 0)[0]
        if xs.size >= 2:
            lx, rx = int(xs[0]), int(xs[-1])
            mid = (lx + rx) / 2.0
            wid = rx - lx
            rows.append((ry, lx, rx, mid, wid))

    if not rows:
        return w // 2, top_y, bottom_y

    # If AI keypoints available, prefer a cannon zone defined relative to knee->hoof
    cannon_start_y = None
    if target_knee is not None and target_hoof is not None:
        try:
            leg_height_ai = int(target_hoof[1] - target_knee[1])
            cannon_start_y = int(target_knee[1]) + int(leg_height_ai * 0.10)
            cannon_end_y = int(target_knee[1]) + int(leg_height_ai * 0.40)
            cannon_rows = [r for r in rows if cannon_start_y <= r[0] <= cannon_end_y]
        except Exception:
            cannon_rows = []
    else:
        # choose cannon zone as 10%-40% of leg height (shaft region)
        leg_h = bottom_y - top_y + 1
        s_y = top_y + int(leg_h * 0.10)
        e_y = top_y + int(leg_h * 0.40)
        cannon_rows = [r for r in rows if s_y <= r[0] <= e_y]
    if not cannon_rows:
        cannon_rows = rows

    centres = np.array([r[3] for r in cannon_rows], dtype=np.float32)
    cx = int(round(float(np.median(centres))))

    # Use cannon_start_y (if available) as axis top to ensure centre-line runs through cannon
    if cannon_start_y is not None:
        axis_top_y = max(top_y, cannon_start_y)
    else:
        axis_top_y = top_y
    axis_bottom_y = min(h - 1, bottom_y + 120)
    # clamp cx inside bounds
    all_lx = min(r[1] for r in rows)
    all_rx = max(r[2] for r in rows)
    cx = int(np.clip(cx, all_lx + 1, all_rx - 1))
    logging.info("Cannon axis X=%d top=%d bottom=%d", cx, axis_top_y, axis_bottom_y)
    return cx, axis_top_y, axis_bottom_y


def analyze_symmetry(leg_mask: np.ndarray, center_x: int, top_y: int, bottom_y: int):
    """Row-by-row symmetry; returns (green_mask, red_mask, dominant_side)."""
    h, w = leg_mask.shape
    green = np.zeros((h, w), dtype=np.uint8)
    red = np.zeros((h, w), dtype=np.uint8)

    total_left = 0
    total_right = 0
    rows = []
    for ry in range(top_y, min(bottom_y + 1, h)):
        xs = np.where(leg_mask[ry] > 0)[0]
        if xs.size < 2:
            rows.append(None)
            continue
        lx, rx = int(xs[0]), int(xs[-1])
        if lx >= center_x or rx <= center_x:
            rows.append(None)
            continue
        lw = center_x - lx
        rw = rx - center_x
        total_left += lw
        total_right += rw
        rows.append((ry, lx, rx, lw, rw))

    if total_left > total_right * 1.02:
        dominant = "LEFT"
    elif total_right > total_left * 1.02:
        dominant = "RIGHT"
    else:
        dominant = "SYMMETRIC"

    for item in rows:
        if item is None:
            continue
        ry, lx, rx, lw, rw = item
        sw = min(lw, rw)
        if dominant == "LEFT":
            # non-dominant right entirely green
            if center_x < rx + 1:
                green[ry, center_x: rx + 1] = leg_mask[ry, center_x: rx + 1]
            # matched left portion -> green
            if center_x - sw < center_x:
                green[ry, max(0, center_x - sw): center_x] = leg_mask[ry, max(0, center_x - sw): center_x]
            # leftover left -> red
            if lw > sw:
                extra_start = lx
                extra_end = max(0, center_x - sw)
                if extra_start < extra_end:
                    red[ry, extra_start:extra_end] = leg_mask[ry, extra_start:extra_end]
        elif dominant == "RIGHT":
            if lx < center_x:
                green[ry, lx: center_x] = leg_mask[ry, lx: center_x]
            if center_x < center_x + sw + 1:
                green[ry, center_x: min(w, center_x + sw + 1)] = leg_mask[ry, center_x: min(w, center_x + sw + 1)]
            if rw > sw:
                extra_start = min(w, center_x + sw + 1)
                extra_end = rx + 1
                if extra_start < extra_end:
                    red[ry, extra_start:extra_end] = leg_mask[ry, extra_start:extra_end]
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
    if do_debug:
        cv2.imwrite(str(p.parent / f"{p.stem}_silhouette.png"), mask)
        cv2.imwrite(str(p.parent / f"{p.stem}_foreground.png"), cv2.bitwise_and(img, img, mask=mask))

    # Try AI keypoints if inferencer provided
    leg_masks = []
    leg_infos = []
    if inferencer is not None:
        try:
            legs = get_ai_leg_keypoints(inferencer, str(p))
        except Exception:
            legs = []
        if legs:
            # try hoof-seeded watershed segmentation to separate touching silhouettes
            hooves = [hoof for (_, hoof) in legs]
            parts = seed_watershed_from_hooves(mask, hooves, rgba=rgba)
            if parts:
                for idx, ((knee, hoof), part) in enumerate(zip(legs, parts)):
                    if part is None:
                        continue
                    leg_masks.append(part)
                    leg_infos.append({'mask': part, 'knee': knee, 'hoof': hoof})
                    if do_debug:
                        try:
                            cv2.imwrite(str(p.parent / f"{p.stem}_leg_isolated_{idx}.png"), cv2.bitwise_and(img, img, mask=part))
                        except Exception:
                            pass
            else:
                # fallback to per-keypoint contour selection
                for knee, hoof in legs:
                    lm = select_front_leg_from_keypoints(mask, knee, hoof, debug=do_debug)
                    if lm is not None:
                        leg_masks.append(lm)
                        leg_infos.append({'mask': lm, 'knee': knee, 'hoof': hoof})
    # If no AI legs found, fallback to heuristic single-leg selection
    if not leg_masks:
        lm = select_front_leg_fallback(mask, debug=do_debug)
        if lm is None:
            logging.warning("No front leg found for %s", p.name)
            cv2.imwrite(str(p.parent / f"{p.stem}_analyzed.jpg"), img)
            return
        leg_masks = [lm]
        leg_infos = [{'mask': lm, 'knee': (int(w/2), 0), 'hoof': (int(w/2), int(h-1))}]

    # Process each detected leg and combine overlays
    h, w = img.shape[:2]
    combined_green = np.zeros((h, w), dtype=np.uint8)
    combined_red = np.zeros((h, w), dtype=np.uint8)
    debug_lines = []
    labels = []
    for info in leg_infos:
        leg_mask = info['mask']
        cx, ty, by = find_cannon_bone_axis(leg_mask)
        green, red, dominant = analyze_symmetry(leg_mask, cx, ty, by)
        combined_green = np.maximum(combined_green, green)
        combined_red = np.maximum(combined_red, red)
        debug_lines.append((cx, ty, by))
        labels.append(dominant)

    out = apply_overlay(img, combined_green, combined_red, alpha=0.55)

    # draw centre line and label
    cv2.line(out, (cx, ty), (cx, by), (255, 80, 0), max(2, int(w * 0.004)))
    cv2.putText(out, f"Front Leg Dominant: {dominant}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imwrite(str(p.parent / f"{p.stem}_analyzed.jpg"), out)
    if do_debug:
        dbg = img.copy()
        dbg[green > 0] = [34, 197, 94]
        dbg[red > 0] = [48, 48, 220]
        cv2.line(dbg, (cx, ty), (cx, by), (255, 80, 0), max(2, int(w * 0.004)))
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
