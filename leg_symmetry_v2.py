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
from mmpose.apis import MMPoseInferencer


logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

_DEPTH_MODEL_ID = depth-anythingDepth-Anything-V2-Small-hf
_depth_pipe = None


# ---------------------------------------------------------------------------
# Depth estimation
# ---------------------------------------------------------------------------

def get_depth_pipe()
    global _depth_pipe
    if _depth_pipe is None
        logging.info(Loading Depth Anything V2 Small …)
        _depth_pipe = hf_pipeline(depth-estimation, model=_DEPTH_MODEL_ID, device=-1)
        logging.info(Depth Anything V2 ready.)
    return _depth_pipe


def estimate_depth(image_bgr np.ndarray,
                   fg_mask np.ndarray  None = None) - np.ndarray
    Depth Anything V2 → float32 depth map [0, 1].  1.0 = closest.

    FIX #1 Background pixels are filled with neutral gray (127) before
    inference so the model is not confused by black zeros.  After inference,
    depth outside fg_mask is zeroed so only horse pixels contribute to scoring.
    
    h, w = image_bgr.shape[2]

    if fg_mask is not None
        input_img = image_bgr.copy()
        input_img[fg_mask == 0] = 127
    else
        input_img = image_bgr

    logging.info(Running Depth Anything V2 …)
    rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(rgb)
    result = get_depth_pipe()(pil_img)
    depth_np = np.array(result[depth]).astype(np.float32)

    if depth_np.shape[0] != h or depth_np.shape[1] != w
        depth_np = cv2.resize(depth_np, (w, h), interpolation=cv2.INTER_LINEAR)

    d_min, d_max = depth_np.min(), depth_np.max()
    depth_np = (depth_np - d_min)  (d_max - d_min) if d_max  d_min else np.zeros_like(depth_np)

    if fg_mask is not None
        depth_np[fg_mask == 0] = 0.0

    logging.info(Depth map ready (shape=%s)., depth_np.shape)
    return depth_np


# ---------------------------------------------------------------------------
# Foreground extraction
# ---------------------------------------------------------------------------

def extract_foreground_rgba(image_bgr np.ndarray) - tuple
    Run rembg and return (rgba HxWx4, mask uint8).

    FIX #11 saves RGBA with transparency instead of black background.
    FIX v3 After rembg, small disconnected components far from the image
    bottom-centre are removed. These are almost always tail strands or
    background artifacts that confuse the leg selector.
    
    h, w = image_bgr.shape[2]
    logging.info(Running rembg on image at full resolution (%dx%d), w, h)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgba = remove(rgb)
    if rgba is None or rgba.ndim  3 or rgba.shape[2]  4
        raise RuntimeError(rembg returned unexpected result)
    alpha = rgba[, , 3]
    _, mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)

    # FIX v3 Remove spurious small components (tail wisps, background noise).
    # Keep only connected components that are large enough relative to the
    # biggest component (assumed to be the horse bodyleg).
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    if num_labels  2  # 0=background, 1+=foreground components
        areas = stats[1, cv2.CC_STAT_AREA]  # exclude background label 0
        max_area = int(areas.max())
        clean_mask = np.zeros_like(mask)
        for lab in range(1, num_labels)
            comp_area = stats[lab, cv2.CC_STAT_AREA]
            # Keep component if it's at least 8% of the largest component
            if comp_area = max_area  0.08
                clean_mask[labels == lab] = 255
        if cv2.countNonZero(clean_mask)  cv2.countNonZero(mask)  0.50
            mask = clean_mask
            logging.info(Removed %d small spurious rembg components.,
                         num_labels - 1 - np.sum(
                             stats[1, cv2.CC_STAT_AREA] = max_area  0.08))

    return rgba, mask.astype(np.uint8)


# ---------------------------------------------------------------------------
# FIX #7 — split_mask_on_width (module level)
# ---------------------------------------------------------------------------

def split_mask_on_width(mask_region np.ndarray,
                        min_narrow_frac float = 0.6,
                        min_width_px int = 30,
                        debug bool = False) - list[np.ndarray]
    Split a mask at narrow 'waist' rows; return component masks sorted by area desc.
    ys = np.where(np.any(mask_region  0, axis=1))[0]
    if ys.size == 0
        return [mask_region]
    y0, y1 = int(ys[0]), int(ys[-1])

    widths = np.zeros(y1 - y0 + 1, dtype=np.int32)
    for i, ry in enumerate(range(y0, y1 + 1))
        xs = np.where(mask_region[ry]  0)[0]
        widths[i] = int(xs[-1] - xs[0]) if xs.size = 2 else 0

    nonzero = widths[widths  0]
    if nonzero.size == 0
        return [mask_region]
    median_w = int(np.median(nonzero))
    thresh = max(min_width_px, int(median_w  min_narrow_frac))

    narrow = widths  thresh
    cut_rows = []
    i = 0
    while i  len(narrow)
        if narrow[i]
            j = i
            while j + 1  len(narrow) and narrow[j + 1]
                j += 1
            cut_rows.append(y0 + (i + j)  2)
            i = j + 1
        else
            i += 1

    if not cut_rows
        return [mask_region]

    split = mask_region.copy()
    pad = 2
    for r in cut_rows
        split[max(y0, r - pad) min(y1, r + pad) + 1, ] = 0

    num_labels, labels = cv2.connectedComponents(split)
    parts = []
    for lab in range(1, num_labels)
        part = np.zeros_like(mask_region)
        part[labels == lab] = 255
        if cv2.countNonZero(part)  50
            parts.append(part)

    if not parts
        return [mask_region]
    parts.sort(key=lambda m cv2.countNonZero(m), reverse=True)
    if debug
        logging.info(split_mask_on_width %d parts (median_w=%d thresh=%d cuts=%s),
                     len(parts), median_w, thresh, cut_rows)
    return parts


# ---------------------------------------------------------------------------
# FIX #9 (ACTIVATED) — depth-based foreground prefilter
# ---------------------------------------------------------------------------

def depth_prefilter_mask(mask np.ndarray,
                          depth_map np.ndarray,
                          percentile float = 25.0) - np.ndarray
    Remove farback-leg pixels from the foreground mask before leg selection.

    Keeps foreground pixels whose depth = the Nth percentile of foreground
    depths.  In front-on shots this drops the back leg (blue in TURBO) while
    keeping the front leg (orangered).

    FIX #12 — percentile lowered from 50 → 25.
    When the camera is at ground level pointing upward the hoof is the closest
    point (depth ≈ 1.0) while the cannon bone is further away (depth ≈ 0.4–0.6).
    A 50th-percentile threshold sits exactly at the hoofcannon-bone boundary,
    stripping the cannon bone entirely.  25th-percentile keeps the vast majority
    of the front leg while still discarding the clearly-far back leg pixels.

    FIX #13 — height-preservation safety guard.
    The existing pixel-count guard ( 15 %) does not catch the case where the
    top of the leg is cut off (cannon bone has low pixel count relative to the
    wide hoof).  An additional check compares the bounding-box HEIGHT of the
    filtered mask to the original if height shrinks by more than 35 % the
    filter is discarding the top of the leg, so the original is returned.
    
    fg_depths = depth_map[mask  0]
    if fg_depths.size == 0
        return mask

    # Record original bounding-box height for the height-safety check
    ys_orig = np.where(np.any(mask  0, axis=1))[0]
    orig_height = int(ys_orig[-1] - ys_orig[0]) if ys_orig.size = 2 else 0

    thresh = float(np.percentile(fg_depths, percentile))
    filtered = np.zeros_like(mask)
    filtered[(mask  0) & (depth_map = thresh)] = 255

    # Close small gaps so the kept region stays contiguous
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, k)

    # Safety guard 1 pixel count (original)
    if cv2.countNonZero(filtered)  cv2.countNonZero(mask)  0.15
        logging.warning(depth_prefilter_mask pixel count too small — returning original mask.)
        return mask

    # Safety guard 2 (FIX #13  v3) height preservation
    # FIX v3 threshold relaxed from 65% → 50%. The old 65% guard was
    # too aggressive — it fell back to the full (back-leg-inclusive) mask
    # too often. 50% still catches catastrophic top-crop while allowing
    # the filter to do its job of removing the back leg.
    if orig_height  0
        ys_filt = np.where(np.any(filtered  0, axis=1))[0]
        if ys_filt.size = 2
            filt_height = int(ys_filt[-1] - ys_filt[0])
            if filt_height  orig_height  0.50
                logging.warning(
                    depth_prefilter_mask height shrank to %.0f%% (orig=%d filt=%d) 
                    — top of leg cut off; returning original mask.,
                    100.0  filt_height  orig_height, orig_height, filt_height)
                return mask

    logging.info(depth_prefilter_mask kept %.1f%% of fg pixels (thresh depth=%.3f),
                 100.0  cv2.countNonZero(filtered)  max(1, cv2.countNonZero(mask)), thresh)
    return filtered


# ---------------------------------------------------------------------------
# FIX #3 — tail rejection helper
# ---------------------------------------------------------------------------

def is_likely_tail(contour np.ndarray, mask np.ndarray) - bool
    Return True if contour resembles a tail rather than a leg.

    Improvements over v2
    - Stricter aspect ratio threshold (4 instead of 6)
    - Narrower width fraction threshold (0.10 instead of 0.07)
    - Position check tail blobs attached to upper portion of image
    - Stronger width-taper check (tapers significantly top→bottom)
    - Extra check blob center-X far from image horizontal center
    
    x, y, cw, ch = cv2.boundingRect(contour)
    h_img, w_img = mask.shape
    if ch == 0
        return False

    aspect = ch  max(cw, 1)

    # Stricter taller + thinner than before
    if aspect  4 and cw  w_img  0.10
        return True

    # Tail typically exits from top of frame or near top edge
    # If the blob starts in the top 30% AND is very narrow, it's likely tail
    if y  h_img  0.30 and cw  w_img  0.12 and aspect  3
        return True

    # Sample widths in top 20% and bottom 20% of this bounding box
    top_end = y + max(1, int(ch  0.20))
    bot_start = y + int(ch  0.80)
    top_ws, bot_ws, mid_ws = [], [], []
    mid_start = y + int(ch  0.35)
    mid_end = y + int(ch  0.65)

    for ry in range(y, min(top_end + 1, h_img))
        xs = np.where(mask[ry]  0)[0]
        if xs.size = 2
            top_ws.append(int(xs[-1] - xs[0]))
    for ry in range(bot_start, min(y + ch + 1, h_img))
        xs = np.where(mask[ry]  0)[0]
        if xs.size = 2
            bot_ws.append(int(xs[-1] - xs[0]))
    for ry in range(mid_start, min(mid_end + 1, h_img))
        xs = np.where(mask[ry]  0)[0]
        if xs.size = 2
            mid_ws.append(int(xs[-1] - xs[0]))

    if top_ws and bot_ws
        avg_top = float(np.mean(top_ws))
        avg_bot = float(np.mean(bot_ws))
        # Stronger taper ratio tail tapers more aggressively top→bottom
        if avg_bot  avg_top  0.40 and avg_bot  w_img  0.05
            return True
        # Tail is roughly uniform width and very narrow throughout
        if avg_top  w_img  0.06 and avg_bot  w_img  0.06
            return True

    # A leg widens at the hoof (bottom). If mid is the narrowest point AND
    # bottom is also narrow (no hoof flare), it's probably tail not leg.
    if mid_ws and bot_ws and top_ws
        avg_mid = float(np.mean(mid_ws))
        avg_bot = float(np.mean(bot_ws))
        avg_top = float(np.mean(top_ws))
        # Tail roughly uniform or tapering, no flare at bottom
        # Leg pastern narrows then hoof flares — bot_w  mid_w
        no_hoof_flare = avg_bot  avg_mid  1.10
        uniformly_thin = avg_top  w_img  0.09 and avg_mid  w_img  0.09
        if no_hoof_flare and uniformly_thin
            return True

    return False


# ---------------------------------------------------------------------------
# Leg selection — fallback (no MMPose)
# ---------------------------------------------------------------------------

def select_front_leg_fallback(mask np.ndarray,
                               depth_map np.ndarray  None = None,
                               debug bool = False) - np.ndarray  None
    Select the frontmost front leg using depth + heuristics.

    Receives depth-filtered mask (FIX #9) so back-leg pixels are already
    removed before this function runs.

    FIX v3 Raise split threshold so the fetlock waist does NOT cause the
    hoof to be split off as a separate component. min_narrow_frac raised
    from 0.55 → 0.35 so only truly narrow connections (separate legs) are
    cut, not the pastern-to-hoof transition.

    FIX v3 After candidate selection, if the chosen mask does not reach
    within 5% of the image bottom and there are foreground pixels below it
    that were filtered out, grow the mask downward to recover the hoof.
    
    h, w = mask.shape
    zone = np.zeros_like(mask)
    zone[int(h  0.20)] = mask[int(h  0.20)]

    raw_contours, _ = cv2.findContours(zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not raw_contours
        return None

    combined = np.zeros_like(mask)
    for cnt in raw_contours
        if cv2.contourArea(cnt) = 500
            cv2.drawContours(combined, [cnt], -1, 255, cv2.FILLED)

    # FIX v3 use 0.35 instead of 0.55 — only split truly separate legs,
    # not the natural narrowing at fetlockpastern junction.
    parts = split_mask_on_width(combined,
                                min_narrow_frac=0.35,
                                min_width_px=max(20, int(0.05  w)),
                                debug=debug)

    if len(parts) == 1 and cv2.countNonZero(parts[0])  (h  w  0.03)
        hooves = []
        for cnt in raw_contours
            if cv2.contourArea(cnt)  500
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            hooves.append((float(bx + bw  2), float(min(by + bh + 20, h - 1))))
        if len(hooves) = 2
            ws_parts = seed_watershed_from_hooves(combined, hooves)
            ws_parts = [p for p in ws_parts if p is not None and cv2.countNonZero(p)  50]
            if len(ws_parts)  len(parts)
                logging.info(Watershed separated %d parts from touching-leg blob, len(ws_parts))
                parts = ws_parts

    img_cx = w  2.0
    candidates = []
    for part in parts
        cnts, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts
            continue
        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area  500
            continue

        if is_likely_tail(cnt, part)
            if debug
                logging.info(Rejected tail-like contour (area=%.0f), area)
            continue

        bx, by, bw, bh = cv2.boundingRect(cnt)
        cx_part = bx + bw  2.0
        dx = abs(cx_part - img_cx)

        bottom_width = 0
        for ry in range(by + int(bh  0.75), min(by + bh + 1, h))
            xs = np.where(part[ry]  0)[0]
            if xs.size  0
                bottom_width = max(bottom_width, int(xs[-1] - xs[0]))

        avg_depth = 0.0
        if depth_map is not None
            fg_px = part  0
            if np.any(fg_px)
                avg_depth = float(depth_map[fg_px].mean())

        if debug
            logging.info(Candidate area=%.0f bottom_w=%d dx=%.1f depth=%.3f,
                         area, bottom_width, dx, avg_depth)
        candidates.append({
            'mask' part, 'area' area,
            'bottom_width' bottom_width, 'dx' dx, 'avg_depth' avg_depth,
        })

    # FIX v3 lower bottom_width threshold from 0.12 → 0.06 of image width
    # since the split at fetlock may have made the bottom part narrow.
    # Primary filter is still area = 800.
    candidates = [c for c in candidates
                  if c['area'] = 800 and c['bottom_width'] = max(15, int(0.06  w))]
    if not candidates
        return None

    candidates.sort(key=lambda c (c['avg_depth'], c['area']), reverse=True)
    best = candidates[0]
    logging.info(Selected front leg (area=%.0f depth=%.3f), best['area'], best['avg_depth'])

    selected_mask = best['mask']

    # FIX v3 Hoof recovery — if the selected mask doesn't reach near the
    # image bottom but there are foreground pixels below it in the raw mask,
    # bridge the gap with morphological dilation so the hoof is included.
    ys_sel = np.where(np.any(selected_mask  0, axis=1))[0]
    if ys_sel.size  0
        sel_bottom = int(ys_sel[-1])
        # Check if there are fg pixels in the original mask below our selection
        below_mask = mask[sel_bottom, ]
        if np.any(below_mask  0) and sel_bottom  h  0.90
            # Dilate downward to absorb the disconnected hoof region
            kernel_tall = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 40))
            grown = cv2.dilate(selected_mask, kernel_tall, iterations=3)
            # Only keep pixels that are also in the original foreground mask
            grown = cv2.bitwise_and(grown, mask)
            if cv2.countNonZero(grown)  cv2.countNonZero(selected_mask)
                logging.info(Hoof recovery grew mask from bottom_y=%d, sel_bottom)
                selected_mask = grown

    return selected_mask


# ---------------------------------------------------------------------------
# Leg selection — AI path (MMPose keypoints)
# ---------------------------------------------------------------------------

def select_front_leg_from_keypoints(mask np.ndarray,
                                    knee tuple[float, float],
                                    hoof tuple[float, float],
                                    debug bool = False) - np.ndarray  None
    Select a leg contour that best matches provided kneehoof keypoints.

    Receives depth-filtered mask (FIX #9).
    
    h, w = mask.shape
    start_y = max(0, int(round(knee[1])) - 40)
    zone = np.zeros_like(mask)
    zone[start_y] = mask[start_y]
    contours, _ = cv2.findContours(zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours
        return None

    hoof_pt = (float(hoof[0]), float(hoof[1]))
    best_cnt, best_score = None, None
    for cnt in contours
        area = cv2.contourArea(cnt)
        if area  200
            continue
        dist = cv2.pointPolygonTest(cnt, hoof_pt, True)
        M = cv2.moments(cnt)
        if M['m00'] != 0
            cx = int(M['m10']  M['m00'])
        else
            bx, by, bw, bh = cv2.boundingRect(cnt)
            cx = bx + bw  2
        dx_knee = abs(cx - int(round(knee[0])))
        score = (dist, -dx_knee, area)
        if debug
            logging.info(candidate area=%d dist=%.2f dx_knee=%d, area, dist, int(dx_knee))
        if best_score is None or score  best_score
            best_score = score
            best_cnt = cnt

    if best_cnt is None
        return None

    lm = np.zeros_like(mask)
    cv2.drawContours(lm, [best_cnt], -1, 255, cv2.FILLED)

    if cv2.countNonZero(lm)  (mask.shape[0]  mask.shape[1]  0.02)
        parts = split_mask_on_width(lm,
                                    min_narrow_frac=0.6,
                                    min_width_px=max(20, int(0.06  mask.shape[1])),
                                    debug=debug)
        if len(parts)  1
            kx = int(round(knee[0]))
            best_part = min(
                parts,
                key=lambda p abs(int(np.mean(np.where(p  0)[1])) - kx)
                              if np.any(p  0) else float('inf')
            )
            lm = best_part

    ky, hy = int(round(knee[1])), int(round(hoof[1]))
    top_clip = max(0, ky - 20)
    bottom_clip = min(mask.shape[0] - 1, hy + 60)
    band = np.zeros_like(mask)
    band[top_clip bottom_clip + 1, ] = 1
    lm = cv2.bitwise_and(lm, lm, mask=band.astype(np.uint8))
    lm = cv2.morphologyEx(lm, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    lm = cv2.dilate(lm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)

    xs = np.where(lm[min(bottom_clip, mask.shape[0] - 1)]  0)[0]
    if xs.size == 0
        if debug
            logging.info(rejecting candidate no pixels at hoof row)
        return None
    if xs[-1] - xs[0]  max(20, int(0.08  mask.shape[1]))
        if debug
            logging.info(rejecting candidate bottom too narrow)
        return None
    return lm


# ---------------------------------------------------------------------------
# Watershed leg separator
# ---------------------------------------------------------------------------

def seed_watershed_from_hooves(mask np.ndarray,
                                hooves list[tuple[float, float]],
                                rgba np.ndarray  None = None,
                                snap_radius int = 30) - list
    Segment mask into regions seeded at each hoof point using watershed.
    if mask is None or mask.size == 0
        return []
    bm = (mask  0).astype(np.uint8)  255
    h, w = bm.shape

    def snap_to_mask(xf, yf)
        x, y = int(round(xf)), int(round(yf))
        if 0 = x  w and 0 = y  h and bm[y, x]  0
            return (x, y)
        for r in range(1, snap_radius + 1)
            best, bestd = None, None
            for yy in range(max(0, y - r), min(h, y + r + 1))
                for xx in range(max(0, x - r), min(w, x + r + 1))
                    if bm[yy, xx]  0
                        d = (xx - x)  2 + (yy - y)  2
                        if bestd is None or d  bestd
                            bestd, best = d, (xx, yy)
            if best
                return best
        return None

    markers = np.zeros((h, w), dtype=np.int32)
    seed_points = []
    for i, (kx, ky) in enumerate(hooves, start=1)
        s = snap_to_mask(kx, ky)
        seed_points.append(s)
        if s
            cv2.circle(markers, s, 6, i, -1)

    if all(s is None for s in seed_points)
        return []

    try
        if rgba is not None
            gray = cv2.cvtColor(rgba[..., 3], cv2.COLOR_RGB2GRAY)
            inv = (255 - gray).astype(np.float32)  255.0
            dist = cv2.distanceTransform((bm  255).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
            if dist.max() = 0
                return []
            topo = dist  (1.0 + 0.7  inv)
            topo8 = np.uint8((topo  topo.max())  255.0)
            img3 = cv2.cvtColor(topo8, cv2.COLOR_GRAY2BGR)
        else
            dist = cv2.distanceTransform((bm  255).astype(np.uint8), cv2.DIST_L2, 5)
            if dist.max() = 0
                return []
            dist8 = np.uint8((dist  dist.max())  255.0)
            img3 = cv2.cvtColor(dist8, cv2.COLOR_GRAY2BGR)
        cv2.watershed(img3, markers)
    except Exception as e
        logging.warning(watershed failed %s, e)
        return []

    parts = []
    for i in range(1, len(hooves) + 1)
        m = np.zeros_like(bm)
        m[markers == i] = 255
        parts.append(m if cv2.countNonZero(m)  50 else None)
    return parts


# ---------------------------------------------------------------------------
# MMPose helpers
# ---------------------------------------------------------------------------

def get_ai_leg_keypoints(inferencer,
                         image_path str) - list[tuple[tuple[float, float], tuple[float, float]]]
    if inferencer is None
        return []
    try
        res = inferencer(image_path)
    except Exception as e
        logging.warning(MMPose inference failed %s, e)
        return []

    try
        results = next(iter(res)) if (hasattr(res, '__iter__') and not isinstance(res, dict)) else res
    except Exception
        results = res

    preds = None
    if isinstance(results, dict) and results.get('predictions')
        preds = results['predictions'][0]
    elif isinstance(results, list) and results
        preds = results[0]
    if not preds
        return []

    kpts, scores = None, None
    if isinstance(preds, dict)
        kpts = preds.get('keypoints') or preds.get('preds')
        scores = preds.get('keypoint_scores') or preds.get('scores')
    if kpts is None
        return []

    legs = []
    try
        if isinstance(kpts, np.ndarray)
            kpts = kpts.tolist()
        if len(kpts)  10
            def sc(i)
                return float(scores[i]) if scores and len(scores)  i else 1.0
            if sc(6)  0.12 and sc(7)  0.12
                legs.append((tuple(kpts[6][2]), tuple(kpts[7][2])))
            if sc(9)  0.12 and sc(10)  0.12
                legs.append((tuple(kpts[9][2]), tuple(kpts[10][2])))
    except Exception
        return []
    return legs


# ---------------------------------------------------------------------------
# FIX #8 (IMPLEMENTED) — Cannon-bone axis strictly vertical centre line
# ---------------------------------------------------------------------------

def find_cannon_bone_axis(leg_mask np.ndarray,
                          target_knee tuple[float, float]  None = None,
                          target_hoof tuple[float, float]  None = None
                          ) - tuple[tuple[int, int], tuple[int, int]]
    Return a strictly vertical centre-line axis for the cannon bone.

    FIX v3 Centre X is computed from the NARROWEST tubular section of the
    leg (the cannon bone) rather than a fixed 10–40% height zone. The
    cannon is the narrowest straight region between the fetlock (which
    widens at the joint) and the knee.  Using the narrowest rows makes the
    line independent of mask contamination from the tail or hoof flare.

    Algorithm
    1. Collect (row, left_x, right_x, mid_x, width) for every row.
    2. Sort rows by width ascending; take the narrowest 20% that are
       contiguous (i.e. form a band) — these are cannon bone rows.
    3. Median of their mid_x values = cannon centre X.
    4. The line runs from top_y to bottom_y of the full mask (blue line
       spans the entire selected leg region).
    
    h, w = leg_mask.shape
    clean = cv2.morphologyEx(leg_mask, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    ys_mask = np.where(np.any(clean  0, axis=1))[0]
    if ys_mask.size == 0
        return (w  2, 0), (w  2, h - 1)
    top_y, bottom_y = int(ys_mask[0]), int(ys_mask[-1])

    rows = []
    for ry in range(top_y, bottom_y + 1)
        xs = np.where(clean[ry]  0)[0]
        if xs.size = 2
            rows.append((ry, int(xs[0]), int(xs[-1]),
                         (float(xs[0]) + float(xs[-1]))  2.0,
                         int(xs[-1] - xs[0])))

    if not rows
        return (w  2, top_y), (w  2, bottom_y)

    # --- Cannon-zone selection (FIX v3) ---
    # Strategy exclude the bottom 25% (hoof flare) and top 10% (knee area),
    # then within the remaining band find the narrowest rows — these are the
    # cannon bone shaft.
    leg_h = bottom_y - top_y + 1
    shaft_top = top_y + int(leg_h  0.10)
    shaft_bot = top_y + int(leg_h  0.75)
    shaft_rows = [r for r in rows if shaft_top = r[0] = shaft_bot]

    if not shaft_rows
        shaft_rows = rows  # fallback

    # Sort shaft rows by width; take the narrowest 30% as cannon candidates
    shaft_rows_sorted = sorted(shaft_rows, key=lambda r r[4])
    n_cannon = max(5, int(len(shaft_rows_sorted)  0.30))
    cannon_rows = shaft_rows_sorted[n_cannon]

    # Use median mid_x of these narrowest rows — robust to outliers
    fit_xs = np.array([r[3] for r in cannon_rows], dtype=np.float64)
    median_cx = int(round(np.median(fit_xs)))

    # Clamp to actual mask extent
    pt_top = (median_cx, top_y)
    pt_bottom = (median_cx, bottom_y)

    logging.info(Cannon axis v3 (narrowest-rows) top=%s bottom=%s cx=%d (from %d rows),
                 pt_top, pt_bottom, median_cx, len(cannon_rows))
    return pt_top, pt_bottom


# ---------------------------------------------------------------------------
# Symmetry analysis with vertical centre line
# ---------------------------------------------------------------------------

def analyze_symmetry(leg_mask np.ndarray,
                     pt_top tuple[int, int],
                     pt_bottom tuple[int, int])
    Row-by-row symmetry analysis.

    With the vertical line fix (#8), cx_at(ry) always returns the same X
    (pt_top[0] == pt_bottom[0]), making split straightforward and accurate.
    
    h, w = leg_mask.shape
    green = np.zeros((h, w), dtype=np.uint8)
    red = np.zeros((h, w), dtype=np.uint8)

    top_y, bottom_y = pt_top[1], pt_bottom[1]
    dy = bottom_y - top_y

    def cx_at(ry int) - int
        Centre X at row ry — constant for vertical line.
        if dy == 0
            return pt_top[0]
        t = (ry - top_y)  dy
        return int(round(pt_top[0] + t  (pt_bottom[0] - pt_top[0])))

    total_left = total_right = 0
    row_data = []
    for ry in range(top_y, min(bottom_y + 1, h))
        xs = np.where(leg_mask[ry]  0)[0]
        if xs.size  2
            row_data.append(None)
            continue
        lx, rx = int(xs[0]), int(xs[-1])
        cx = cx_at(ry)
        if lx = cx or rx = cx
            row_data.append(None)
            continue
        lw = cx - lx
        rw = rx - cx
        total_left += lw
        total_right += rw
        row_data.append((ry, lx, rx, lw, rw, cx))

    if total_left  total_right  1.02
        dominant = LEFT
    elif total_right  total_left  1.02
        dominant = RIGHT
    else
        dominant = SYMMETRIC

    for item in row_data
        if item is None
            continue
        ry, lx, rx, lw, rw, cx = item
        sw = min(lw, rw)
        if dominant == LEFT
            green[ry, cx rx + 1] = leg_mask[ry, cx rx + 1]
            green[ry, max(0, cx - sw) cx] = leg_mask[ry, max(0, cx - sw) cx]
            if lw  sw
                es, ee = lx, max(0, cx - sw)
                if es  ee
                    red[ry, esee] = leg_mask[ry, esee]
        elif dominant == RIGHT
            green[ry, lx cx] = leg_mask[ry, lx cx]
            green[ry, cx min(w, cx + sw + 1)] = leg_mask[ry, cx min(w, cx + sw + 1)]
            if rw  sw
                es, ee = min(w, cx + sw + 1), rx + 1
                if es  ee
                    red[ry, esee] = leg_mask[ry, esee]
        else
            green[ry, lx rx + 1] = leg_mask[ry, lx rx + 1]

    green = cv2.bitwise_and(green, leg_mask)
    red = cv2.bitwise_and(red, leg_mask)
    logging.info(Dominant side %s (left=%d right=%d), dominant, total_left, total_right)
    return green, red, dominant


def apply_overlay(img np.ndarray, green_mask np.ndarray,
                  red_mask np.ndarray, alpha float = 0.55) - np.ndarray
    res = img.astype(np.float32)
    orig = res.copy()
    COLOR_GREEN = np.array([34, 197, 94], dtype=np.float32)
    COLOR_RED = np.array([48, 48, 220], dtype=np.float32)
    gm, rm = green_mask  0, red_mask  0
    res[gm] = orig[gm]  (1 - alpha) + COLOR_GREEN  alpha
    res[rm] = orig[rm]  (1 - alpha) + COLOR_RED  alpha
    return np.clip(res, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------------

def process_image(path str, do_debug bool = False, inferencer=None) - None
    p = Path(path)
    img = cv2.imread(str(p))
    if img is None
        logging.error(Cannot read %s, p)
        return
    h, w = img.shape[2]
    logging.info(Processing %s (%dx%d), p.name, w, h)

    rgba, mask = extract_foreground_rgba(img)

    # FIX #11 save RGBA with transparent background
    rgba_bgra = cv2.cvtColor(np.array(rgba if isinstance(rgba, np.ndarray) else np.array(rgba)),
                              cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(str(p.parent  f{p.stem}_foreground.png), rgba_bgra)
    logging.info(Saved foreground image (RGBA with transparency).)

    fg_bgr = cv2.bitwise_and(img, img, mask=mask)

    # FIX #1 pass mask so background gets neutral-gray fill, not black zeros
    depth_map = estimate_depth(fg_bgr, fg_mask=mask)
    depth_color = cv2.applyColorMap((depth_map  255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.imwrite(str(p.parent  f{p.stem}_depth.png), depth_color)
    logging.info(Saved depth map.)

    # FIX #9  #12 apply depth prefilter — removes back-leg pixels so leg
    # selection only sees the frontmost leg.  percentile=25 is conservative
    # enough to keep the full cannon bone even in ground-level shots while
    # still stripping the clearly-far back leg (blue in TURBO).
    # The height-safety guard (FIX #13) returns the original if the top of
    # the leg is accidentally cut off.
    depth_mask = depth_prefilter_mask(mask, depth_map, percentile=25.0)
    if do_debug
        cv2.imwrite(str(p.parent  f{p.stem}_depth_mask.png), depth_mask)
        logging.info(Saved depth-filtered mask (debug).)

    leg_masks = []
    leg_infos = []

    # --- AI path ---
    if inferencer is not None
        try
            legs = get_ai_leg_keypoints(inferencer, str(p))
        except Exception
            legs = []
        if legs
            best_leg, best_depth_val = None, -1.0
            for knee, hoof in legs
                line_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.line(line_mask,
                         (int(knee[0]), int(knee[1])),
                         (int(hoof[0]), int(hoof[1])), 255, thickness=5)
                overlap = (line_mask  0) & (mask  0)
                avg_d = float(depth_map[overlap].mean()) if np.any(overlap) else 0.0
                logging.info(Leg candidate knee=(%.1f,%.1f) hoof=(%.1f,%.1f) depth=%.3f,
                             knee, hoof, avg_d)
                if avg_d  best_depth_val
                    best_depth_val, best_leg = avg_d, (knee, hoof)

            if best_leg is not None
                knee, hoof = best_leg
                # FIX #9 use depth_mask instead of raw mask
                lm = select_front_leg_from_keypoints(depth_mask, knee, hoof, debug=do_debug)
                if lm is not None
                    leg_masks.append(lm)
                    leg_infos.append({'mask' lm, 'knee' knee, 'hoof' hoof})
                    cv2.imwrite(str(p.parent  f{p.stem}_isolated_leg.png), lm)
                    logging.info(Saved isolated leg (AI path).)

    # --- Fallback path ---
    if not leg_masks
        logging.warning(AI keypoints missing or failed — using fallback leg selection.)
        # FIX #9 use depth_mask so back-leg pixels are excluded from selection
        lm = select_front_leg_fallback(depth_mask, depth_map=depth_map, debug=do_debug)
        if lm is None
            logging.warning(No front leg found for %s, p.name)
            cv2.imwrite(str(p.parent  f{p.stem}_analyzed.jpg), img)
            return
        leg_masks = [lm]
        leg_infos = [{'mask' lm, 'knee' (w  2.0, 0.0), 'hoof' (w  2.0, float(h - 1))}]
        cv2.imwrite(str(p.parent  f{p.stem}_isolated_leg.png), lm)
        logging.info(Saved isolated leg (fallback path).)

    # --- Symmetry analysis ---
    combined_green = np.zeros((h, w), dtype=np.uint8)
    combined_red = np.zeros((h, w), dtype=np.uint8)

    # FIX #5 accumulate per-leg draw info in a list
    per_leg_draw list[dict] = []

    for info in leg_infos
        pt_top, pt_bottom = find_cannon_bone_axis(
            info['mask'],
            target_knee=info.get('knee'),
            target_hoof=info.get('hoof'),
        )
        green, red, dominant = analyze_symmetry(info['mask'], pt_top, pt_bottom)
        combined_green = np.maximum(combined_green, green)
        combined_red = np.maximum(combined_red, red)
        per_leg_draw.append({'pt_top' pt_top, 'pt_bottom' pt_bottom, 'dominant' dominant})

    out = apply_overlay(img, combined_green, combined_red, alpha=0.55)

    # FIX #5 draw every leg's axis and label
    for i, di in enumerate(per_leg_draw)
        cv2.line(out, di['pt_top'], di['pt_bottom'], (255, 80, 0), max(2, int(w  0.004)))
        cv2.putText(out, fLeg {i + 1} Dominant {di['dominant']},
                    (10, 30 + i  32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imwrite(str(p.parent  f{p.stem}_analyzed.jpg), out)

    if do_debug
        dbg = img.copy()
        dbg[combined_green  0] = [34, 197, 94]
        dbg[combined_red  0] = [48, 48, 220]
        for di in per_leg_draw
            cv2.line(dbg, di['pt_top'], di['pt_bottom'], (255, 80, 0), max(2, int(w  0.004)))
        cv2.imwrite(str(p.parent  f{p.stem}_debug.png), dbg)

    logging.info(Saved %s_analyzed.jpg, p.stem)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main()
    parser = argparse.ArgumentParser(description=Horse leg symmetry analyzer (v3))
    parser.add_argument(images, nargs=+, help=Image files or glob patterns)
    parser.add_argument(--debug, action=store_true, help=Save intermediate debug images)
    parser.add_argument(--use-ai, action=store_true,
                        help=Enable MMPose AI keypoint detection if available)
    parser.add_argument(--model-path, type=str, default=None,
                        help=Optional local model path for MMPoseInferencer)
    parser.add_argument(--device, type=str, default=None,
                        help=Device for MMPose (e.g. cpu or cuda0))
    args = parser.parse_args()

    inputs = sorted(set(f for pat in args.images for f in glob.glob(pat)))
    if not inputs
        logging.error(No input images found.)
        sys.exit(1)

    inferencer = None
    if args.use_ai and MMPoseInferencer is not None
        try
            kwargs = {device args.device} if args.device else {}
            pose = args.model_path or 'rtmpose-m_8xb64-210e_ap10k-256x256'
            inferencer = MMPoseInferencer(pose2d=pose, kwargs)
            logging.info(MMPose inferencer initialized.)
        except Exception as e
            logging.warning(Failed to initialize MMPoseInferencer %s, e)

    for img_path in inputs
        try
            process_image(img_path, do_debug=args.debug, inferencer=inferencer)
        except Exception as e
            logging.exception(Failed processing %s %s, img_path, e)


if __name__ == __main__
    main()