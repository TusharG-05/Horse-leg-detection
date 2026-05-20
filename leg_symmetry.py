#!/usr/bin/env python3
"""
Horse Leg Symmetry Analyzer — v3 (Cannon-bone to Hoof, Lighting-Robust)
========================================================================
Pipeline
--------
1. Remove background (rembg) at FULL resolution — no downscaling.
2. Detect the single most-prominent FRONT leg visible (closest to camera /
   largest apparent area in the lower-front region).
3. Find the CANNON BONE of that leg (the slender shaft above the fetlock)
   and derive a vertical centre-line that runs from the top of the cannon
   bone all the way down through the hoof.
4. Split the leg into LEFT and RIGHT halves relative to that centre-line.
5. Identify the DOMINANT side (the side that has more area = wider).
6. NON-dominant side → GREEN overlay.
   ASYMMETRIC extra pixels on the DOMINANT side → RED overlay.

Lighting robustness
-------------------
* Foreground isolation uses rembg alpha mask (illumination-independent).
* All subsequent geometry relies purely on the binary silhouette — no
  colour thresholding whatsoever — so lighting variations are irrelevant.

Usage
-----
    python leg_symmetry.py image1.jpg [image2.jpg ...]
    python leg_symmetry.py *.jpg --debug
"""

import cv2
import numpy as np
import sys
import argparse
import glob
import logging
from pathlib import Path
from rembg import remove

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# ─────────────────────────────────────────────────────────────────────────────
# 1.  BACKGROUND REMOVAL  (full-resolution, no downscaling)
# ─────────────────────────────────────────────────────────────────────────────

def extract_foreground_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    Run rembg on the FULL-RESOLUTION image and return a clean binary mask.
    Lighting conditions do not affect this step because rembg uses a
    deep-learning salient-object detector that is illumination-invariant.
    """
    h, w = image_bgr.shape[:2]
    logging.info("Running rembg on full-resolution image (%dx%d) …", w, h)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    subject_rgba = remove(image_rgb)           # RGBA numpy array

    alpha = subject_rgba[:, :, 3]
    _, binary_mask = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)

    # Light morphological closing to fill small holes in the mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, k)

    logging.info("Foreground mask extracted.")
    return binary_mask.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  BODY / LEG SPLIT — find the crotch row
# ─────────────────────────────────────────────────────────────────────────────

def find_split_row(mask: np.ndarray, min_seg_ratio: float = 0.012) -> int:
    """
    Scan downward from 25 % of image height to 80 % and return the first
    row where the silhouette breaks into ≥ 2 distinct blobs (legs split from
    the body trunk).  Includes a lookahead to reject one-row noise.
    """
    h, w = mask.shape
    min_w = max(1, int(w * min_seg_ratio))
    start_y = int(h * 0.25)
    end_y   = int(h * 0.80)

    for y in range(start_y, end_y):
        segs = _row_segments(mask[y], min_w)
        if len(segs) >= 2:
            lookahead = max(5, int(h * 0.025))
            if all(len(_row_segments(mask[ly], min_w)) >= 2
                   for ly in range(y + 1, min(y + lookahead, end_y))):
                logging.info("Crotch split detected at Y=%d (%.1f%% of height)", y, y / h * 100)
                return y

    # Smart fallback: scan the silhouette from 20% down to find the narrowest
    # horizontal extent — that row is close to where the cannon bone begins.
    # This is much better than a fixed 42% guess.
    scan_start = int(h * 0.20)
    scan_end   = int(h * 0.75)
    min_width  = w + 1
    best_y     = int(h * 0.42)
    for y in range(scan_start, scan_end):
        xs = np.where(mask[y] > 0)[0]
        if len(xs) < 2:
            continue
        row_w = int(xs[-1]) - int(xs[0])
        if row_w < min_width:
            min_width = row_w
            best_y = y
    logging.warning("No crotch split found; using narrowest-row fallback Y=%d", best_y)
    return best_y


def _row_segments(row: np.ndarray, min_width: int) -> list:
    """Return list of (start, end) pixel-column pairs for foreground runs."""
    padded = np.pad(row, 1, constant_values=0)
    diff   = np.diff(padded.astype(np.int32))
    starts = np.where(diff == 255)[0]
    ends   = np.where(diff == -255)[0]
    return [(s, e) for s, e in zip(starts, ends) if (e - s) >= min_width]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FRONT-LEG SELECTION
#     Among all leg blobs, pick the ONE that is most prominent / closest
#     to the camera.  Heuristics:
#       • Largest visible area  (near camera ↔ larger apparent size)
#       • Positioned in the FRONT half of the horse  (smaller X if horse
#         faces right; we use the blob that is more central / has the most
#         area in the lower third of the image)
#       • Good vertical continuity (true leg, not tail or body flap)
# ─────────────────────────────────────────────────────────────────────────────

def select_front_leg(mask: np.ndarray, split_row: int) -> np.ndarray | None:
    """
    Returns a single-leg binary mask (h×w, uint8) for the most prominent
    front leg, or None if nothing plausible is found.
    """
    h, w = mask.shape

    # Work only below the crotch split
    leg_zone = np.zeros_like(mask)
    leg_zone[split_row:] = mask[split_row:]

    contours, _ = cv2.findContours(leg_zone, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    min_area  = h * w * 0.0008          # ignore tiny specks
    max_width = int(w * 0.65)           # reject obviously merged two-leg blobs
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        if ch < h * 0.08:              # too short to be a real leg
            continue
        if cw > max_width:             # too wide → definitely two merged legs
            continue
        bottom   = y + ch
        # vertical continuity: fraction of bounding-box rows that contain pixels
        continuity = sum(1 for ry in range(y, min(y + ch, h))\
                         if np.any(mask[ry, x:x + cw])) / max(1, ch)
        aspect   = ch / max(cw, 1)

        # Penalise wide contours — a single leg should be much taller than wide
        # aspect >= 2 is ideal (tall/narrow). aspect < 1.5 → likely merged.
        aspect_score = min(aspect / 2.0, 1.0)   # 0–1, penalises width

        # Score: tall, narrow, continuous, and reaching the hoof
        bottom_score = bottom / h
        score = (area / (h * w) * 2.0 +
                 bottom_score      * 2.0 +
                 continuity        * 2.0 +
                 aspect_score      * 2.0)

        candidates.append({'cnt': cnt, 'score': score, 'area': area,
                            'x': x, 'y': y, 'w': cw, 'h': ch,
                            'bottom': bottom})


    if not candidates:
        return None

    # Pick the single best candidate
    candidates.sort(key=lambda c: c['score'], reverse=True)
    best = candidates[0]
    logging.info("Front leg selected: bbox=(%d,%d,%d,%d) score=%.3f",
                 best['x'], best['y'], best['w'], best['h'], best['score'])

    # Build its isolated mask
    leg_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(leg_mask, [best['cnt']], -1, 255, cv2.FILLED)
    return leg_mask


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CANNON-BONE DETECTION & CENTRE-LINE
#     The cannon bone is the narrowest, most uniform-width shaft of the leg
#     above the fetlock joint.  We find it by scanning row widths from the
#     TOP of the leg bounding box downward and locating the region of minimum,
#     stable width — that is the shaft.  The vertical centre of that shaft is
#     the axis; we extend the line from the top of the cannon bone all the
#     way to the bottom of the hoof (bottom of bounding box).
# ─────────────────────────────────────────────────────────────────────────────

def find_cannon_bone_axis(leg_mask: np.ndarray) -> tuple[int, int, int]:
    """
    Returns (center_x, axis_top_y, axis_bottom_y) where:
      • center_x     — X coordinate of the cannon-bone centre line
      • axis_top_y   — top of the cannon-bone region (shaft start)
      • axis_bottom_y— bottom of the hoof (full coverage)

    The axis is derived from the narrowest stable band of pixel widths
    (= the cannon bone shaft), but the LINE is drawn from there down to
    the very bottom of the leg mask, covering cannon bone + fetlock + hoof.
    """
    h, w = leg_mask.shape
    rows_with_pixels = []

    # Morphological closing to smooth the silhouette before width measurement
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    clean  = cv2.morphologyEx(leg_mask, cv2.MORPH_CLOSE, kernel)

    ys = np.where(np.any(clean > 0, axis=1))[0]
    if len(ys) == 0:
        cx = w // 2
        return cx, 0, h - 1

    top_y    = int(ys[0])
    bottom_y = int(ys[-1])

    for ry in range(top_y, bottom_y + 1):
        xs = np.where(clean[ry] > 0)[0]
        if len(xs) >= 2:
            lx  = int(xs[0])
            rx  = int(xs[-1])
            mid = (lx + rx) / 2.0
            wid = rx - lx
            rows_with_pixels.append((ry, lx, rx, mid, wid))

    if not rows_with_pixels:
        return w // 2, top_y, bottom_y

    # ── cannon bone shaft: upper 10–50 % of the leg contour height ───────────
    # The cannon bone sits in the upper portion of the isolated leg contour
    # (below the knee bump, above the fetlock narrowing).
    # Using the median centre-x from this band gives a stable, lighting-
    # independent axis that sits correctly in the cannon bone shaft.
    leg_height     = bottom_y - top_y
    cannon_start_y = top_y + int(leg_height * 0.10)
    cannon_end_y   = top_y + int(leg_height * 0.40)

    cannon_rows = [r for r in rows_with_pixels
                   if cannon_start_y <= r[0] <= cannon_end_y]
    if not cannon_rows:
        cannon_rows = rows_with_pixels   # fallback: all rows

    centres = np.array([r[3] for r in cannon_rows], dtype=np.float32)
    cx = int(round(float(np.median(centres))))

    # axis_top_y = very top of the leg contour (top of cannon bone)
    axis_top_y = top_y

    # Clamp centre_x to be inside the bounding box
    all_lx = min(r[1] for r in rows_with_pixels)
    all_rx = max(r[2] for r in rows_with_pixels)
    cx = int(np.clip(cx, all_lx + 1, all_rx - 1))

    logging.info("Cannon-bone axis: X=%d  top_y=%d (cannon zone %d-%d)  hoof_bottom=%d",
                 cx, axis_top_y, cannon_start_y, cannon_end_y, bottom_y)
    return cx, axis_top_y, bottom_y


# ─────────────────────────────────────────────────────────────────────────────
# 5.  ROW-BY-ROW SYMMETRY ANALYSIS
#     Covers from axis_top_y (cannon bone start) down to axis_bottom_y (hoof).
# ─────────────────────────────────────────────────────────────────────────────

def analyze_symmetry(leg_mask: np.ndarray,
                     center_x: int,
                     top_y: int,
                     bottom_y: int) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Returns (green_mask, red_mask, dominant_side) where:
      • green_mask — symmetric (non-dominant) side pixels
      • red_mask   — asymmetric extra pixels on the dominant side
      • dominant_side — 'LEFT' | 'RIGHT' | 'SYMMETRIC'

    Logic
    -----
    For every row from top_y to bottom_y:
      lw = width on the left of center_x
      rw = width on the right of center_x
      sw = min(lw, rw)   ← the matched symmetric half-width

    The matched band [center_x-sw … center_x+sw] is symmetric → GREEN
    (but only the side that is NOT the dominant side — see below).
    The extra pixels on the dominant side → RED.

    The dominant side is determined globally (total area on each side).
    Non-dominant side entirely → GREEN.
    Dominant side: only the matched portion on THAT side → GREEN;
                   the leftover extra portion → RED.
    """
    h, w = leg_mask.shape
    green = np.zeros((h, w), dtype=np.uint8)
    red   = np.zeros((h, w), dtype=np.uint8)

    # Accumulate total areas left/right for global dominant-side decision
    total_left  = 0
    total_right = 0

    row_data = []   # store per-row info for painting

    for ry in range(top_y, min(bottom_y + 1, h)):
        xs = np.where(leg_mask[ry] > 0)[0]
        if len(xs) < 2:
            row_data.append(None)
            continue
        lx = int(xs[0])
        rx = int(xs[-1])
        if lx >= center_x or rx <= center_x:
            row_data.append(None)
            continue
        lw = center_x - lx
        rw = rx - center_x
        total_left  += lw
        total_right += rw
        row_data.append((ry, lx, rx, lw, rw))

    # Determine dominant side
    if total_left > total_right * 1.02:
        dominant = "LEFT"
    elif total_right > total_left * 1.02:
        dominant = "RIGHT"
    else:
        dominant = "SYMMETRIC"

    logging.info("Dominant side: %s  (left_px=%d, right_px=%d)",
                 dominant, total_left, total_right)

    for item in row_data:
        if item is None:
            continue
        ry, lx, rx, lw, rw = item
        sw = min(lw, rw)   # symmetric half-width

        if dominant == "LEFT":
            # Non-dominant side = RIGHT → entirely GREEN
            green[ry, center_x: min(w, rx + 1)] = np.bitwise_and(
                leg_mask[ry, center_x: min(w, rx + 1)],
                np.full(min(w, rx + 1) - center_x, 255, dtype=np.uint8)
            )
            # Dominant side = LEFT:  matched portion → GREEN, extra → RED
            green[ry, max(0, center_x - sw): center_x] = np.bitwise_and(
                leg_mask[ry, max(0, center_x - sw): center_x],
                np.full(center_x - max(0, center_x - sw), 255, dtype=np.uint8)
            )
            if lw > sw:  # there is leftover on left
                extra_start = max(0, lx)
                extra_end   = max(0, center_x - sw)
                if extra_start < extra_end:
                    red[ry, extra_start:extra_end] = np.bitwise_and(
                        leg_mask[ry, extra_start:extra_end],
                        np.full(extra_end - extra_start, 255, dtype=np.uint8)
                    )

        elif dominant == "RIGHT":
            # Non-dominant side = LEFT → entirely GREEN
            green[ry, max(0, lx): center_x] = np.bitwise_and(
                leg_mask[ry, max(0, lx): center_x],
                np.full(center_x - max(0, lx), 255, dtype=np.uint8)
            )
            # Dominant side = RIGHT: matched portion → GREEN, extra → RED
            green[ry, center_x: min(w, center_x + sw + 1)] = np.bitwise_and(
                leg_mask[ry, center_x: min(w, center_x + sw + 1)],
                np.full(min(w, center_x + sw + 1) - center_x, 255, dtype=np.uint8)
            )
            if rw > sw:
                extra_start = min(w, center_x + sw + 1)
                extra_end   = min(w, rx + 1)
                if extra_start < extra_end:
                    red[ry, extra_start:extra_end] = np.bitwise_and(
                        leg_mask[ry, extra_start:extra_end],
                        np.full(extra_end - extra_start, 255, dtype=np.uint8)
                    )

        else:  # SYMMETRIC — paint everything green
            green[ry, max(0, lx): min(w, rx + 1)] = np.bitwise_and(
                leg_mask[ry, max(0, lx): min(w, rx + 1)],
                np.full(min(w, rx + 1) - max(0, lx), 255, dtype=np.uint8)
            )

    # Final clip to the actual leg pixels
    green = cv2.bitwise_and(green, leg_mask)
    red   = cv2.bitwise_and(red,   leg_mask)
    return green, red, dominant


# ─────────────────────────────────────────────────────────────────────────────
# 6.  COLOUR OVERLAY
# ─────────────────────────────────────────────────────────────────────────────

def apply_overlay(original_bgr: np.ndarray,
                  green_mask: np.ndarray,
                  red_mask: np.ndarray,
                  alpha: float = 0.55) -> np.ndarray:
    """Alpha-blend green/red symmetry maps over the original colour image."""
    result  = original_bgr.astype(np.float32)
    orig_f  = original_bgr.astype(np.float32)

    COLOR_GREEN = np.array([34, 197, 94],  dtype=np.float32)   # vibrant green (BGR)
    COLOR_RED   = np.array([48,  48, 220], dtype=np.float32)   # vivid red   (BGR)

    gm = green_mask > 0
    rm = red_mask   > 0
    result[gm] = orig_f[gm] * (1 - alpha) + COLOR_GREEN * alpha
    result[rm] = orig_f[rm] * (1 - alpha) + COLOR_RED   * alpha

    return np.clip(result, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def process_image(input_path: str, do_debug: bool = False) -> None:
    input_path = Path(input_path)
    image = cv2.imread(str(input_path))
    if image is None:
        raise ValueError(f"Cannot load image: {input_path}")

    h, w = image.shape[:2]
    logging.info("═" * 60)
    logging.info("Processing: %s  (%d × %d)", input_path.name, w, h)

    # ── Step 1: Background removal (full resolution, lighting-invariant) ──
    mask = extract_foreground_mask(image)

    if do_debug:
        sil_path = input_path.parent / f"{input_path.stem}_silhouette.png"
        cv2.imwrite(str(sil_path), mask)
        fg = cv2.bitwise_and(image, image, mask=mask)
        fg_path = input_path.parent / f"{input_path.stem}_foreground.png"
        cv2.imwrite(str(fg_path), fg)
        logging.info("Debug: silhouette → %s", sil_path)
        logging.info("Debug: foreground → %s", fg_path)

    # ── Step 2: Find where legs split from the body trunk ──
    split_row = find_split_row(mask)

    # ── Step 3: Select the single most-prominent front leg ──
    leg_mask = select_front_leg(mask, split_row)
    if leg_mask is None:
        logging.warning("No front leg detected — saving original image.")
        out_path = input_path.parent / f"{input_path.stem}_analyzed.jpg"
        cv2.imwrite(str(out_path), image)
        return

    if do_debug:
        leg_vis = cv2.bitwise_and(image, image, mask=leg_mask)
        lv_path = input_path.parent / f"{input_path.stem}_leg_isolated.png"
        cv2.imwrite(str(lv_path), leg_vis)
        logging.info("Debug: leg isolated → %s", lv_path)

    # ── Step 4: Cannon-bone centre-line (cannon bone → hoof bottom) ──
    center_x, axis_top_y, axis_bottom_y = find_cannon_bone_axis(leg_mask)

    # ── Step 5: Row-by-row symmetry analysis ──
    green_mask, red_mask, dominant_side = analyze_symmetry(
        leg_mask, center_x, axis_top_y, axis_bottom_y
    )

    # ── Step 6: Apply colour overlays to original image ──
    result = apply_overlay(image, green_mask, red_mask, alpha=0.55)

    # ── Step 7: Draw the cannon-bone centre-line in BLUE ──
    line_thickness = max(2, int(w * 0.004))
    cv2.line(result,
             (center_x, axis_top_y),
             (center_x, axis_bottom_y),
             (255, 80, 0),          # bright blue (BGR)
             line_thickness)

    # Annotate dominant side
    label = f"Dominant: {dominant_side}"
    font_scale = max(0.6, w / 2000.0)
    cv2.putText(result, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 0, 255), 2, cv2.LINE_AA)

    # ── Save foreground image (always) ──
    fg_path = input_path.parent / f"{input_path.stem}_foreground.jpg"
    fg_img  = cv2.bitwise_and(image, image, mask=mask)
    cv2.imwrite(str(fg_path), fg_img)
    logging.info("Foreground → %s", fg_path)

    # ── Save result ──
    out_path = input_path.parent / f"{input_path.stem}_analyzed.jpg"
    cv2.imwrite(str(out_path), result)
    logging.info("✓ Saved → %s", out_path)

    if do_debug:
        # Also save green/red masks separately for inspection
        dbg_path = input_path.parent / f"{input_path.stem}_debug.png"
        dbg = image.copy()
        dbg[green_mask > 0] = [34, 197, 94]
        dbg[red_mask   > 0] = [48,  48, 220]
        cv2.line(dbg, (center_x, axis_top_y), (center_x, axis_bottom_y),
                 (255, 80, 0), line_thickness)
        cv2.imwrite(str(dbg_path), dbg)
        logging.info("Debug: overlay → %s", dbg_path)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Horse Leg Symmetry Analyzer — cannon bone to hoof, "
                    "lighting-robust, full resolution"
    )
    parser.add_argument("images", nargs="+",
                        help="Image files or glob patterns")
    parser.add_argument("--debug", action="store_true",
                        help="Save intermediate debug images")
    parser.add_argument("--no-downscale", action="store_true",
                        help="No downscale (ignored, always full resolution now)")
    parser.add_argument("--max-dim", type=int, default=1000,
                        help="Max dimension (ignored, always full resolution now)")
    args = parser.parse_args()

    # Expand globs
    inputs = sorted(set(f for pat in args.images for f in glob.glob(pat)))
    if not inputs:
        logging.error("No images found for patterns: %s", args.images)
        sys.exit(1)

    for img_path in inputs:
        try:
            process_image(img_path, do_debug=args.debug)
        except Exception as exc:
            logging.error("Failed: %s — %s", img_path, exc, exc_info=True)