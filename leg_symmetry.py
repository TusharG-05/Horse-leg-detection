#!/usr/bin/env python3
"""
Horse Leg Symmetry Analyzer  (Single-Leg Edition)
===================================================
Uses `rembg` to isolate the horse, then selects the ONE most
prominent leg in the image. All other blobs (other legs, tail,
body) are ignored.

Overlays on the ORIGINAL COLOR image:
  - BLUE line  : vertical center axis of the leg
  - GREEN area : symmetric portion (equal on both sides of axis)
  - RED area   : asymmetric portion (the "extra" side)

Install:
    pip install opencv-python rembg onnxruntime numpy

Usage:
    python leg_symmetry_v2.py image1.jpg [image2.jpg ...]
"""

import cv2
import numpy as np
import sys
from pathlib import Path
from rembg import remove


# ──────────────────────────────────────────────
# 1.  BACKGROUND REMOVAL
# ──────────────────────────────────────────────

def extract_foreground_mask(image_bgr):
    """
    Runs rembg on the original image resolution.
    Returns a full-resolution binary mask (255=horse, 0=background).
    """
    rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgba = remove(rgb)                                   # rembg returns RGBA
    alpha = rgba[:, :, 3]
    _, mask = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)

    return mask


# ──────────────────────────────────────────────
# 2.  SINGLE-BEST-LEG DETECTION
# ──────────────────────────────────────────────

def detect_closeup(mask, h, w):
    """
    Returns True when this looks like a close-up shot where a single leg
    dominates the frame (common worm's-eye / ground-level photography).

    Heuristic: if the largest foreground blob spans > 40% of image width
    AND is reasonably tall, this is a close-up.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    largest = max(contours, key=cv2.contourArea)
    _, _, cw, ch = cv2.boundingRect(largest)
    return (cw / w) > 0.40 and (ch / h) > 0.30


def score_candidate(cnt, h, w, closeup=False):
    """
    Returns a numeric score for how likely a contour is a horse leg
    (higher = more likely to be the intended leg).

    Key rules used to ELIMINATE bad candidates:
      • tail  → very thin relative to image width
      • body  → too wide relative to image width
      • noise → too small, too short, or doesn't reach near the bottom

    closeup=True loosens the width ceiling for ground-level close-up shots
    where the leg naturally fills most of the frame.
    """
    x, y, cw, ch = cv2.boundingRect(cnt)
    area   = cv2.contourArea(cnt)
    bottom = y + ch
    aspect = ch / max(cw, 1)

    # ── Hard disqualifiers ──────────────────────────────────────

    # Must reach within the bottom 20% of the image
    if bottom < h * 0.80:
        return -1

    # Must have a minimum height (at least 15% of image height)
    if ch < h * 0.15:
        return -1

    # Too thin  →  likely a tail (< 3% of image width)
    if cw < w * 0.03:
        return -1

    # Too wide  →  likely the body / multiple merged legs
    # Close-up shots: leg can fill up to 80% of frame width
    # Normal shots:   anything above 45% is likely the merged body
    max_width_ratio = 0.80 if closeup else 0.45
    if cw > w * max_width_ratio:
        return -1

    # Must be taller than wide (it's a leg, not a hoof platform)
    # Slightly relaxed for close-ups where the hoof is very large in frame
    min_aspect = 1.0 if closeup else 1.2
    if aspect < min_aspect:
        return -1

    # ── Positive scoring ────────────────────────────────────────

    score = 0.0

    # 1. Large contour → more prominent leg
    score += area * 0.001

    # 2. Tall and narrow → ideal leg shape
    score += aspect * 500

    # 3. Reaches further down → hoof closer to ground → better candidate
    score += bottom * 0.5

    # 4. Prefer legs near the horizontal centre of the image
    #    (the most-photographed leg tends to be centred)
    cx_img = x + cw / 2
    dist_from_centre = abs(cx_img - w / 2)
    score -= dist_from_centre * 0.8

    # 5. Width bonus: normal shot 5-25%, close-up 30-75% of frame
    width_ratio = cw / w
    if closeup and 0.30 <= width_ratio <= 0.80:
        score += 400
    elif not closeup and 0.05 <= width_ratio <= 0.25:
        score += 300

    return score


def score_front_leg_candidate(cnt, h, w):
    """
    Estimates how likely a leg contour is the foreground leg.

    This is a pixel heuristic, not a true depth estimate. It favors the leg
    with more visible bulk and ground contact in the lower part of the image,
    which is often the leg in front when legs overlap in side-view shots.
    """
    x, y, cw, ch = cv2.boundingRect(cnt)
    area = cv2.contourArea(cnt)

    leg_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(leg_mask, [cnt], -1, 255, cv2.FILLED)

    lower_start = y + int(ch * 0.45)
    lower_end = min(h, y + ch)
    lower_area = np.sum(leg_mask[lower_start:lower_end] > 0)

    hoof_start = y + int(ch * 0.70)
    hoof_end = min(h, y + ch)
    hoof_area = np.sum(leg_mask[hoof_start:hoof_end] > 0)

    ground_row = min(h - 1, y + ch - 1)
    ground_width = np.sum(leg_mask[ground_row] > 0)

    vertical_fill = area / max(cw * ch, 1)
    width_ratio = cw / max(w, 1)

    score = 0.0
    score += lower_area * 1.4
    score += hoof_area * 1.8
    score += ground_width * 120.0
    score += vertical_fill * 800.0
    score += width_ratio * 250.0

    # Prefer the leg whose visible lower shape is more substantial.
    # This helps choose the foreground leg when one leg occludes the other.
    return score


def find_best_leg(mask, h, w):
    """
    Finds all plausible contours in the lower portion of the silhouette
    and returns the single highest-scoring one.

    Automatically switches to close-up mode when the dominant blob
    fills a large portion of the frame (worm-eye / ground-level shots).
    """
    closeup = detect_closeup(mask, h, w)
    print(f"  Shot mode : {chr(39)}CLOSE-UP{chr(39)} (wide leg)" if closeup else "  Shot mode : NORMAL")

    roi_top  = int(h * 0.10)
    roi_mask = np.zeros_like(mask)
    roi_mask[roi_top:] = mask[roi_top:]

    k3 = np.ones((3, 3), np.uint8)
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN,  k3, iterations=1)
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, k3, iterations=2)

    contours, _ = cv2.findContours(
        roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, None

    candidates = []

    for cnt in contours:
        s = score_candidate(cnt, h, w, closeup=closeup)
        if s >= 0:
            front_score = score_front_leg_candidate(cnt, h, w)
            candidates.append((s, front_score, cnt))

    if not candidates:
        return None, None

    # First keep only the best plausible leg candidates.
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    top_score = candidates[0][0]
    top_candidates = [item for item in candidates if item[0] >= top_score - 150]

    # If multiple legs are plausible, prefer the foreground one by its lower
    # visible bulk. Otherwise fall back to the best general leg candidate.
    best_score, best_front_score, best_cnt = max(
        top_candidates, key=lambda item: (item[1], item[0])
    )

    x, y, cw, ch = cv2.boundingRect(best_cnt)
    print(f"  Best leg  : bbox=({x},{y},{cw},{ch}), "
          f"score={best_score:.1f}, front={best_front_score:.1f}, "
          f"width={cw/w:.0%} of frame, aspect={ch/max(cw,1):.2f}")

    leg_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(leg_mask, [best_cnt], -1, 255, cv2.FILLED)

    return best_cnt, leg_mask


# ──────────────────────────────────────────────
# 3.  CENTER AXIS
# ──────────────────────────────────────────────

def find_center_x(leg_mask, x, y, cw, ch):
    """
    Computes the vertical centre axis of the leg.

    Strategy:
        1. Use the cannon / fetlock band (roughly top 20–50% of leg height).
            This part of the leg is usually straighter than the hoof flare,
            giving the most reliable reference for the plumb line.
      2. Average the midpoints of each row in that region.
      3. Fall back to bounding-box centre if the region is empty.

    Why not the fetlock or hoof? Those areas often flare or are
    asymmetric themselves, which would bias the axis.
    """
    h_img, _ = leg_mask.shape

    start_y = y + int(ch * 0.20)   # start below the body junction
    end_y   = y + int(ch * 0.50)   # stop before the hoof flare dominates

    midpoints = []
    for row_y in range(max(0, start_y), min(h_img, end_y)):
        pixels = np.nonzero(leg_mask[row_y])[0]
        if len(pixels) >= 2:
            midpoints.append((int(pixels[0]) + int(pixels[-1])) / 2)

    if midpoints:
        return int(np.median(midpoints))   # median is more robust than mean

    return x + cw // 2  # fallback


# ──────────────────────────────────────────────
# 4.  ROW-BY-ROW SYMMETRY ANALYSIS
# ──────────────────────────────────────────────

def analyze_symmetry(leg_mask, center_x, y, ch, h, w):
    """
    For every row of the leg:
      • symmetric part  → green_mask  (min width on both sides of axis)
      • asymmetric part → red_mask    (the "extra" side — can be LEFT or RIGHT)

    NOTE: We do NOT collapse to only the dominant side.
          Real legs often have small asymmetries on both sides;
          showing them all is more accurate for clinical use.
    """
    green = np.zeros((h, w), dtype=np.uint8)
    red   = np.zeros((h, w), dtype=np.uint8)

    # Focus the symmetry map on the lower leg: fetlock, cannon, and hoof.
    # The upper contour can include more of the body/shoulder area than we want
    # to visualize for this specific analysis.
    start_y = y + int(ch * 0.15)

    for row_y in range(start_y, min(y + ch, h)):
        pixels = np.nonzero(leg_mask[row_y])[0]
        if len(pixels) < 2:
            continue

        lx = int(pixels[0])
        rx = int(pixels[-1])

        # Skip rows where the entire leg is on one side of the axis
        if lx >= center_x or rx <= center_x:
            continue

        lw = center_x - lx   # width on left
        rw = rx - center_x   # width on right
        sw = min(lw, rw)     # symmetric (matching) width

        # ── GREEN: the symmetric portion ────────────────────────
        g_start = max(0, center_x - sw)
        g_end   = min(w, center_x + sw + 1)
        green[row_y, g_start:g_end] = 255

        # ── RED: the asymmetric portion ──────────────────────────
        if lw > rw:
            r_start = max(0, lx)
            r_end   = max(0, center_x - sw)
            if r_start < r_end:
                red[row_y, r_start:r_end] = 255

        elif rw > lw:
            r_start = min(w, center_x + sw + 1)
            r_end   = min(w, rx + 1)
            if r_start < r_end:
                red[row_y, r_start:r_end] = 255

    return green, red


# ──────────────────────────────────────────────
# 5.  COLOUR OVERLAY
# ──────────────────────────────────────────────

def apply_overlay(original_bgr, green_mask, red_mask, alpha=0.55):
    """
    Alpha-blends green/red symmetry regions over the ORIGINAL color image.
    Pixels outside the leg keep their original colors unchanged.
    """
    result = original_bgr.copy().astype(np.float32)
    orig_f = original_bgr.astype(np.float32)

    COLOR_GREEN = np.array([50, 220, 50],  dtype=np.float32)  # BGR
    COLOR_RED   = np.array([50,  50, 220], dtype=np.float32)  # BGR

    gm = green_mask > 0
    rm = red_mask   > 0

    result[gm] = orig_f[gm] * (1 - alpha) + COLOR_GREEN * alpha
    result[rm] = orig_f[rm] * (1 - alpha) + COLOR_RED   * alpha

    return np.clip(result, 0, 255).astype(np.uint8)


# ──────────────────────────────────────────────
# 6.  ASYMMETRY REPORT
# ──────────────────────────────────────────────

def asymmetry_report(red_mask, center_x, ch, y, w):
    """
    Returns a human-readable summary of where the asymmetry is
    and how severe it is (as a pixel-area ratio).
    """
    leg_region = red_mask[y:y + ch, :]
    total_red  = np.sum(leg_region > 0)

    if total_red == 0:
        return "No significant asymmetry detected."

    left_red  = np.sum(leg_region[:, :center_x] > 0)
    right_red = np.sum(leg_region[:, center_x:] > 0)

    # Total green (symmetric area)
    # We'll compute the ratio outside; here just return descriptive text
    dominant = "LEFT" if left_red > right_red else "RIGHT"
    ratio     = max(left_red, right_red) / max(min(left_red, right_red), 1)
    severity  = "mild" if ratio < 1.5 else ("moderate" if ratio < 3.0 else "significant")

    return (f"Dominant asymmetry: {dominant} side  |  "
            f"Severity: {severity}  |  "
            f"Left red px: {left_red}, Right red px: {right_red}")


# ──────────────────────────────────────────────
# 7.  MAIN PIPELINE
# ──────────────────────────────────────────────

def process_image(input_path):
    input_path = Path(input_path)
    image = cv2.imread(str(input_path))
    if image is None:
        raise ValueError(f"Cannot load image: {input_path}")

    h, w = image.shape[:2]
    print(f"\n{'='*55}")
    print(f"  Processing : {input_path.name}  ({w}x{h})")
    print(f"{'='*55}")

    # ── Step 1: Background removal ──────────────────────────────
    print("  [1/5] Removing background (rembg)...")
    mask = extract_foreground_mask(image)

    # ── Step 2: Find the single best leg ────────────────────────
    print("  [2/5] Selecting best leg candidate...")
    cnt, leg_mask = find_best_leg(mask, h, w)

    if cnt is None:
        print("  [WARN] No suitable leg found. Saving original image.")
        _save(image, input_path, "_analyzed")
        return

    x, y, cw, ch = cv2.boundingRect(cnt)

    # ── Step 3: Find centre axis ─────────────────────────────────
    print("  [3/5] Computing centre axis...")
    cx = find_center_x(leg_mask, x, y, cw, ch)
    print(f"  Centre axis at X = {cx}")

    # ── Step 4: Symmetry analysis ────────────────────────────────
    print("  [4/5] Analysing symmetry row-by-row...")
    green_mask, red_mask = analyze_symmetry(leg_mask, cx, y, ch, h, w)

    left_red = np.sum(red_mask[:, :cx] > 0)
    right_red = np.sum(red_mask[:, cx:] > 0)
    if left_red > right_red:
        red_mask[:, cx:] = 0
    elif right_red > left_red:
        red_mask[:, :cx] = 0
    else:
        red_mask[:, :] = 0

    report = asymmetry_report(red_mask, cx, ch, y, w)
    print(f"  {report}")

    # ── Step 5: Compose final image ──────────────────────────────
    print("  [5/5] Composing output image...")
    result = apply_overlay(image, green_mask, red_mask)

    # Draw the blue centre-axis line
    line_thickness = max(2, int(w * 0.004))
    axis_top = y + int(ch * 0.15)
    cv2.line(result, (cx, axis_top), (cx, y + ch), (220, 80, 0), line_thickness)

    out_path = _save(result, input_path, "_analyzed")
    print(f"  Saved → {out_path}")


def _save(img, input_path, suffix):
    out = input_path.parent / f"{input_path.stem}{suffix}.jpg"
    cv2.imwrite(str(out), img)
    return out


# ──────────────────────────────────────────────
# 8.  CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    paths = sys.argv[1:]
    if not paths:
        print(__doc__)
        sys.exit(0)

    for p in paths:
        try:
            process_image(p)
        except Exception as e:
            print(f"[ERROR] {p}: {e}")