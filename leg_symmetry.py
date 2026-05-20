#!/usr/bin/env python3
"""
Horse Leg Symmetry Analyzer (Color & rembg version)
===================================================
Uses `rembg` to isolate the horse from the background.
Overlays on the ORIGINAL COLOR image:
  - BLUE line  : vertical center axis of each leg (straight)
  - GREEN area : symmetric portion (equal on both sides)
  - RED area   : asymmetric portion (the "extra" side)

Usage:
    python leg_symmetry.py image1.jpg [image2.jpg ...]
"""

import cv2
import numpy as np
import sys
import argparse
import glob
import os
from pathlib import Path
from rembg import remove

# ──────────────────────────────────────────────
# 1. BACKGROUND REMOVAL (Using rembg with performance scaling)
# ──────────────────────────────────────────────
def extract_foreground_mask(image_bgr, do_downscale=True, max_dim=1000):
    """
    Uses rembg to remove the background and extract a clean binary mask 
    of the horse's silhouette. Performs downscaling on large images 
    for major speedup.
    """
    h, w = image_bgr.shape[:2]

    # Downscale only when requested to speed up rembg processing on large images
    if do_downscale and max_dim and max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        print(f"[INFO] Downscaling image from {w}x{h} to {new_w}x{new_h} for fast background isolation...")
        image_resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        image_resized = image_bgr

    print("[INFO] Running rembg to extract subject... (this may take a moment on first run)")
    
    # rembg expects RGB format
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    
    # Remove background (returns an RGBA image)
    subject_rgba = remove(image_rgb)
    
    # Extract the Alpha channel as our mask
    alpha_channel = subject_rgba[:, :, 3]
    
    # Binarize the alpha channel (ensure pure black/white)
    _, binary_mask_small = cv2.threshold(alpha_channel, 127, 255, cv2.THRESH_BINARY)
    
    # Upscale mask back to original resolution if it was downscaled
    if binary_mask_small.shape[:2] != (h, w):
        binary_mask = cv2.resize(binary_mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        binary_mask = binary_mask_small
        
    return binary_mask

# ──────────────────────────────────────────────
# 2. LEG ISOLATION (With Dynamic Crotch Split Detection)
# ──────────────────────────────────────────────
def find_split_row(mask, min_segment_width_ratio=0.015):
    """
    Scans horizontal rows of the mask to find where the single body silhouette
    splits into multiple leg silhouettes.
    """
    h, w = mask.shape
    
    # Dynamically compute minimum segment width based on image width
    min_width = max(1, int(w * min_segment_width_ratio))
    
    # Scan from 30% of height to 75% of height
    start_y = int(h * 0.30)
    end_y = int(h * 0.75)
    
    for y in range(start_y, end_y):
        row_pixels = mask[y]
        
        # Pad row with zeros to detect segments starting at 0 or ending at w-1
        padded = np.pad(row_pixels, 1, mode='constant', constant_values=0)
        diff = np.diff(padded.astype(np.int32))
        
        starts = np.nonzero(diff == 255)[0]
        ends = np.nonzero(diff == -255)[0]
        
        # Count significant segments
        segment_count = sum(1 for s, e in zip(starts, ends) if (e - s) >= min_width)
        
        # If we successfully split into 2 or more leg segments
        if segment_count >= 2:
            # Lookahead to verify this split is persistent and not random noise
            lookahead = max(5, int(h * 0.03))
            persistent = True
            for ly in range(y + 1, min(y + lookahead, end_y)):
                l_pixels = mask[ly]
                l_padded = np.pad(l_pixels, 1, mode='constant', constant_values=0)
                l_diff = np.diff(l_padded.astype(np.int32))
                l_starts = np.nonzero(l_diff == 255)[0]
                l_ends = np.nonzero(l_diff == -255)[0]
                l_count = sum(1 for ls, le in zip(l_starts, l_ends) if (le - ls) >= min_width)
                if l_count < 2:
                    persistent = False
                    break
            if persistent:
                return y
                
    # Fallback to default of 40% height if no split was detected
    return int(h * 0.40)

def get_leg_contours(mask):
    """
    Isolate the legs from the body dynamically using crotch split detection,
    then finding the downward-extending contours.
    """
    h, w = mask.shape

    # Dynamically find the Y-coordinate where the legs split from the body trunk
    roi_top = find_split_row(mask)
    print(f"[INFO] Dynamic crotch split row detected at Y = {roi_top} (approx. {roi_top/h*100:.1f}% height)")

    leg_only_mask = np.zeros_like(mask)
    leg_only_mask[roi_top:] = mask[roi_top:]

    # Find distinct blobs
    contours, _ = cv2.findContours(leg_only_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    props = []
    min_area = h * w * 0.0015  # Ignore tiny noise
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        bottom = y + ch
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00']) if M.get('m00', 0) != 0 else x + cw // 2
        props.append({'cnt': cnt, 'area': area, 'x': x, 'y': y, 'w': cw, 'h': ch, 'bottom': bottom, 'cx': cx})

    if not props:
        return []

    # Bottom-anchor filter: prefer contours that reach near the image bottom
    max_bottom = max(p['bottom'] for p in props)
    bottom_tol = int(h * 0.10)  # 10% of image height tolerance
    candidates = [p for p in props if p['bottom'] >= (max_bottom - bottom_tol)]

    # Relax tolerance if nothing remains
    if not candidates:
        bottom_tol = int(h * 0.25)
        candidates = [p for p in props if p['bottom'] >= (max_bottom - bottom_tol)]

    # Scoring: area, bottom proximity and closeness to image center
    max_area = max(p['area'] for p in props)
    img_cx = w / 2.0
    for p in candidates:
        area_norm = p['area'] / max_area if max_area > 0 else 0
        bottom_norm = p['bottom'] / h
        center_prox = 1.0 - (abs(p['cx'] - img_cx) / (w / 2.0))
        height_norm = p['h'] / h
        # Penalize tiny hoof-only blobs (likely not full leg)
        size_penalty = 0.5 if p['h'] < (h * 0.12) else 1.0
        p['score'] = (area_norm * 0.55 + bottom_norm * 0.25 + center_prox * 0.1 + height_norm * 0.1) * size_penalty

    candidates.sort(key=lambda p: p['score'], reverse=True)

    # Keep top 2 candidates (most likely front legs)
    selected = [p['cnt'] for p in candidates[:2]]

    # Final heuristics: ensure selected contours are vertically leg-like
    filtered = []
    for cnt in selected:
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = ch / max(cw, 1)
        bottom = y + ch
        if aspect > 1.0 and bottom > h * 0.40:
            filtered.append(cnt)

    # If nothing passes final heuristics, fallback to the largest by area
    if not filtered and props:
        props.sort(key=lambda p: p['area'], reverse=True)
        filtered = [props[0]['cnt']]

    # Sort left to right
    filtered.sort(key=lambda c: cv2.boundingRect(c)[0])
    return filtered

# ──────────────────────────────────────────────
# 3. CENTER AXIS (Enforces Vertical midline at Fetlock joint center)
# ──────────────────────────────────────────────
def find_center_x(leg_mask, x, y, cw, ch):
    """
    Finds the vertical center line based on the cannon bone shaft rather than the
    full leg mask. The cannon bone is typically the straighter, narrower region
    above the fetlock and below the knee, so we estimate the center from that band.
    """
    h, w = leg_mask.shape

    # Morphological closing to fill small holes and stabilise width measurements
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    clean = cv2.morphologyEx(leg_mask, cv2.MORPH_CLOSE, kernel)

    # Search a vertical span that likely contains the cannon shaft. We start
    # a bit above the provided bounding box (in case the bounding box is
    # truncated to the hoof) and scan upwards to find a narrow, consistent
    # band (the shaft).
    search_top = max(0, y - int(ch * 0.6))
    search_bot = min(h, y + ch)

    row_infos = []  # list of (row_y, lx, rx, width)
    for row_y in range(search_top, search_bot):
        pixels = np.nonzero(clean[row_y])[0]
        if len(pixels) >= 2:
            lx = int(pixels[0])
            rx = int(pixels[-1])
            row_infos.append((row_y, lx, rx, rx - lx))

    if row_infos:
        # Smooth widths using a simple median filter over neighboring rows
        widths = np.array([r[3] for r in row_infos], dtype=np.float32)
        smoothed = np.copy(widths)
        kernel_win = 11
        half = kernel_win // 2
        for i in range(len(widths)):
            lo = max(0, i - half)
            hi = min(len(widths), i + half + 1)
            smoothed[i] = float(np.median(widths[lo:hi]))

        # Find a sliding window of rows with the lowest mean smoothed width
        win = max(5, min(25, int(len(smoothed) * 0.2)))
        best_idx = None
        best_val = float('inf')
        for i in range(0, len(smoothed) - win + 1):
            val = float(np.mean(smoothed[i:i+win]))
            if val < best_val:
                best_val = val
                best_idx = i

        centers = []
        if best_idx is not None:
            for j in range(best_idx, min(best_idx + win, len(row_infos))):
                _, lx, rx, _ = row_infos[j]
                centers.append((lx + rx) / 2.0)

        # Fallback: if sliding-window fails, try percentile-based narrow rows
        if not centers:
            widths_arr = np.array([r[3] for r in row_infos], dtype=np.float32)
            pct = 50
            shaft_rows = []
            while pct <= 80 and not shaft_rows:
                width_limit = np.percentile(widths_arr, pct)
                shaft_rows = [r for r in row_infos if r[3] <= width_limit]
                pct += 10
            if not shaft_rows:
                shaft_rows = row_infos
            centers = [ (lx + rx) / 2.0 for (_, lx, rx, _) in shaft_rows ]

        if centers:
            centers = np.array(centers, dtype=np.float32)
            med = float(np.median(centers))
            tol = max(1.0, w * 0.08)
            filt = centers[np.abs(centers - med) <= tol]
            if filt.size >= 1:
                med = float(np.median(filt))
            return int(round(med))

    # Fallbacks
    M = cv2.moments(leg_mask)
    if M.get('m00', 0) != 0:
        return int(M['m10'] / M['m00'])
    return x + cw // 2

# ──────────────────────────────────────────────
# 4. ROW-BY-ROW SYMMETRY ANALYSIS
# ──────────────────────────────────────────────
def analyze_symmetry(leg_mask, center_x, y, ch, h, w):
    """
    Calculates the left and right width of the leg for every row.
    Green = the matched symmetric width.
    Red = the leftover asymmetric width.
    """
    green = np.zeros((h, w), dtype=np.uint8)
    red = np.zeros((h, w), dtype=np.uint8)

    for row_y in range(y, min(y + ch, h)):
        pixels = np.nonzero(leg_mask[row_y])[0]
        if len(pixels) < 2:
            continue

        lx = int(pixels[0])
        rx = int(pixels[-1])

        # If the whole row is on one side of the axis, ignore
        if lx >= center_x or rx <= center_x:
            continue

        lw = center_x - lx   # Width on left of axis
        rw = rx - center_x   # Width on right of axis
        sw = min(lw, rw)     # The symmetric matching width

        # Paint GREEN (symmetric part)
        g_start = max(0, center_x - sw)
        g_end = min(w, center_x + sw + 1)
        green[row_y, g_start:g_end] = 255

        # Paint RED (asymmetric part - whichever side is wider)
        if lw > rw: # Left side is wider
            r_start = max(0, lx)
            r_end = max(0, center_x - sw)
            if r_start < r_end:
                red[row_y, r_start:r_end] = 255
        elif rw > lw: # Right side is wider
            r_start = min(w, center_x + sw + 1)
            r_end = min(w, rx + 1)
            if r_start < r_end:
                red[row_y, r_start:r_end] = 255

    return green, red

# ──────────────────────────────────────────────
# 5. COLOUR OVERLAY
# ──────────────────────────────────────────────
def apply_overlay(original_bgr, green_mask, red_mask, alpha=0.55):
    """
    Blends the green and red symmetry maps transparently over the ORIGINAL color image.
    """
    result = original_bgr.copy().astype(np.float32)
    orig_f = original_bgr.astype(np.float32)

    # Define Colors in BGR format
    COLOR_GREEN = np.array([50, 220, 50], dtype=np.float32)
    COLOR_RED = np.array([50, 50, 220], dtype=np.float32)

    gm = green_mask > 0
    rm = red_mask > 0

    # Alpha blend
    result[gm] = orig_f[gm] * (1 - alpha) + COLOR_GREEN * alpha
    result[rm] = orig_f[rm] * (1 - alpha) + COLOR_RED * alpha

    return np.clip(result, 0, 255).astype(np.uint8)

# ──────────────────────────────────────────────
def clean_previous_outputs(directories):
    """Remove previously generated outputs in the given directories.

    Patterns removed: *_analyzed*, *_foreground*, *_silhouette*, *_cropped_*
    """
    patterns = ["*_analyzed*", "*_foreground*", "*_silhouette*", "*_cropped_*"]
    for d in directories:
        dpath = Path(d)
        if not dpath.is_dir():
            continue
        for pat in patterns:
            for f in dpath.glob(pat):
                try:
                    f.unlink()
                except Exception:
                    pass


# 6. MAIN PIPELINE
# ──────────────────────────────────────────────
def process_image(input_path, do_downscale=True, max_dim=1000):
    input_path = Path(input_path)
    image = cv2.imread(str(input_path))
    if image is None:
        raise ValueError(f"Cannot load image: {input_path}")

    h, w = image.shape[:2]
    print(f"\n[INFO] Processing: {input_path.name} ({w}x{h})")

    # 1. Background removal to get binary silhouette (with scaling)
    mask = extract_foreground_mask(image, do_downscale=do_downscale, max_dim=max_dim)

    # Save silhouette and foreground (background removed) images
    silhouette_path = input_path.parent / f"{input_path.stem}_silhouette.png"
    cv2.imwrite(str(silhouette_path), mask)

    foreground = cv2.bitwise_and(image, image, mask=mask)
    foreground_path = input_path.parent / f"{input_path.stem}_foreground.png"
    cv2.imwrite(str(foreground_path), foreground)

    print(f"[INFO] Saved silhouette to: {silhouette_path}")
    print(f"[INFO] Saved foreground to: {foreground_path}")

    # 2. Extract separated legs dynamically
    leg_contours = get_leg_contours(mask)
    print(f"[INFO] Legs isolated: {len(leg_contours)}")

    if not leg_contours:
        print("[WARN] No legs detected. Saving original image.")
        out_path = input_path.parent / f"{input_path.stem}_analyzed.jpg"
        cv2.imwrite(str(out_path), image)
        print(f"[INFO] Saved to: {out_path}")
        return

    # Prepare global masks for the overlays
    all_green = np.zeros((h, w), dtype=np.uint8)
    all_red = np.zeros((h, w), dtype=np.uint8)
    axis_lines = []

    # 3. Analyze each leg
    for i, cnt in enumerate(leg_contours):
        x, y, cw, ch = cv2.boundingRect(cnt)

        # Draw just this leg onto a blank mask
        leg_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(leg_mask, [cnt], -1, 255, cv2.FILLED)

        # Find center (located at the fetlock center) and analyze symmetry
        cx = find_center_x(leg_mask, x, y, cw, ch)
        axis_lines.append((cx, y, y + ch))

        gm, rm = analyze_symmetry(leg_mask, cx, y, ch, h, w)
        
        # Calculate dominant asymmetry side relative to vertical center
        asym_left = np.sum(rm[:, :cx])
        asym_right = np.sum(rm[:, cx:])
        asym_side = "LEFT" if asym_left > asym_right else ("RIGHT" if asym_right > asym_left else "SYMMETRIC")
        
        # Filter red mask to only color the dominant side red
        if asym_side == "LEFT":
            rm[:, cx:] = 0
        elif asym_side == "RIGHT":
            rm[:, :cx] = 0
        else:
            rm[:, :] = 0
            
        all_green = cv2.bitwise_or(all_green, gm)
        all_red = cv2.bitwise_or(all_red, rm)
        
        print(f"  -> Leg {i+1} | Fetlock Center X: {cx} | Dominant Asymmetry: {asym_side}")

    # 4. Apply coloring over ORIGINAL color image
    result = apply_overlay(image, all_green, all_red)

    # 5. Draw Solid BLUE Center lines (perfectly vertical line drawn at cx)
    for (cx, y_top, y_bot) in axis_lines:
        cv2.line(result, (cx, y_top), (cx, y_bot), (255, 50, 0), max(2, int(w*0.005)))

    # Save output alongside the original image to prevent directory collisions
    out_path = input_path.parent / f"{input_path.stem}_analyzed.jpg"
    cv2.imwrite(str(out_path), result)
    print(f"[INFO] Success! Saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Horse Leg Symmetry Analyzer")
    parser.add_argument("images", nargs="+", help="Image files or glob patterns")
    parser.add_argument("--no-downscale", action="store_true", help="Do not downscale images before running rembg (slower, but higher fidelity)")
    parser.add_argument("--max-dim", type=int, default=1000, help="Maximum dimension for downscaling (default: 1000)")
    args = parser.parse_args()

    # Expand glob patterns and collect unique directories to clean
    expanded = []
    dirs = set()
    for pat in args.images:
        for p in glob.glob(pat):
            expanded.append(p)
            dirs.add(str(Path(p).parent))

    if not expanded:
        print("No images found for the provided patterns.")
        sys.exit(0)

    # Always remove previous generated outputs in each input directory
    clean_previous_outputs(dirs)

    do_downscale = not args.no_downscale
    for img_path in expanded:
        try:
            process_image(img_path, do_downscale=do_downscale, max_dim=args.max_dim)
        except Exception as e:
            print(f"[ERROR] Failed processing {img_path}: {e}")