#!/usr/bin/env python3
# ant_counter_watershed_single_pipeline_plus_obvious_scored.py
#
# Pipeline:
#   multi-scale blackhat -> (Otsu or adaptive) threshold -> morph clean
#   -> optional suppress long lines -> filter mask components
#   -> watershed -> count (area-based for blobs) -> filter segments by shape
#   -> "obvious ants" scored counter (soft scoring instead of hard AND thresholds)
#
# Outputs in --outdir:
#   1_blackhat_big.png
#   1b_blackhat_small.png (if --multi-scale)
#   1c_blackhat_used_max.png (if --multi-scale)
#   2_mask_raw.png
#   3_mask_final.png
#   3d_line_mask.png (if --line-suppress)
#   4_dist_norm.png
#   5_sure_fg.png
#   6_overlay.png  (green=all kept segments, blue=obvious scored)
#
# NOTE: Make sure the file ends with:
#   if __name__ == "__main__":
#       main()
# (no extra "file:/..." text)

import argparse
import os
from datetime import datetime

import cv2
import numpy as np


# -----------------------------
# Preprocess / Mask
# -----------------------------
def preprocess_blackhat(img_bgr, blackhat_ksize=41, blur_ksize=5):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if blur_ksize % 2 == 0:
        blur_ksize += 1
    gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)

    k = int(blackhat_ksize)
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, kernel)
    bh = cv2.normalize(bh, None, 0, 255, cv2.NORM_MINMAX)
    return gray, g, bh


def make_initial_mask(
    bh,
    min_thresh=0,
    open_ksize=3,
    open_iter=1,
    close_iter=1,
    adaptive=False,
    block_size=51,
    C=2,
):
    # Threshold
    if adaptive:
        bs = int(block_size)
        if bs < 3:
            bs = 3
        if bs % 2 == 0:
            bs += 1

        binmask = cv2.adaptiveThreshold(
            bh,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            bs,
            int(C),
        )
    else:
        if int(min_thresh) == 0:
            _, binmask = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, binmask = cv2.threshold(bh, int(min_thresh), 255, cv2.THRESH_BINARY)

    # Gentle cleanup
    ok = int(open_ksize)
    if ok < 1:
        ok = 1
    if ok % 2 == 0:
        ok += 1

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ok, ok))
    clean = cv2.morphologyEx(binmask, cv2.MORPH_OPEN, k_open, iterations=int(open_iter))

    k3 = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k3, iterations=int(close_iter))
    return binmask, clean


def suppress_long_lines(mask, line_len=75, dilate=1):
    """
    Suppress long straight structures (ladder/rake/cage bars) without manual excludes.
    Extract long horizontal/vertical components with MORPH_OPEN using long kernels,
    then subtract them from mask.
    """
    L = int(line_len)
    if L < 9:
        L = 9
    if L % 2 == 0:
        L += 1

    hker = cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1))
    vker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, L))

    horiz = cv2.morphologyEx(mask, cv2.MORPH_OPEN, hker, iterations=1)
    vert = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vker, iterations=1)
    line_mask = cv2.bitwise_or(horiz, vert)

    d = int(dilate)
    if d > 0:
        k3 = np.ones((3, 3), np.uint8)
        line_mask = cv2.dilate(line_mask, k3, iterations=d)

    cleaned = cv2.subtract(mask, line_mask)
    return cleaned, line_mask


def filter_mask_components(mask, min_area=45, max_area=200000, max_aspect=12.0, min_extent=0.02):
    """
    Filter connected components on binary mask BEFORE watershed:
      - remove tiny noise (honeycomb texture)
      - remove very elongated line-like components
      - remove very sparse components in bbox (low extent)
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)

    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < int(min_area) or area > int(max_area):
            continue

        aspect = max(w, h) / max(1, min(w, h))
        if aspect > float(max_aspect):
            continue

        extent = area / float(w * h)
        if extent < float(min_extent):
            continue

        out[labels == i] = 255

    return out


# -----------------------------
# Watershed / Counting
# -----------------------------
def watershed_split(img_bgr, mask, dist_thresh_ratio=0.42, sure_bg_dilate=2):
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    _, sure_fg = cv2.threshold(dist_norm, float(dist_thresh_ratio), 1.0, cv2.THRESH_BINARY)
    sure_fg = (sure_fg * 255).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(mask, kernel, iterations=int(sure_bg_dilate))
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown > 0] = 0

    markers = cv2.watershed(img_bgr, markers)
    return dist_norm, sure_fg, sure_bg, unknown, markers


def count_from_markers(markers, area_min=20, area_single_max=520, area_blob_min=750):
    ids = [i for i in np.unique(markers) if i > 1]  # >1: actual segments
    areas = {}
    for i in ids:
        areas[i] = int(np.sum(markers == i))

    singles = []
    blobs = []
    for i, a in areas.items():
        if a < int(area_min):
            continue
        if a <= int(area_single_max):
            singles.append((i, a))
        else:
            blobs.append((i, a))

    if singles:
        median_single = int(np.median([a for _, a in singles]))
    else:
        median_single = max(1, int(area_single_max) // 2)

    parts = []
    for i, a in singles:
        parts.append((i, 1, a))

    for i, a in blobs:
        if a < int(area_blob_min):
            est = 1
        else:
            est = int(round(a / median_single))
            est = max(2, est)
        parts.append((i, est, a))

    total = sum(est for _, est, _ in parts)
    return total, median_single, singles, blobs, parts


def filter_segments_by_shape(markers, parts, max_aspect=14.0, min_extent=0.04, max_extent=0.95):
    """
    Filter segments AFTER watershed:
      - remove very elongated (lines)
      - remove too sparse in bbox (low extent)
      - remove too rectangle-like (extent near 1) which often are bars/boards
    """
    kept = []
    for seg_id, est, area in parts:
        ys, xs = np.where(markers == seg_id)
        if len(xs) == 0:
            continue

        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        w = max(1, x1 - x0 + 1)
        h = max(1, y1 - y0 + 1)

        aspect = max(w, h) / max(1, min(w, h))
        extent = area / float(w * h)

        if aspect > float(max_aspect):
            continue
        if extent < float(min_extent):
            continue
        if extent > float(max_extent):
            continue

        kept.append((seg_id, est, area))
    return kept


# -----------------------------
# Feature extraction for "obvious" scoring
# -----------------------------
def extract_segment_features(markers, seg_id, bh_used):
    ys, xs = np.where(markers == seg_id)
    if len(xs) == 0:
        return None

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    w = max(1, x1 - x0 + 1)
    h = max(1, y1 - y0 + 1)

    area = int(len(xs))
    aspect = max(w, h) / max(1, min(w, h))
    extent = area / float(w * h)

    # solidity = area / convex hull area
    pts = np.column_stack((xs, ys)).astype(np.int32)
    hull = cv2.convexHull(pts)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 1e-6 else 0.0

    # mean blackhat intensity inside segment
    mean_bh = float(np.mean(bh_used[ys, xs]))

    return {
        "bbox": (x0, y0, x1, y1),
        "w": w,
        "h": h,
        "area": area,
        "aspect": aspect,
        "extent": extent,
        "solidity": solidity,
        "mean_bh": mean_bh,
    }


def clamp01(x):
    return max(0.0, min(1.0, float(x)))


def score_range(x, lo, hi):
    """
    Soft membership:
      - 1 inside [lo, hi]
      - decays outside with a simple ratio
    """
    lo = float(lo)
    hi = float(hi)
    x = float(x)
    if hi <= lo:
        return 0.0
    if x < lo:
        return clamp01(x / lo) if lo > 1e-6 else 0.0
    if x > hi:
        return clamp01(hi / x) if x > 1e-6 else 0.0
    return 1.0


def count_obvious_ants_scored(
    markers,
    parts,
    bh_used,
    score_thr=0.65,
    area_lo=55,
    area_hi=520,
    aspect_lo=1.2,
    aspect_hi=9.0,
    extent_lo=0.06,
    extent_hi=0.75,
    solidity_lo=0.35,
    mean_bh_lo=15.0,
):
    """
    'Obvious' ants by scoring (not hard AND thresholds).
    Returns: (count, list_of_parts)
    where list_of_parts: (seg_id, est=1, area, score)
    """
    obvious = []
    for seg_id, est, _ in parts:
        f = extract_segment_features(markers, seg_id, bh_used)
        if f is None:
            continue

        s_area = score_range(f["area"], area_lo, area_hi)
        s_aspect = score_range(f["aspect"], aspect_lo, aspect_hi)
        s_extent = score_range(f["extent"], extent_lo, extent_hi)

        s_sol = clamp01((f["solidity"] - float(solidity_lo)) / (1.0 - float(solidity_lo) + 1e-6))
        s_bh = clamp01((f["mean_bh"] - float(mean_bh_lo)) / (255.0 - float(mean_bh_lo) + 1e-6))

        # Weighted sum
        score = 0.30 * s_area + 0.20 * s_aspect + 0.20 * s_extent + 0.20 * s_sol + 0.10 * s_bh

        if score >= float(score_thr):
            obvious.append((seg_id, 1, f["area"], float(score)))

    return len(obvious), obvious


# -----------------------------
# Visualization
# -----------------------------
def draw_overlay_two_groups(img_bgr, markers, parts_all, parts_obvious):
    out = img_bgr.copy()
    out[markers == -1] = (0, 0, 255)

    # all kept segments = green
    for seg_id, est, area in parts_all:
        ys, xs = np.where(markers == seg_id)
        if len(xs) == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())

        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), 1)
        cv2.putText(
            out,
            str(est),
            (x0, max(0, y0 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    # obvious = blue thick box + score label
    for seg_id, est, area, score in parts_obvious:
        ys, xs = np.where(markers == seg_id)
        if len(xs) == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())

        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 0, 0), 2)
        cv2.putText(
            out,
            f"{score:.2f}",
            (x0, min(out.shape[0] - 2, y1 + 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return out


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")

    now = datetime.now()
    ap.add_argument("--outdir", default=f"out_ws_{now.strftime('%d_%H_%M_%S')}")

    # blackhat
    ap.add_argument("--blackhat-ksize", type=int, default=41, help="Large-scale blackhat")
    ap.add_argument("--multi-scale", action="store_true", help="Use max(blackhat_big, blackhat_small)")
    ap.add_argument("--small-blackhat-ksize", type=int, default=21, help="Small-scale blackhat")

    # threshold
    ap.add_argument("--adaptive", action="store_true", help="Use adaptiveThreshold instead of Otsu")
    ap.add_argument("--block-size", type=int, default=51)
    ap.add_argument("--C", type=int, default=2)
    ap.add_argument("--min-thresh", type=int, default=0, help="If not adaptive: 0=Otsu else fixed 1..255")

    # morph cleanup
    ap.add_argument("--open-ksize", type=int, default=3)
    ap.add_argument("--open-iter", type=int, default=1)
    ap.add_argument("--close-iter", type=int, default=1)

    # line suppression
    ap.add_argument("--line-suppress", action="store_true")
    ap.add_argument("--line-len", type=int, default=75)
    ap.add_argument("--line-dilate", type=int, default=1)

    # mask component filtering (pre-watershed)
    ap.add_argument("--mask-min-area", type=int, default=45)
    ap.add_argument("--mask-max-aspect", type=float, default=12.0)
    ap.add_argument("--mask-min-extent", type=float, default=0.02)

    # watershed
    ap.add_argument("--dist-ratio", type=float, default=0.42, help="smaller -> more splits")
    ap.add_argument("--bg-dilate", type=int, default=2)

    # counting
    ap.add_argument("--area-min", type=int, default=20)
    ap.add_argument("--area-single-max", type=int, default=520)
    ap.add_argument("--area-blob-min", type=int, default=750)

    # segment shape filtering (post-watershed)
    ap.add_argument("--seg-max-aspect", type=float, default=14.0)
    ap.add_argument("--seg-min-extent", type=float, default=0.04)
    ap.add_argument("--seg-max-extent", type=float, default=0.95)

    # obvious scored
    ap.add_argument("--ob-score", type=float, default=0.65, help="score threshold for 'obvious' (0..1)")
    ap.add_argument("--ob-area-lo", type=int, default=55)
    ap.add_argument("--ob-area-hi", type=int, default=520)
    ap.add_argument("--ob-aspect-lo", type=float, default=1.2)
    ap.add_argument("--ob-aspect-hi", type=float, default=9.0)
    ap.add_argument("--ob-extent-lo", type=float, default=0.06)
    ap.add_argument("--ob-extent-hi", type=float, default=0.75)
    ap.add_argument("--ob-solidity-lo", type=float, default=0.35)
    ap.add_argument("--ob-meanbh-lo", type=float, default=15.0)

    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Не могу прочитать: {args.image}")

    # --- blackhat(s) ---
    _, _, bh_big = preprocess_blackhat(img, blackhat_ksize=args.blackhat_ksize)

    bh_used = bh_big
    bh_small = None
    if args.multi_scale:
        _, _, bh_small = preprocess_blackhat(img, blackhat_ksize=args.small_blackhat_ksize)
        bh_used = cv2.max(bh_big, bh_small)

    # --- threshold + clean ---
    raw, mask = make_initial_mask(
        bh_used,
        min_thresh=args.min_thresh,
        open_ksize=args.open_ksize,
        open_iter=args.open_iter,
        close_iter=args.close_iter,
        adaptive=args.adaptive,
        block_size=args.block_size,
        C=args.C,
    )

    # --- optional line suppression ---
    line_mask = None
    if args.line_suppress:
        mask, line_mask = suppress_long_lines(mask, line_len=args.line_len, dilate=args.line_dilate)

    # --- pre-watershed mask filtering ---
    mask = filter_mask_components(
        mask,
        min_area=args.mask_min_area,
        max_aspect=args.mask_max_aspect,
        min_extent=args.mask_min_extent,
    )

    # --- watershed ---
    dist_norm, sure_fg, sure_bg, unknown, markers = watershed_split(
        img, mask, dist_thresh_ratio=args.dist_ratio, sure_bg_dilate=args.bg_dilate
    )

    total_est, median_single, singles, blobs, parts = count_from_markers(
        markers,
        area_min=args.area_min,
        area_single_max=args.area_single_max,
        area_blob_min=args.area_blob_min,
    )

    # --- post-watershed segment filtering ---
    parts = filter_segments_by_shape(
        markers,
        parts,
        max_aspect=args.seg_max_aspect,
        min_extent=args.seg_min_extent,
        max_extent=args.seg_max_extent,
    )
    total_est = sum(est for _, est, _ in parts)

    # --- obvious ants scored ---
    obvious_count, obvious_parts = count_obvious_ants_scored(
        markers,
        parts,
        bh_used,
        score_thr=args.ob_score,
        area_lo=args.ob_area_lo,
        area_hi=args.ob_area_hi,
        aspect_lo=args.ob_aspect_lo,
        aspect_hi=args.ob_aspect_hi,
        extent_lo=args.ob_extent_lo,
        extent_hi=args.ob_extent_hi,
        solidity_lo=args.ob_solidity_lo,
        mean_bh_lo=args.ob_meanbh_lo,
    )

    overlay = draw_overlay_two_groups(img, markers, parts, obvious_parts)

    # --- save debug ---
    os.makedirs(args.outdir, exist_ok=True)
    cv2.imwrite(os.path.join(args.outdir, "1_blackhat_big.png"), bh_big)
    if args.multi_scale and bh_small is not None:
        cv2.imwrite(os.path.join(args.outdir, "1b_blackhat_small.png"), bh_small)
        cv2.imwrite(os.path.join(args.outdir, "1c_blackhat_used_max.png"), bh_used)

    cv2.imwrite(os.path.join(args.outdir, "2_mask_raw.png"), raw)
    cv2.imwrite(os.path.join(args.outdir, "3_mask_final.png"), mask)
    if args.line_suppress and line_mask is not None:
        cv2.imwrite(os.path.join(args.outdir, "3d_line_mask.png"), line_mask)

    cv2.imwrite(os.path.join(args.outdir, "4_dist_norm.png"), (dist_norm * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(args.outdir, "5_sure_fg.png"), sure_fg)
    cv2.imwrite(os.path.join(args.outdir, "6_overlay.png"), overlay)

    print(f"Оценка муравьёв (с учётом слипшихся): {total_est}")
    print(f"Явных муравьёв (score >= {args.ob_score:.2f}): {obvious_count}")
    print(f"Median площадь 'одиночного' сегмента: {median_single}")
    print(f"Сегментов после фильтров: {len(parts)}")
    print(f"Смотри оверлей: {args.outdir}/6_overlay.png")


if __name__ == "__main__":
    main()

#use to run
#python ant_counter.py photo.jpg --multi-scale --small-blackhat-ksize 21 --adaptive --block-size 51 --C 2 --line-suppress --line-len 75 --line-dilate 1 --dist-ratio 0.42 --ob-score 0.75


