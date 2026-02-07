#!/usr/bin/env python3
# ant_counter.py
#
# Single pipeline:
#   LAB L + CLAHE -> blackhat (multi-scale optional) -> chroma suppression (optional)
#   -> threshold (Otsu / adaptive) -> morph clean
#   -> optional suppress long lines -> pre-filter mask CC
#   -> watershed -> area-based counting for blobs
#   -> post-filter segments by shape
#   -> "obvious ants" scoring + optional orientation (PCA or skeleton-PCA)
#
# Debug images saved into --outdir

import argparse
import os
from datetime import datetime

import cv2
import numpy as np


# -----------------------------
# LAB helpers
# -----------------------------
def bgr_to_lab(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    return L, A, B


def chroma_weight(A_u8, B_u8, gamma=1.2, blur=0, floor=0.05):
    """
    Weight map HIGH for low-chroma (near gray) pixels, LOW for colorful pixels.
    A,B in [0..255], 128 is neutral.
    Returns float32 w in [0..1].
    """
    a = A_u8.astype(np.float32) - 128.0
    b = B_u8.astype(np.float32) - 128.0
    chroma = np.sqrt(a * a + b * b)  # 0..~181
    c = np.clip(chroma / 181.0, 0.0, 1.0)
    w = 1.0 - c
    if gamma != 1.0:
        w = np.power(np.clip(w, 0.0, 1.0), float(gamma))
    if int(blur) > 0:
        k = int(blur)
        if k < 3:
            k = 3
        if k % 2 == 0:
            k += 1
        w = cv2.GaussianBlur(w, (k, k), 0)
    w = np.clip(w, float(floor), 1.0)
    return w


def clahe_u8(gray_u8, clip=2.0, grid=8):
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(grid), int(grid)))
    return clahe.apply(gray_u8)


def blackhat_u8(gray_u8, ksize):
    k = int(ksize)
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bh = cv2.morphologyEx(gray_u8, cv2.MORPH_BLACKHAT, ker)
    bh = cv2.normalize(bh, None, 0, 255, cv2.NORM_MINMAX)
    return bh


def preprocess_lab_blackhat(
    img_bgr,
    blur_ksize=5,
    clahe_clip=2.0,
    clahe_grid=8,
    blackhat_ksize=41,
    multi_scale=False,
    small_blackhat_ksize=21,
    use_chroma_suppress=True,
    chroma_gamma=1.2,
    chroma_blur=0,
    chroma_floor=0.05,
):
    """
    Returns:
      L_blur_u8, L_clahe_u8,
      bh_big_u8, bh_small_u8_or_None, bh_used_u8,
      w_float_or_None, bh_weighted_u8
    """
    L, A, B = bgr_to_lab(img_bgr)

    k = int(blur_ksize)
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1
    L_blur = cv2.GaussianBlur(L, (k, k), 0)

    Lc = clahe_u8(L_blur, clip=clahe_clip, grid=clahe_grid)

    bh_big = blackhat_u8(Lc, blackhat_ksize)
    bh_small = None
    bh_used = bh_big

    if multi_scale:
        bh_small = blackhat_u8(Lc, small_blackhat_ksize)
        bh_used = cv2.max(bh_big, bh_small)

    w = None
    bh_w = bh_used.copy()
    if use_chroma_suppress:
        w = chroma_weight(A, B, gamma=chroma_gamma, blur=chroma_blur, floor=chroma_floor)  # float32 0..1
        bh_w = (bh_used.astype(np.float32) * w).astype(np.uint8)

    return L_blur, Lc, bh_big, bh_small, bh_used, w, bh_w


def mean_chroma_in_seg(A_u8, B_u8, xs, ys):
    a = A_u8[ys, xs].astype(np.float32) - 128.0
    b = B_u8[ys, xs].astype(np.float32) - 128.0
    chroma = np.sqrt(a * a + b * b)  # 0..~181
    return float(np.mean(chroma))


# -----------------------------
# Preprocess / Mask
# -----------------------------
def make_initial_mask(
    score_u8,
    adaptive=False,
    block_size=51,
    C=2,
    min_thresh=0,
    open_ksize=3,
    open_iter=1,
    close_iter=1,
):
    # Threshold
    if adaptive:
        bs = int(block_size)
        if bs < 3:
            bs = 3
        if bs % 2 == 0:
            bs += 1
        binmask = cv2.adaptiveThreshold(
            score_u8,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            bs,
            int(C),
        )
    else:
        if int(min_thresh) == 0:
            _, binmask = cv2.threshold(score_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, binmask = cv2.threshold(score_u8, int(min_thresh), 255, cv2.THRESH_BINARY)

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
    Extract long horizontal/vertical strokes and subtract from mask.
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
    Filter binary mask components before watershed:
      - too small
      - too elongated (aspect)
      - too sparse in bbox (extent)
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

        extent = area / float(max(1, w * h))
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
    ids = [i for i in np.unique(markers) if i > 1]
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

    count = 0
    parts = []

    # singles
    count += len(singles)
    for i, a in singles:
        parts.append((i, 1, a))

    # blobs -> estimate
    for i, a in blobs:
        if a < int(area_blob_min):
            est = 1
        else:
            est = int(round(a / median_single))
            est = max(2, est)
        count += est
        parts.append((i, est, a))

    return count, median_single, singles, blobs, parts


# -----------------------------
# Segment features / filters
# -----------------------------
def segment_bbox(markers, seg_id):
    ys, xs = np.where(markers == seg_id)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1, y1, xs, ys


def pca_orientation(xs, ys):
    """
    PCA on pixel coordinates.
    Returns:
      angle_deg in [-90..90], linearity in [0..1] (approx), eigen_ratio (lam1/lam2)
    """
    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    if pts.shape[0] < 5:
        return 0.0, 0.0, 1.0

    mean = pts.mean(axis=0, keepdims=True)
    X = pts - mean
    cov = (X.T @ X) / max(1, (pts.shape[0] - 1))
    w, v = np.linalg.eigh(cov)  # ascending
    lam2, lam1 = float(w[0]), float(w[1])  # lam1 >= lam2
    vec = v[:, 1]  # principal
    angle = np.degrees(np.arctan2(vec[1], vec[0]))
    while angle <= -90:
        angle += 180
    while angle > 90:
        angle -= 180

    ratio = (lam1 + 1e-9) / (lam2 + 1e-9)
    linearity = 1.0 - (lam2 / (lam1 + 1e-9))  # ~0 round ->1 line
    return float(angle), float(np.clip(linearity, 0.0, 1.0)), float(ratio)


def zhang_suen_thinning(bin_u8):
    """
    Simple Zhang-Suen thinning for binary (0/255) image.
    Returns 0/255 skeleton.
    """
    img = (bin_u8 > 0).astype(np.uint8).copy()
    changed = True
    h, w = img.shape

    def neighbors(y, x):
        p2 = img[y - 1, x]
        p3 = img[y - 1, x + 1]
        p4 = img[y, x + 1]
        p5 = img[y + 1, x + 1]
        p6 = img[y + 1, x]
        p7 = img[y + 1, x - 1]
        p8 = img[y, x - 1]
        p9 = img[y - 1, x - 1]
        return p2, p3, p4, p5, p6, p7, p8, p9

    def transitions(ps):
        p2, p3, p4, p5, p6, p7, p8, p9 = ps
        seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
        t = 0
        for i in range(8):
            if seq[i] == 0 and seq[i + 1] == 1:
                t += 1
        return t

    while changed:
        changed = False
        to_remove = []

        # step 1
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if img[y, x] != 1:
                    continue
                ps = neighbors(y, x)
                n = sum(ps)
                if n < 2 or n > 6:
                    continue
                if transitions(ps) != 1:
                    continue
                p2, p3, p4, p5, p6, p7, p8, p9 = ps
                if p2 * p4 * p6 != 0:
                    continue
                if p4 * p6 * p8 != 0:
                    continue
                to_remove.append((y, x))
        if to_remove:
            for y, x in to_remove:
                img[y, x] = 0
            changed = True

        to_remove = []
        # step 2
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if img[y, x] != 1:
                    continue
                ps = neighbors(y, x)
                n = sum(ps)
                if n < 2 or n > 6:
                    continue
                if transitions(ps) != 1:
                    continue
                p2, p3, p4, p5, p6, p7, p8, p9 = ps
                if p2 * p4 * p8 != 0:
                    continue
                if p2 * p6 * p8 != 0:
                    continue
                to_remove.append((y, x))
        if to_remove:
            for y, x in to_remove:
                img[y, x] = 0
            changed = True

    return (img * 255).astype(np.uint8)


def skeleton_orientation(segmask_u8):
    """
    Compute orientation on skeleton pixels (PCA).
    segmask_u8: 0/255 mask of ONE segment cropped.
    Returns angle_deg, linearity
    """
    sk = zhang_suen_thinning(segmask_u8)
    ys, xs = np.where(sk > 0)
    if len(xs) < 5:
        ys, xs = np.where(segmask_u8 > 0)
        if len(xs) < 5:
            return 0.0, 0.0
    angle, lin, _ = pca_orientation(xs, ys)
    return angle, lin


def filter_segments_by_shape(markers, parts, max_aspect=14.0, min_extent=0.04, max_extent=0.95, max_bbox_area=1e9):
    kept = []
    for seg_id, est, area in parts:
        bb = segment_bbox(markers, seg_id)
        if bb is None:
            continue
        x0, y0, x1, y1, xs, ys = bb
        w = max(1, x1 - x0 + 1)
        h = max(1, y1 - y0 + 1)

        if (w * h) > float(max_bbox_area):
            continue

        aspect = max(w, h) / max(1, min(w, h))
        extent = area / float(max(1, w * h))

        if aspect > float(max_aspect):
            continue
        if extent < float(min_extent):
            continue
        if extent > float(max_extent):
            continue

        kept.append((seg_id, est, area))
    return kept


# -----------------------------
# "Obvious ant" scoring (high-confidence)
# -----------------------------
def obvious_ant_score(
    bh_u8,  # blackhat-weighted score (0..255)
    markers,
    seg_id,
    area,
    A_u8=None,
    B_u8=None,
    chroma_hi=18.0,
    orient="pca",  # "pca" or "skeleton"
    area_lo=55,
    area_hi=420,
    meanbh_lo=30,
    aspect_lo=1.6,
    extent_hi=0.85,
    linearity_lo=0.35,
):
    """
    Returns (is_obvious, score01, angle_deg, linearity, bbox, dbg_dict)
    """
    bb = segment_bbox(markers, seg_id)
    if bb is None:
        return False, 0.0, 0.0, 0.0, None, {}
    x0, y0, x1, y1, xs, ys = bb
    w = max(1, x1 - x0 + 1)
    h = max(1, y1 - y0 + 1)

    aspect = max(w, h) / max(1, min(w, h))
    extent = area / float(max(1, w * h))

    mean_bh = float(np.mean(bh_u8[ys, xs]))

    if orient == "skeleton":
        segmask = np.zeros((h, w), np.uint8)
        segmask[(ys - y0), (xs - x0)] = 255
        angle, linearity = skeleton_orientation(segmask)
    else:
        angle, linearity, _ratio = pca_orientation(xs, ys)

    mean_chr = 0.0
    if A_u8 is not None and B_u8 is not None:
        mean_chr = mean_chroma_in_seg(A_u8, B_u8, xs, ys)

    dbg = dict(
        area=int(area),
        meanBH=float(mean_bh),
        lin=float(linearity),
        ang=float(angle),
        asp=float(aspect),
        ext=float(extent),
        meanChr=float(mean_chr),
        bb=(x0, y0, x1, y1),
    )

    # Hard gates
    if area < int(area_lo) or area > int(area_hi):
        return False, 0.0, angle, linearity, (x0, y0, x1, y1), dbg
    if mean_bh < float(meanbh_lo):
        return False, 0.0, angle, linearity, (x0, y0, x1, y1), dbg
    if aspect < float(aspect_lo):
        return False, 0.0, angle, linearity, (x0, y0, x1, y1), dbg
    if extent > float(extent_hi):
        return False, 0.0, angle, linearity, (x0, y0, x1, y1), dbg
    if linearity < float(linearity_lo):
        return False, 0.0, angle, linearity, (x0, y0, x1, y1), dbg
    if (A_u8 is not None and B_u8 is not None) and (mean_chr > float(chroma_hi)):
        return False, 0.0, angle, linearity, (x0, y0, x1, y1), dbg

    # Soft score 0..1
    s_area = 1.0 - abs((area - 160.0) / 160.0)
    s_area = float(np.clip(s_area, 0.0, 1.0))
    s_bh = float(np.clip((mean_bh - meanbh_lo) / 80.0, 0.0, 1.0))
    s_lin = float(np.clip((linearity - linearity_lo) / 0.5, 0.0, 1.0))
    s_asp = float(np.clip((aspect - aspect_lo) / 4.0, 0.0, 1.0))
    score01 = 0.35 * s_bh + 0.25 * s_lin + 0.20 * s_asp + 0.20 * s_area
    score01 = float(np.clip(score01, 0.0, 1.0))

    return True, score01, angle, linearity, (x0, y0, x1, y1), dbg


# -----------------------------
# Visualization
# -----------------------------
def draw_overlay(img_bgr, markers, parts, obvious_list=None, show_angles=False, show_ids=False):
    out = img_bgr.copy()
    out[markers == -1] = (0, 0, 255)  # watershed borders

    # draw all parts (green)
    for seg_id, est, _area in parts:
        bb = segment_bbox(markers, seg_id)
        if bb is None:
            continue
        x0, y0, x1, y1, xs, ys = bb
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

    # draw obvious (blue)
    if obvious_list:
        for item in obvious_list:
            x0, y0, x1, y1 = item["bbox"]
            sc = item["score"]
            ang = item["angle"]
            seg_id = item["seg_id"]

            cv2.rectangle(out, (x0, y0), (x1, y1), (255, 0, 0), 2)

            label = f"{sc:.2f}"
            if show_angles:
                label += f" {ang:+.0f}°"
            if show_ids:
                label = f"id={seg_id} " + label

            cv2.putText(
                out,
                label,
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
    ap.add_argument("--outdir", default=f"out_labws_{now.strftime('%d_%H_%M_%S')}")

    # LAB + blackhat
    ap.add_argument("--blur-ksize", type=int, default=5)
    ap.add_argument("--clahe-clip", type=float, default=2.0)
    ap.add_argument("--clahe-grid", type=int, default=8)

    ap.add_argument("--blackhat-ksize", type=int, default=41)
    ap.add_argument("--multi-scale", action="store_true")
    ap.add_argument("--small-blackhat-ksize", type=int, default=21)

    ap.add_argument("--no-chroma", action="store_true", help="Disable LAB chroma suppression")
    ap.add_argument("--chroma-gamma", type=float, default=1.2)
    ap.add_argument("--chroma-blur", type=int, default=0)
    ap.add_argument("--chroma-floor", type=float, default=0.05)

    # threshold
    ap.add_argument("--adaptive", action="store_true")
    ap.add_argument("--block-size", type=int, default=51)
    ap.add_argument("--C", type=int, default=2)
    ap.add_argument("--min-thresh", type=int, default=0)

    # morph
    ap.add_argument("--open-ksize", type=int, default=3)
    ap.add_argument("--open-iter", type=int, default=1)
    ap.add_argument("--close-iter", type=int, default=1)

    # line suppression
    ap.add_argument("--line-suppress", action="store_true")
    ap.add_argument("--line-len", type=int, default=75)
    ap.add_argument("--line-dilate", type=int, default=1)

    # pre-watershed CC filter
    ap.add_argument("--mask-min-area", type=int, default=45)
    ap.add_argument("--mask-max-aspect", type=float, default=12.0)
    ap.add_argument("--mask-min-extent", type=float, default=0.02)

    # watershed
    ap.add_argument("--dist-ratio", type=float, default=0.42)
    ap.add_argument("--bg-dilate", type=int, default=2)

    # counting
    ap.add_argument("--area-min", type=int, default=20)
    ap.add_argument("--area-single-max", type=int, default=520)
    ap.add_argument("--area-blob-min", type=int, default=750)

    # post-filter segments
    ap.add_argument("--seg-max-aspect", type=float, default=14.0)
    ap.add_argument("--seg-min-extent", type=float, default=0.04)
    ap.add_argument("--seg-max-extent", type=float, default=0.95)

    # obvious ants
    ap.add_argument("--obvious", action="store_true", help="Compute high-confidence ants")
    ap.add_argument("--ob-area-lo", type=int, default=55)
    ap.add_argument("--ob-area-hi", type=int, default=420)
    ap.add_argument("--ob-meanbh-lo", type=float, default=30.0)
    ap.add_argument("--ob-aspect-lo", type=float, default=1.6)
    ap.add_argument("--ob-extent-hi", type=float, default=0.85)
    ap.add_argument("--ob-linearity-lo", type=float, default=0.35)
    ap.add_argument("--ob-chroma-hi", type=float, default=18.0, help="Reject 'obvious' if mean LAB chroma > this")
    ap.add_argument("--ob-orient", choices=["pca", "skeleton"], default="pca")
    ap.add_argument("--ob-show-angles", action="store_true")
    ap.add_argument("--ob-show-ids", action="store_true")

    # debug listing
    ap.add_argument("--near-obvious", action="store_true", help="Print top segments by score-like criteria")
    ap.add_argument("--near-top", type=int, default=60)

    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Не могу прочитать: {args.image}")

    # LAB channels for chroma in obvious scoring
    _, A_u8, B_u8 = bgr_to_lab(img)

    # --- LAB blackhat + chroma suppression (score image) ---
    Lb, Lc, bh_big, bh_small, bh_used, w, bh_w = preprocess_lab_blackhat(
        img,
        blur_ksize=args.blur_ksize,
        clahe_clip=args.clahe_clip,
        clahe_grid=args.clahe_grid,
        blackhat_ksize=args.blackhat_ksize,
        multi_scale=args.multi_scale,
        small_blackhat_ksize=args.small_blackhat_ksize,
        use_chroma_suppress=(not args.no_chroma),
        chroma_gamma=args.chroma_gamma,
        chroma_blur=args.chroma_blur,
        chroma_floor=args.chroma_floor,
    )

    # --- threshold + clean ---
    raw, mask = make_initial_mask(
        bh_w,
        adaptive=args.adaptive,
        block_size=args.block_size,
        C=args.C,
        min_thresh=args.min_thresh,
        open_ksize=args.open_ksize,
        open_iter=args.open_iter,
        close_iter=args.close_iter,
    )

    # --- optional line suppression ---
    line_mask = None
    if args.line_suppress:
        mask, line_mask = suppress_long_lines(mask, line_len=args.line_len, dilate=args.line_dilate)

    # --- pre-watershed filtering ---
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

    total, median_single, singles, blobs, parts = count_from_markers(
        markers,
        area_min=args.area_min,
        area_single_max=args.area_single_max,
        area_blob_min=args.area_blob_min,
    )

    # --- post-filter segments by shape ---
    parts = filter_segments_by_shape(
        markers,
        parts,
        max_aspect=args.seg_max_aspect,
        min_extent=args.seg_min_extent,
        max_extent=args.seg_max_extent,
    )
    total = sum(est for _, est, _ in parts)

    # --- obvious ants ---
    obvious_list = []
    obvious_count = 0

    if args.obvious:
        for seg_id, est, area in parts:
            ok, sc, ang, lin, bbox, dbg = obvious_ant_score(
                bh_w,
                markers,
                seg_id,
                area,
                A_u8=A_u8,
                B_u8=B_u8,
                chroma_hi=args.ob_chroma_hi,
                orient=args.ob_orient,
                area_lo=args.ob_area_lo,
                area_hi=args.ob_area_hi,
                meanbh_lo=args.ob_meanbh_lo,
                aspect_lo=args.ob_aspect_lo,
                extent_hi=args.ob_extent_hi,
                linearity_lo=args.ob_linearity_lo,
            )
            if ok and bbox is not None:
                obvious_list.append(
                    {"seg_id": seg_id, "score": sc, "angle": ang, "linearity": lin, "bbox": bbox, "dbg": dbg}
                )
                obvious_count += 1

    # optional: print “near-obvious”
    if args.near_obvious:
        # rank by (how many hard-gates passed) then meanBH
        rows = []
        for seg_id, est, area in parts:
            ok, sc, ang, lin, bbox, dbg = obvious_ant_score(
                bh_w,
                markers,
                seg_id,
                area,
                A_u8=A_u8,
                B_u8=B_u8,
                chroma_hi=args.ob_chroma_hi,
                orient=args.ob_orient,
                area_lo=args.ob_area_lo,
                area_hi=args.ob_area_hi,
                meanbh_lo=args.ob_meanbh_lo,
                aspect_lo=args.ob_aspect_lo,
                extent_hi=args.ob_extent_hi,
                linearity_lo=args.ob_linearity_lo,
            )
            if bbox is None:
                continue

            passed = 0
            passed += int(args.ob_area_lo <= area <= args.ob_area_hi)
            passed += int(dbg["meanBH"] >= args.ob_meanbh_lo)
            passed += int(dbg["asp"] >= args.ob_aspect_lo)
            passed += int(dbg["ext"] <= args.ob_extent_hi)
            passed += int(dbg["lin"] >= args.ob_linearity_lo)
            # chroma gate counts too:
            passed += int(dbg["meanChr"] <= args.ob_chroma_hi)

            rows.append((passed, dbg["meanBH"], seg_id, dbg))

        rows.sort(key=lambda t: (t[0], t[1]), reverse=True)

        print(f"\n[Near-obvious] showing top {args.near_top} segments:")
        for passed, meanbh, seg_id, d in rows[: int(args.near_top)]:
            print(
                f"id={seg_id:4d} ok={passed}/6 area={d['area']:4d} meanBH={d['meanBH']:5.1f} "
                f"chr={d['meanChr']:4.1f} lin={d['lin']:.2f} ang={d['ang']:+6.0f}° "
                f"asp={d['asp']:.2f} ext={d['ext']:.2f} bb={d['bb']}"
            )

    overlay = draw_overlay(
        img,
        markers,
        parts,
        obvious_list=obvious_list,
        show_angles=args.ob_show_angles,
        show_ids=args.ob_show_ids,
    )

    # --- save debug ---
    os.makedirs(args.outdir, exist_ok=True)

    cv2.imwrite(os.path.join(args.outdir, "0_L_blur.png"), Lb)
    cv2.imwrite(os.path.join(args.outdir, "0_L_clahe.png"), Lc)
    cv2.imwrite(os.path.join(args.outdir, "1_blackhat_big.png"), bh_big)
    if args.multi_scale and bh_small is not None:
        cv2.imwrite(os.path.join(args.outdir, "1b_blackhat_small.png"), bh_small)
        cv2.imwrite(os.path.join(args.outdir, "1c_blackhat_used_max.png"), bh_used)

    if w is not None:
        cv2.imwrite(os.path.join(args.outdir, "1d_chroma_weight.png"), (np.clip(w, 0, 1) * 255).astype(np.uint8))

    cv2.imwrite(os.path.join(args.outdir, "1e_blackhat_weighted.png"), bh_w)
    cv2.imwrite(os.path.join(args.outdir, "2_mask_raw.png"), raw)
    cv2.imwrite(os.path.join(args.outdir, "3_mask_final.png"), mask)
    if args.line_suppress and line_mask is not None:
        cv2.imwrite(os.path.join(args.outdir, "3d_line_mask.png"), line_mask)

    cv2.imwrite(os.path.join(args.outdir, "4_dist_norm.png"), (dist_norm * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(args.outdir, "5_sure_fg.png"), sure_fg)
    cv2.imwrite(os.path.join(args.outdir, "6_overlay.png"), overlay)

    # --- print ---
    print(f"Оценка муравьёв (с учётом слипшихся): {total}")
    if args.obvious:
        print(f"Явных муравьёв (high-confidence): {obvious_count}")
    print(f"Median площадь 'одиночного' сегмента: {median_single}")
    print(f"Сегментов после фильтров: {len(parts)}")
    print(f"Смотри оверлей: {args.outdir}/6_overlay.png")


if __name__ == "__main__":
    main()

#(venv) PS C:\Users\Igor\PycharmProjects\study> python ant_counter.py photo.jpg `
#>>   --multi-scale --small-blackhat-ksize 21 `
#>>   --adaptive --block-size 71 --C 3 `
#>>   --line-suppress --line-len 75 --line-dilate 1 `
#>>   --dist-ratio 0.38 `
#>>   --mask-min-extent 0.04 --mask-min-area 55 `
#>>   --obvious --ob-show-angles --ob-show-ids `
#>>   --ob-orient skeleton `
#>>   --ob-meanbh-lo 22 --ob-linearity-lo 0.25 --ob-area-lo 45 `
#>>   --ob-chroma-hi 12 --chroma-gamma 1.8 `
#>>   --near-obvious
