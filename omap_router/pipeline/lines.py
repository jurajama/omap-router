"""Stage 8.5: thick black lines (walls, fences) -> barrier mask."""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from ..config import Settings, TerrainClass

_NEIGH = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)


def buildings(labels: np.ndarray, px_per_mm: float, cfg: Settings) -> np.ndarray:
    """Solid grey areas. Thin grey fringes (anti-aliased edges of black lines are
    often classified grey) are removed by an opening of ``building_min_width_mm``."""
    grey = (labels == TerrainClass.GREY).astype(np.uint8)
    k = max(3, int(round(cfg.building_min_width_mm * px_per_mm)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(grey, cv2.MORPH_OPEN, kernel).astype(bool)


def building_outline(labels: np.ndarray, px: int, px_per_mm: float, cfg: Settings) -> np.ndarray:
    """Black pixels within ``px`` of a building."""
    if px <= 0:
        return np.zeros(labels.shape, bool)
    b = buildings(labels, px_per_mm, cfg).astype(np.uint8)
    near = cv2.dilate(b, np.ones((2 * px + 1, 2 * px + 1), np.uint8)).astype(bool)
    return near & (labels == TerrainClass.BLACK)


def stroke_segments(black: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Skeletonise and split at junctions.

    Returns ``(skeleton, segment_labels, n_segments)``; junction pixels have label 0.
    """
    skel = skeletonize(black)
    nb = cv2.filter2D(skel.astype(np.uint8), -1, _NEIGH, borderType=cv2.BORDER_CONSTANT)
    junction = skel & (nb >= 3)
    seg = skel & ~junction
    lab, n = ndimage.label(seg, structure=np.ones((3, 3)))
    return skel, lab, n


def profile_fwhm(lightness: np.ndarray, black: np.ndarray, skel: np.ndarray) -> np.ndarray:
    """Stroke width (px) at every skeleton pixel, measured on the lightness image.

    The lightness profile is sampled across the stroke (normal to the local
    principal axis of the skeleton) and its full width at half
    depth is returned. Unlike counting black-classified pixels this keeps the
    anti-aliased edges, which on low-resolution scans are classified as
    brown/grey and make thick walls look thin. Returns 0 where no width could
    be measured.
    """
    ys, xs = np.nonzero(skel)
    out = np.zeros(skel.shape, np.float32)
    if len(ys) == 0:
        return out
    # local direction: principal axis of skeleton pixels in a 7x7 window
    sk = skel.astype(np.float32)
    yy, xx = np.mgrid[0 : skel.shape[0], 0 : skel.shape[1]].astype(np.float32)
    box = (7, 7)
    cnt = cv2.boxFilter(sk, -1, box, normalize=False)[ys, xs]
    mx = cv2.boxFilter(sk * xx, -1, box, normalize=False)[ys, xs] / cnt
    my = cv2.boxFilter(sk * yy, -1, box, normalize=False)[ys, xs] / cnt
    cxx = cv2.boxFilter(sk * xx * xx, -1, box, normalize=False)[ys, xs] / cnt - mx * mx
    cyy = cv2.boxFilter(sk * yy * yy, -1, box, normalize=False)[ys, xs] / cnt - my * my
    cxy = cv2.boxFilter(sk * xx * yy, -1, box, normalize=False)[ys, xs] / cnt - mx * my
    tangent = 0.5 * np.arctan2(2 * cxy, cxx - cyy)
    nx, ny = -np.sin(tangent), np.cos(tangent)  # normal direction (x, y)
    step = 0.25
    offs = np.arange(-4.0, 4.0 + 1e-6, step)
    px = xs[:, None] + nx[:, None] * offs
    py = ys[:, None] + ny[:, None] * offs
    prof = ndimage.map_coordinates(lightness.astype(np.float32), [py, px], order=1, mode="nearest")
    c = len(offs) // 2
    lmin = prof[:, c - 2 : c + 3].min(axis=1)
    bg = np.maximum(prof[:, :4].max(axis=1), prof[:, -4:].max(axis=1))
    below = prof <= ((lmin + bg) / 2)[:, None]
    # contiguous run of "dark" samples through the centre
    left = below[:, : c + 1][:, ::-1]
    right = below[:, c:]
    n_left = np.where(left.all(axis=1), left.shape[1], np.argmin(left, axis=1))
    n_right = np.where(right.all(axis=1), right.shape[1], np.argmin(right, axis=1))
    width = (n_left + n_right - 1) * step
    ok = below[:, c] & (bg - lmin > 10)
    out[ys, xs] = np.where(ok, width, 0.0)
    return out


def barrier_mask(
    labels: np.ndarray, px_per_mm: float, cfg: Settings, lightness: np.ndarray | None = None
) -> np.ndarray:
    """Black strokes wider than ``thick_line_mm`` (dilated back to stroke width).

    Width per stroke segment is the larger of two estimates: black pixel area
    / skeleton length (accurate on sharp, high-resolution scans), and, when
    ``lightness`` (CIE L*) is given, the median full width at half depth of the
    lightness profile, corrected for scan blur (``line_blur_px``).
    """
    black = labels == TerrainClass.BLACK
    black &= ~building_outline(labels, cfg.building_outline_px, px_per_mm, cfg)
    if not black.any():
        return np.zeros(labels.shape, bool)
    skel, seg_lab, n = stroke_segments(black)
    if n == 0:
        return np.zeros(labels.shape, bool)

    # assign every black pixel to its nearest skeleton pixel
    _, (ri, ci) = ndimage.distance_transform_edt(~skel, return_indices=True)
    owner = seg_lab[ri, ci]
    owner[~black] = 0
    # junction pixels belong to the nearest segment as well
    _, (sr, sc) = ndimage.distance_transform_edt(seg_lab == 0, return_indices=True)
    owner_j = seg_lab[sr, sc]
    owner = np.where((owner == 0) & black, owner_j, owner)
    owner[~black] = 0

    area = np.bincount(owner.ravel(), minlength=n + 1).astype(float)
    length = np.bincount(seg_lab.ravel(), minlength=n + 1).astype(float)
    width = np.divide(area, np.maximum(length, 1.0))
    width[0] = 0.0
    thick_px = cfg.thick_line_mm * px_per_mm
    # very short segments (junction stubs, blobs) get unreliable widths; require some length
    thick = (width >= thick_px) & (length >= 2)
    thick[0] = False
    barrier = thick[owner] & black

    if lightness is not None:
        # Profile widths are averaged along the skeleton in a small window rather
        # than per segment: on low-resolution scans thick strokes have many tiny
        # skeleton spurs that cut them into short, unreliable segments.
        fw = profile_fwhm(lightness, black, skel)
        valid = (fw > 0).astype(np.float32)
        win = (2 * cfg.line_window_px + 1,) * 2
        num = cv2.boxFilter(fw * valid, -1, win, normalize=False)
        den = cv2.boxFilter(valid, -1, win, normalize=False)
        local = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        # a blurred stroke of width w measures about sqrt(w^2 + blur^2)
        thick_skel = skel & (local >= np.hypot(thick_px, cfg.line_blur_px)) & (den >= cfg.line_min_length_px)
        barrier |= thick_skel[ri, ci] & black

    # drop compact components (text, point symbols)
    min_box = cfg.line_min_bbox_mm * px_per_mm
    comp, nc = ndimage.label(barrier, structure=np.ones((3, 3)))
    if nc:
        keep = np.zeros(nc + 1, bool)
        for i, sl in enumerate(ndimage.find_objects(comp), start=1):
            if sl is None:
                continue
            h = sl[0].stop - sl[0].start
            w = sl[1].stop - sl[1].start
            big, small = max(h, w), max(1, min(h, w))
            keep[i] = big >= min_box or big / small >= cfg.line_min_elongation_ratio and big >= 0.5 * min_box
        barrier = keep[comp]
    return barrier
