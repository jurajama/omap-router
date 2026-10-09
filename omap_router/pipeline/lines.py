"""Stage 8.5: thick black lines (walls, fences) -> barrier mask."""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from ..config import Settings, TerrainClass

_NEIGH = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)


def building_outline(labels: np.ndarray, px: int) -> np.ndarray:
    """Black pixels within ``px`` of a building (grey)."""
    grey = (labels == TerrainClass.GREY).astype(np.uint8)
    if px <= 0:
        return np.zeros(labels.shape, bool)
    near = cv2.dilate(grey, np.ones((2 * px + 1, 2 * px + 1), np.uint8)).astype(bool)
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


def barrier_mask(labels: np.ndarray, px_per_mm: float, cfg: Settings) -> np.ndarray:
    """Black strokes wider than ``thick_line_mm`` (dilated back to stroke width)."""
    black = labels == TerrainClass.BLACK
    black &= ~building_outline(labels, cfg.building_outline_px)
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
