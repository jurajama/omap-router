"""Stage 8.3: course overprint (purple) mask and removal from the terrain labels."""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from ..config import TerrainClass


def purple_mask(labels: np.ndarray) -> np.ndarray:
    return labels == TerrainClass.PURPLE


def remove_purple(labels: np.ndarray, purple: np.ndarray, dilate_px: int = 0) -> np.ndarray:
    """Replace overprint pixels with the label of the nearest non-purple pixel.

    ``dilate_px`` grows the mask first so anti-aliased fringes (purple blended
    with the background, often classified as grey/brown) are replaced too.
    """
    mask = purple
    if dilate_px > 0 and mask.any():
        k = 2 * dilate_px + 1
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    if not mask.any() or mask.all():
        return labels.copy()
    _, (ri, ci) = ndimage.distance_transform_edt(mask, return_indices=True)
    out = labels.copy()
    out[mask] = labels[ri[mask], ci[mask]]
    return out


def hatch_mask(purple: np.ndarray, px_per_mm: float, cfg) -> np.ndarray:
    """Purple cross-hatched areas: union of two detectors (see below)."""
    return _hatch_by_components(purple, px_per_mm, cfg) | _hatch_by_density(purple, px_per_mm, cfg)


def _hatch_by_density(purple: np.ndarray, px_per_mm: float, cfg) -> np.ndarray:
    """Fine or fragmented hatching: dense purple texture that encloses many gaps.

    Leg lines and circles are too sparse for the density threshold; digits are
    dense but enclose at most a few gaps.
    """
    out = np.zeros(purple.shape, bool)
    if not purple.any():
        return out
    win = max(5, int(round(cfg.hatch_window_mm * px_per_mm)) | 1)
    dens = cv2.boxFilter(purple.astype(np.float32), -1, (win, win))
    cand = (dens >= cfg.hatch_density_min_ratio).astype(np.uint8)
    # leg lines touching the area leave thin protrusions in the density map
    k = max(3, int(round(cfg.hatch_open_mm * px_per_mm)) | 1)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    min_side = cfg.hatch_min_size_mm * px_per_mm
    max_hole = (cfg.hatch_hole_max_mm * px_per_mm) ** 2
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    for k in range(1, n):
        x, y, w, h, _area = stats[k]
        if w < min_side or h < min_side:
            continue
        region = lab[y : y + h, x : x + w] == k
        inner = cv2.erode(region.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        gaps = (~purple[y : y + h, x : x + w] & region).astype(np.uint8)
        ng, glab, gstats, _ = cv2.connectedComponentsWithStats(gaps, connectivity=4)
        enclosed = 0
        for g in range(1, ng):
            if gstats[g, cv2.CC_STAT_AREA] > max_hole:
                continue
            if inner[glab == g].all():
                enclosed += 1
        if enclosed >= cfg.hatch_min_holes:
            out[y : y + h, x : x + w] |= region
    return out


def _hatch_by_components(purple: np.ndarray, px_per_mm: float, cfg) -> np.ndarray:
    """Areas cross-hatched with purple (out-of-bounds overprint).

    A hatch is a purple component enclosing many small holes of similar size;
    leg lines, circles and digits have at most a couple of holes. Each hatch
    component is filled (holes included) and stripped of attached leg lines by
    an opening wider than a line but narrower than the hatched area.
    """
    out = np.zeros(purple.shape, bool)
    m = purple.astype(np.uint8)
    if not m.any():
        return out
    min_hole = (cfg.hatch_hole_min_mm * px_per_mm) ** 2
    max_hole = (cfg.hatch_hole_max_mm * px_per_mm) ** 2
    min_side = cfg.hatch_min_size_mm * px_per_mm
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    for k in range(1, n):
        x, y, w, h, _area = stats[k]
        if w < min_side or h < min_side:
            continue
        comp = (lab[y : y + h, x : x + w] == k).astype(np.uint8)
        comp = cv2.copyMakeBorder(comp, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        contours, hier = cv2.findContours(comp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is None:
            continue
        holes = [c for c, hh in zip(contours, hier[0], strict=True) if hh[3] >= 0]
        sizes = np.array([cv2.contourArea(c) for c in holes])
        is_cell = (sizes >= min_hole) & (sizes <= max_hole)
        small = sizes[is_cell]
        if len(small) < cfg.hatch_min_holes:
            continue
        # fill only hatch cells, not e.g. a control circle attached by a leg line
        filled = comp.copy()
        cv2.drawContours(filled, [c for c, ok in zip(holes, is_cell, strict=True) if ok], -1, 1, -1)
        # remove leg lines and other thin attachments; keep the solid hatched block
        cell = max(3, int(round(2.0 * np.sqrt(np.median(small)))), int(round(cfg.hatch_open_mm * px_per_mm)) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cell, cell))
        filled = cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel)
        out[y : y + h, x : x + w] |= filled[1:-1, 1:-1].astype(bool)
    return out
