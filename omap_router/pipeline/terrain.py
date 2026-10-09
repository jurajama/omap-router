"""Stage 8.6: forbidden mask and cost grid (+ user edit polygons)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

from ..config import Settings, TerrainClass, class_cost
from ..models import EditPolygon
from .lines import building_outline


@dataclass
class CostGrid:
    cost: np.ndarray  # float32 (Hc, Wc); inf = forbidden
    cell_px: int  # fine pixels per cell side
    barrier_cells: np.ndarray  # bool (Hc, Wc)

    @property
    def forbidden(self) -> np.ndarray:
        return ~np.isfinite(self.cost)


def rasterize_polygons(
    edits: list[EditPolygon], kind: str, shape: tuple[int, int], offset: tuple[int, int] = (0, 0)
) -> np.ndarray:
    """Fill polygons of ``kind`` given in original-image pixels; ``offset`` is the crop origin."""
    m = np.zeros(shape, np.uint8)
    for e in edits:
        if e.kind != kind or len(e.points) < 3:
            continue
        pts = np.array([(x - offset[0], y - offset[1]) for x, y in e.points], np.float64)
        cv2.fillPoly(m, [np.rint(pts).astype(np.int32)], 1)
    return m.astype(bool)


def water_mask(labels: np.ndarray, px_per_mm: float, cfg: Settings) -> np.ndarray:
    """Blue areas; thin blue lines (magnetic north lines, streams) are removed by opening."""
    blue = (labels == TerrainClass.BLUE).astype(np.uint8)
    k = max(3, int(round(cfg.water_min_width_mm * px_per_mm)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(blue, cv2.MORPH_OPEN, kernel).astype(bool)


def outside_map_mask(labels: np.ndarray, px_per_mm: float, cfg: Settings) -> np.ndarray:
    """Blank paper around the mapped area.

    Map content (anything that is not white) is closed with a kernel of
    ``map_gap_mm`` and holes are filled; white regions that stay outside are
    unmapped. White areas inside the map (open land, forest with contours,
    courtyards) are enclosed by content and therefore kept. Thin blue/green
    lines (magnetic north lines) run across the blank paper, so they do not
    count as content.
    """
    white = labels == TerrainClass.WHITE
    content = ~white
    k_thin = max(3, int(round(cfg.water_min_width_mm * px_per_mm)))
    thin_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_thin, k_thin))
    for cls in (TerrainClass.BLUE, TerrainClass.GREEN):
        m = (labels == cls).astype(np.uint8)
        thin = m.astype(bool) & ~cv2.morphologyEx(m, cv2.MORPH_OPEN, thin_kernel).astype(bool)
        content &= ~thin
    if not content.any():
        return np.zeros(labels.shape, bool)
    k = max(3, int(round(cfg.map_gap_mm * px_per_mm)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    # pad so content near the image border is not closed against the edge
    pad = k
    padded = cv2.copyMakeBorder(content.astype(np.uint8), pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    closed = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, kernel)[pad:-pad, pad:-pad].astype(bool)
    mapped = ndimage.binary_fill_holes(closed)
    return white & ~mapped


def auto_forbidden(
    labels: np.ndarray,
    barrier: np.ndarray,
    px_per_mm: float,
    cfg: Settings,
    hatch: np.ndarray | None = None,
    outside: np.ndarray | None = None,
) -> np.ndarray:
    """Forbidden mask from map content only: buildings (+outline), olive, water,
    barriers, purple-hatched out-of-bounds areas and blank paper outside the map."""
    forb = (labels == TerrainClass.GREY) | (labels == TerrainClass.OLIVE)
    if hatch is not None:
        forb |= hatch
    if outside is not None:
        forb |= outside
    forb |= building_outline(labels, 1)
    forb |= water_mask(labels, px_per_mm, cfg)
    forb |= barrier
    return forb


def fine_cost(labels: np.ndarray, cfg: Settings) -> np.ndarray:
    lut = np.full(256, cfg.cost_unknown, np.float32)
    for cls, c in class_cost(cfg).items():
        lut[int(cls)] = c
    cost = lut[labels]
    # forbidden classes are handled by the forbidden mask; give them a finite
    # cost here so "allow" polygons on top of them become passable
    cost[~np.isfinite(cost)] = cfg.cost_unknown
    return cost


def apply_edits(
    forbidden: np.ndarray,
    barrier: np.ndarray,
    edits: list[EditPolygon],
    offset: tuple[int, int] = (0, 0),
) -> tuple[np.ndarray, np.ndarray]:
    """Add user forbid polygons, then subtract allow polygons (applied last)."""
    shape = forbidden.shape
    forbid = rasterize_polygons(edits, "forbid", shape, offset)
    allow = rasterize_polygons(edits, "allow", shape, offset)
    forb = (forbidden | forbid) & ~allow
    bar = (barrier | forbid) & ~allow
    return forb, bar


def block_diagonal_leaks(forbidden: np.ndarray) -> np.ndarray:
    """Close diagonal gaps so 8-connected paths cannot slip between two forbidden
    cells that only touch at a corner (e.g. a one pixel wide diagonal wall).
    This is not a morphological closing: only exact corner contacts are filled."""
    f = forbidden.copy()
    a = forbidden[:-1, 1:] & forbidden[1:, :-1]  # anti-diagonal pair
    f[:-1, :-1] |= a
    b = forbidden[:-1, :-1] & forbidden[1:, 1:]  # main diagonal pair
    f[:-1, 1:] |= b
    return f


def cost_grid(cost: np.ndarray, forbidden: np.ndarray, barrier: np.ndarray, cell_px: int) -> CostGrid:
    """Downsample to ``cell_px`` cells.

    A cell is forbidden only if *all* its pixels are forbidden (keeps narrow
    passages open), or if a barrier line crosses it (keeps thin walls closed).
    Passable cell cost = mean cost of its passable pixels.
    """
    k = max(1, int(cell_px))
    h, w = cost.shape
    if k == 1:
        out = cost.astype(np.float32).copy()
        out[block_diagonal_leaks(forbidden | barrier)] = np.inf
        return CostGrid(out, 1, barrier.copy())
    hc, wc = -(-h // k), -(-w // k)
    ph, pw = hc * k - h, wc * k - w
    c = np.pad(cost.astype(np.float32), ((0, ph), (0, pw)), constant_values=0)
    f = np.pad(forbidden, ((0, ph), (0, pw)), constant_values=True)
    b = np.pad(barrier, ((0, ph), (0, pw)), constant_values=False)
    c = c.reshape(hc, k, wc, k)
    f = f.reshape(hc, k, wc, k)
    b = b.reshape(hc, k, wc, k)
    passable = ~f
    n_pass = passable.sum(axis=(1, 3))
    s = (c * passable).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (s / n_pass).astype(np.float32)
    bar_cells = b.any(axis=(1, 3))
    mean[block_diagonal_leaks((n_pass == 0) | bar_cells)] = np.inf
    return CostGrid(mean, k, bar_cells)


def cell_size_px(px_per_mm: float, cfg: Settings) -> int:
    return max(1, int(round(cfg.cell_mm * px_per_mm)))
