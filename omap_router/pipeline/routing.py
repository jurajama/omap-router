"""Stage 8.7: least-cost paths per leg."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from shapely.geometry import LineString
from skimage.graph import MCP_Geometric

from .terrain import CostGrid


@dataclass
class LegResult:
    ok: bool
    path_xy: list[tuple[float, float]]  # fine-pixel coords of the routed image (cropped space)
    cost: float
    message: str | None = None


def xy_to_rc(x: float, y: float, cell_px: int) -> tuple[int, int]:
    """Pixel ``(x, y)`` -> grid ``(row, col)``."""
    return int(math.floor(y / cell_px)), int(math.floor(x / cell_px))


def rc_to_xy(r: int, c: int, cell_px: int) -> tuple[float, float]:
    """Grid ``(row, col)`` -> pixel ``(x, y)`` at the cell centre."""
    return (c + 0.5) * cell_px, (r + 0.5) * cell_px


class Snapper:
    """Snap points to the nearest passable cell (stored coordinates are not moved).

    Control circles are centred on features; the centre often lies in a tiny
    passable pocket (inside a symbol or between outlines). Cells of passable
    components smaller than ``min_component_cells`` are therefore skipped.
    """

    def __init__(self, grid: CostGrid, min_component_cells: int = 0):
        self.grid = grid
        target = ~grid.forbidden
        if min_component_cells > 0 and target.any():
            lab, n = ndimage.label(target, structure=np.ones((3, 3)))
            size = np.bincount(lab.ravel(), minlength=n + 1)
            size[0] = 0
            big = size >= min(min_component_cells, int(size.max()))
            target = big[lab]
        if not target.any():
            self._idx = None
        else:
            _, self._idx = ndimage.distance_transform_edt(~target, return_indices=True)

    def snap(self, x: float, y: float) -> tuple[int, int] | None:
        if self._idx is None:
            return None
        h, w = self.grid.cost.shape
        r, c = xy_to_rc(x, y, self.grid.cell_px)
        r = min(max(r, 0), h - 1)
        c = min(max(c, 0), w - 1)
        return int(self._idx[0][r, c]), int(self._idx[1][r, c])


def segment_clear(grid: CostGrid, a: tuple[float, float], b: tuple[float, float]) -> bool:
    """True if the straight segment a->b (pixel coords) only crosses passable cells."""
    forb = grid.forbidden
    h, w = forb.shape
    k = grid.cell_px
    d = math.hypot(b[0] - a[0], b[1] - a[1])
    n = max(2, int(math.ceil(d / (0.25 * k))) + 1)
    t = np.linspace(0.0, 1.0, n)
    xs = a[0] + (b[0] - a[0]) * t
    ys = a[1] + (b[1] - a[1]) * t
    rows = np.clip(np.floor(ys / k).astype(int), 0, h - 1)
    cols = np.clip(np.floor(xs / k).astype(int), 0, w - 1)
    return not forb[rows, cols].any()


def path_clear(grid: CostGrid, pts: list[tuple[float, float]]) -> bool:
    return all(segment_clear(grid, pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def simplify(grid: CostGrid, pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(pts) < 3:
        return pts
    simp = list(LineString(pts).simplify(0.5 * grid.cell_px, preserve_topology=False).coords)
    simp = [(float(x), float(y)) for x, y in simp]
    return simp if path_clear(grid, simp) else pts


def path_length(pts: list[tuple[float, float]]) -> float:
    return float(sum(math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1)))


def route_leg(grid: CostGrid, snapper: Snapper, a: tuple[float, float], b: tuple[float, float]) -> LegResult:
    """Route from a to b (pixel coords in the grid's image space)."""
    sa, sb = snapper.snap(*a), snapper.snap(*b)
    if sa is None or sb is None:
        return LegResult(False, [], math.inf, "no passable terrain")
    if sa == sb:
        p = rc_to_xy(*sa, grid.cell_px)
        return LegResult(True, [p, p], 0.0)
    mcp = MCP_Geometric(grid.cost, fully_connected=True)
    costs, _ = mcp.find_costs([sa], [sb], find_all_ends=True)
    total = float(costs[sb])
    if not np.isfinite(total):
        return LegResult(False, [], math.inf, "unreachable: destination is enclosed by forbidden areas")
    cells = mcp.traceback(sb)
    pts = [rc_to_xy(r, c, grid.cell_px) for r, c in cells]
    pts = simplify(grid, pts)
    return LegResult(True, pts, total * grid.cell_px)
