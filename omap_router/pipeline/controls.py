"""Stage 8.4: scale-free control circle, start and finish detection + ordering.

All coordinates in this module are ``(x, y)`` pixels of the (cropped) image the
mask was computed from; ``run.py`` converts them to original-image space.
"""

from __future__ import annotations

import itertools
import logging
import math
import re
import shutil
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import Settings

log = logging.getLogger(__name__)

N_ANGLES = 72


@dataclass
class Circle:
    x: float
    y: float
    r: float
    support: float = 0.0


@dataclass
class Detection:
    r0: float | None
    px_per_mm: float | None
    controls: list[Circle] = field(default_factory=list)  # ordered if ``ordered``
    start: tuple[float, float, float] | None = None  # x, y, circumradius
    finish: Circle | None = None  # outer radius
    order: list[int] | None = None  # indices into ``nodes`` (start, controls..., finish)
    nodes: list[tuple[str, float, float, float]] = field(default_factory=list)  # kind, x, y, r
    ordered: bool = False
    messages: list[str] = field(default_factory=list)
    candidates: list[Circle] = field(default_factory=list)  # verified wide-search circles
    codes: dict[int, int] = field(default_factory=dict)  # node index -> printed control code


# ---------------------------------------------------------------- primitives


def _sample(mask: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    xi = np.rint(xs).astype(np.int64)
    yi = np.rint(ys).astype(np.int64)
    inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    out = np.zeros(xs.shape, bool)
    out[inside] = mask[yi[inside], xi[inside]]
    return out


def ring_support(mask: np.ndarray, x: float, y: float, r: float, tol: float, n: int = N_ANGLES) -> float:
    """Fraction of ``n`` angles with a mask pixel within ``r ± tol``."""
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    offs = np.arange(-tol, tol + 0.01, 0.5)
    rr = r + offs[:, None]
    xs = x + rr * np.cos(ang)[None, :]
    ys = y + rr * np.sin(ang)[None, :]
    hit = _sample(mask, xs, ys).any(axis=0)
    return float(hit.mean())


def sector_coverage(mask: np.ndarray, x: float, y: float, r: float, tol: float, sectors: int = 8) -> float:
    """Fraction of angular sectors where at least a quarter of the angles hit the ring.

    Separates real (possibly cut-open) circles from tangent straight lines, whose
    hits are concentrated in one or two sectors.
    """
    ang = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)
    offs = np.arange(-tol, tol + 0.01, 0.5)
    rr = r + offs[:, None]
    hit = _sample(mask, x + rr * np.cos(ang)[None, :], y + rr * np.sin(ang)[None, :]).any(axis=0)
    per = hit.reshape(sectors, -1).mean(axis=1)
    return float((per >= 0.25).mean())


def ring_support_batch(
    mask: np.ndarray, xs: np.ndarray, ys: np.ndarray, rs: np.ndarray, tol: float, n: int = N_ANGLES
) -> np.ndarray:
    """Vectorised ``ring_support`` for many (x, y, r) triples."""
    xs, ys, rs = (np.asarray(v, float).ravel() for v in (xs, ys, rs))
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    offs = np.arange(-tol, tol + 0.01, 0.5)
    rr = rs[:, None, None] + offs[None, :, None]  # (k, o, 1)
    px = xs[:, None, None] + rr * np.cos(ang)[None, None, :]
    py = ys[:, None, None] + rr * np.sin(ang)[None, None, :]
    hit = _sample(mask, px, py).any(axis=1)
    return hit.mean(axis=1)


def interior_fill(mask: np.ndarray, x: float, y: float, r: float) -> float:
    """Fraction of mask pixels inside the disc of radius 0.6 r."""
    h, w = mask.shape
    rr = 0.6 * r
    x0, x1 = int(max(0, math.floor(x - rr))), int(min(w, math.ceil(x + rr) + 1))
    y0, y1 = int(max(0, math.floor(y - rr))), int(min(h, math.ceil(y + rr) + 1))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    yy, xx = np.mgrid[y0:y1, x0:x1]
    disc = (xx - x) ** 2 + (yy - y) ** 2 <= rr * rr
    if not disc.any():
        return 0.0
    return float(mask[y0:y1, x0:x1][disc].mean())


def _ring_tol(r: float, cfg: Settings) -> float:
    return max(cfg.ring_tolerance_min_px, cfg.ring_tolerance_ratio * r)


def _hough(mask: np.ndarray, rmin: int, rmax: int, cfg: Settings, min_dist: float) -> list[Circle]:
    img = cv2.GaussianBlur(mask.astype(np.uint8) * 255, (5, 5), 1.2)
    rmin = max(1, int(rmin))
    rmax = max(rmin + 1, int(rmax))
    res = cv2.HoughCircles(
        img,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=max(1.0, min_dist),
        param1=cfg.hough_param1,
        param2=cfg.hough_param2,
        minRadius=rmin,
        maxRadius=rmax,
    )
    if res is None:
        return []
    return [Circle(float(x), float(y), float(r)) for x, y, r in res[0]]


def band_stats(mask: np.ndarray, x: float, y: float, r: float) -> tuple[float, float, float]:
    """Radial statistics around a circle of radius ``r``.

    Returns ``(inner, outer, thickness)``: fraction of angles with purple in the
    band 0.5–0.7 r (inside the ring), in the band 1.3–1.5 r (outside), and the
    median radial ink thickness relative to ``r``. A printed control circle is
    a thin ring with empty surroundings, unlike digits, hatching or line
    crossings.
    """
    ang = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)
    rr = np.arange(0.5 * r, 1.5 * r + 0.01, 0.5)
    xs = x + rr[:, None] * np.cos(ang)[None, :]
    ys = y + rr[:, None] * np.sin(ang)[None, :]
    p = _sample(mask, xs, ys)
    inner = float(p[rr < 0.7 * r].any(axis=0).mean())
    outer = float(p[rr > 1.3 * r].any(axis=0).mean())
    thickness = float(np.median(p.sum(axis=0) * 0.5) / r)
    return inner, outer, thickness


def _verify(mask: np.ndarray, cands: list[Circle], cfg: Settings) -> list[Circle]:
    """Ring support filter, local polish, then strict shape checks."""
    out = []
    seen: set[tuple[int, int, int]] = set()
    for c in cands:
        key = (int(round(c.x)), int(round(c.y)), int(round(c.r)))
        if key in seen:
            continue
        seen.add(key)
        s = ring_support(mask, c.x, c.y, c.r, _ring_tol(c.r, cfg))
        if s < cfg.ring_support_min_ratio:
            continue
        p = _polish(mask, Circle(c.x, c.y, c.r, s))
        tight = ring_support(mask, p.x, p.y, p.r, max(1.0, cfg.ring_tight_tolerance_ratio * p.r))
        if tight < cfg.ring_support_min_ratio:
            continue
        if interior_fill(mask, p.x, p.y, p.r) > cfg.filled_interior_max_ratio:
            continue
        tol = max(1.0, cfg.ring_tight_tolerance_ratio * p.r)
        if sector_coverage(mask, p.x, p.y, p.r, tol, cfg.ring_sectors) < cfg.ring_sector_min_ratio:
            continue
        inner, outer, thick = band_stats(mask, p.x, p.y, p.r)
        if inner > cfg.ring_inner_max_ratio or outer > cfg.ring_outer_max_ratio or thick > cfg.ring_max_thickness_ratio:
            continue
        out.append(Circle(p.x, p.y, p.r, tight))
    return out


def _nms(circles: list[Circle], dist: float | None = None) -> list[Circle]:
    """Keep the best-supported circle among those with centres closer than ``dist`` (or own r)."""
    keep: list[Circle] = []
    for c in sorted(circles, key=lambda c: (-c.support, -c.r)):
        d = dist if dist is not None else c.r
        if all(math.hypot(c.x - k.x, c.y - k.y) >= d for k in keep):
            keep.append(c)
    return keep


def _polish(mask: np.ndarray, c: Circle) -> Circle:
    """Local search of centre/radius maximising tight (±1 px) ring support."""
    step = max(1.0, 0.05 * c.r)
    d = np.array([-2, -1, 0, 1, 2]) * step
    dr = np.array([-2, -1, -0.5, 0, 0.5, 1, 2]) * step
    gx, gy, gr = np.meshgrid(c.x + d, c.y + d, np.maximum(1.0, c.r + dr), indexing="ij")
    s = ring_support_batch(mask, gx, gy, gr, 1.0, n=36)
    # prefer the unmodified circle on ties
    centre = int(np.flatnonzero((gx.ravel() == c.x) & (gy.ravel() == c.y) & (gr.ravel() == c.r))[0])
    i = centre if s[centre] >= s.max() - 1e-9 else int(np.argmax(s))
    return Circle(float(gx.ravel()[i]), float(gy.ravel()[i]), float(gr.ravel()[i]), float(s[i]))


# ---------------------------------------------------------------- radius


def estimate_r0(mask: np.ndarray, cfg: Settings) -> tuple[float | None, list[Circle]]:
    """Wide Hough search + verification + radius consensus.

    Returns ``(r0, verified_circles)``; ``r0`` is None if fewer than
    ``min_circles_for_r0`` circles agree on a radius.
    """
    h, w = mask.shape
    rmax = max(cfg.hough_min_radius_px + 1, int(min(h, w) * cfg.hough_max_radius_ratio))
    cands = _hough(mask, cfg.hough_min_radius_px, rmax, cfg, min_dist=cfg.hough_min_radius_px)
    verified = _nms(_verify(mask, cands, cfg))
    if not verified:
        return None, []
    radii = np.array([c.r for c in verified])
    sup = np.array([c.support for c in verified])
    best_score, best_r, best_n = 0.0, None, 0
    for r in radii:
        near = np.abs(radii - r) <= 0.08 * r
        score = float(sup[near].sum())
        if score > best_score:
            best_score = score
            best_n = int(near.sum())
            best_r = float(np.average(radii[near], weights=sup[near]))
    log.info("wide search: %d candidates, %d verified, mode r=%.1f (n=%d)", len(cands), len(verified), best_r or 0, best_n)
    if best_n < cfg.min_circles_for_r0:
        return None, verified
    return best_r, verified


# ---------------------------------------------------------------- finish / start


def _radial_profile(mask: np.ndarray, x: float, y: float, radii: np.ndarray) -> np.ndarray:
    radii = np.asarray(radii, float)
    if radii.size == 0:
        return np.zeros(0)
    ones = np.ones_like(radii)
    return ring_support_batch(mask, x * ones, y * ones, radii, 0.75)


def _finish_score(mask: np.ndarray, x: float, y: float, r0: float, cfg: Settings) -> tuple[float, float, float]:
    """Returns (score, r_inner, r_outer); score 0 if no concentric pair."""
    ppm = 2 * r0 / cfg.control_diameter_mm
    r_in = cfg.finish_inner_diameter_mm / 2 * ppm
    r_out = cfg.finish_outer_diameter_mm / 2 * ppm
    tol = cfg.finish_radius_tol_ratio
    radii = np.arange(r_in * (1 - tol), r_out * (1 + tol) + 0.5, 0.5)
    prof = _radial_profile(mask, x, y, radii)
    in_win = (radii >= r_in * (1 - tol)) & (radii <= r_in * (1 + tol))
    out_win = (radii >= r_out * (1 - tol)) & (radii <= r_out * (1 + tol))
    if not in_win.any() or not out_win.any():
        return 0.0, 0.0, 0.0
    i_in = np.flatnonzero(in_win)[np.argmax(prof[in_win])]
    i_out = np.flatnonzero(out_win)[np.argmax(prof[out_win])]
    ri, ro = radii[i_in], radii[i_out]
    ratio = ro / max(ri, 1e-6)
    nominal = cfg.finish_outer_diameter_mm / cfg.finish_inner_diameter_mm
    if not (nominal * (1 - tol) <= ratio <= nominal * (1 + tol)) or ro - ri < 0.2 * r0:
        return 0.0, 0.0, 0.0
    # there must be a clear gap between the rings and empty space inside/outside
    between = (radii > ri + 1.0) & (radii < ro - 1.0)
    if not between.any() or prof[between].min() > 0.75 * min(prof[i_in], prof[i_out]):
        return 0.0, 0.0, 0.0
    s = min(prof[i_in], prof[i_out])
    if s < cfg.ring_support_min_ratio:
        return 0.0, 0.0, 0.0
    inside = _radial_profile(mask, x, y, np.arange(0.4 * ri, 0.75 * ri, 0.5))
    outside = _radial_profile(mask, x, y, np.arange(1.25 * ro, 1.45 * ro, 0.5))
    if inside.max(initial=0) > cfg.ring_inner_max_ratio or outside.max(initial=0) > cfg.ring_outer_max_ratio:
        return 0.0, 0.0, 0.0
    return float(s), float(ri), float(ro)


def find_finish(mask: np.ndarray, centers: list[Circle], r0: float, cfg: Settings) -> Circle | None:
    best: tuple[float, Circle] | None = None
    ppm = 2 * r0 / cfg.control_diameter_mm
    r_in = cfg.finish_inner_diameter_mm / 2 * ppm
    r_out = cfg.finish_outer_diameter_mm / 2 * ppm
    tol = cfg.finish_radius_tol_ratio
    win_in = np.arange(r_in * (1 - tol), r_in * (1 + tol), 1.0)
    win_out = np.arange(r_out * (1 - tol), r_out * (1 + tol), 1.0)
    reach = max(2, int(round(cfg.finish_center_tol_ratio * r0 / 2)))
    tested: set[tuple[int, int]] = set()
    for c in centers:
        key = (int(c.x // 2), int(c.y // 2))
        if key in tested:
            continue
        tested.add(key)
        # cheap pre-checks: empty centre, then both rings present
        if interior_fill(mask, c.x, c.y, 0.75 * r_in) > 0.1:
            continue
        ones_in, ones_out = np.ones_like(win_in), np.ones_like(win_out)
        pre = 0.8 * cfg.ring_support_min_ratio
        if ring_support_batch(mask, c.x * ones_in, c.y * ones_in, win_in, 1.0, n=36).max() < pre:
            continue
        if ring_support_batch(mask, c.x * ones_out, c.y * ones_out, win_out, 1.0, n=36).max() < pre:
            continue
        for dy in range(-reach, reach + 1):
            for dx in range(-reach, reach + 1):
                s2, ri, ro = _finish_score(mask, c.x + dx, c.y + dy, r0, cfg)
                if s2 > 0 and (best is None or s2 > best[0]):
                    best = (s2, Circle(c.x + dx, c.y + dy, ro, s2))
    return best[1] if best else None


def _triangle_vertices(cx: float, cy: float, side: float, theta: float) -> np.ndarray:
    rad = side / math.sqrt(3)
    ang = np.deg2rad(theta + np.array([-90.0, 30.0, 150.0]))
    return np.stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)], axis=1)


def _triangle_ok(mask: np.ndarray, cx: float, cy: float, side: float, theta: float) -> bool:
    """Interior empty and little ink just outside the edges (attached leg lines allowed)."""
    inr = side / (2 * math.sqrt(3))
    if interior_fill(mask, cx, cy, inr / 0.6 * 0.5) > 0.1:
        return False
    v = _triangle_vertices(cx, cy, side, theta)
    xs, ys = [], []
    for i in range(3):
        a, b = v[i], v[(i + 1) % 3]
        mid = (a + b) / 2
        out = mid - np.array([cx, cy])
        out /= np.linalg.norm(out)
        for t in np.linspace(0.25, 0.75, 9):
            p = a + (b - a) * t + out * max(2.0, 0.35 * inr)
            xs.append(p[0])
            ys.append(p[1])
    return _sample(mask, np.array(xs), np.array(ys)).mean() <= 0.25


def find_start(
    mask: np.ndarray, r0: float, cfg: Settings, exclude: list[tuple[float, float, float]]
) -> tuple[float, float, float] | None:
    """Chamfer-match rotated triangle outlines of side ~ start_side_mm.

    Robust to gaps in the outline and to leg lines attached at a vertex.
    Returns (cx, cy, circumradius).
    """
    ppm = 2 * r0 / cfg.control_diameter_mm
    side_nom = cfg.start_side_mm * ppm
    tol = cfg.start_side_tol_ratio
    trunc = max(3.0, 0.15 * side_nom)
    dt_full = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
    dt = np.minimum(dt_full, trunc).astype(np.float32)
    # the triangle centre is empty: no ink within half the smallest inradius
    min_inr = side_nom * (1 - tol) / (2 * math.sqrt(3))
    empty_centre = dt_full > 0.5 * min_inr
    max_score = max(1.0, 0.04 * side_nom)
    cands: list[tuple[float, float, float, float, float]] = []  # score, x, y, side, theta
    for scale in np.linspace(1 - tol, 1 + tol, 5):
        side = side_nom * scale
        rad = side / math.sqrt(3)
        k = int(math.ceil(rad)) + 2
        for theta in np.arange(0.0, 120.0, 6.0):
            tpl = np.zeros((2 * k + 1, 2 * k + 1), np.uint8)
            v = _triangle_vertices(k, k, side, theta)
            cv2.polylines(tpl, [np.rint(v).astype(np.int32)], True, 1, 1)
            kern = tpl.astype(np.float32) / tpl.sum()
            score = cv2.filter2D(dt, cv2.CV_32F, kern, borderType=cv2.BORDER_CONSTANT)
            low = (score <= max_score) & empty_centre
            if not low.any():
                continue
            _, lab = cv2.connectedComponents(low.astype(np.uint8), connectivity=8)
            ys, xs = np.nonzero(low)
            vals, labs = score[ys, xs], lab[ys, xs]
            order = np.lexsort((vals, labs))
            first = order[np.r_[True, labs[order][1:] != labs[order][:-1]]]
            for j in first:
                cands.append((float(vals[j]), float(xs[j]), float(ys[j]), side, theta))
    for sc, x, y, side, theta in sorted(cands):
        if any(math.hypot(x - ex, y - ey) < er for ex, ey, er in exclude):
            continue
        if _triangle_ok(mask, x, y, side, theta):
            log.info("start triangle at (%.0f, %.0f) side=%.1f score=%.2f", x, y, side, sc)
            return (x, y, side / math.sqrt(3))
    return None


# ---------------------------------------------------------------- ordering


def _clean_for_lines(mask: np.ndarray, nodes: list[tuple[str, float, float, float]], r0: float) -> np.ndarray:
    m = mask.astype(np.uint8).copy()
    thick = max(3, int(round(0.25 * r0)))
    for kind, x, y, r in nodes:
        if kind == "finish":
            # both rings
            cv2.circle(m, (int(round(x)), int(round(y))), int(round(r)), 0, thick)
            cv2.circle(m, (int(round(x)), int(round(y))), int(round(r * 5 / 7)), 0, thick)
        elif kind == "start":
            cv2.circle(m, (int(round(x)), int(round(y))), int(round(r * 1.1)), 0, -1)
        else:
            cv2.circle(m, (int(round(x)), int(round(y))), int(round(r)), 0, thick)
    return m.astype(bool)


def line_support(mask: np.ndarray, a: tuple[float, float, float], b: tuple[float, float, float], r0: float) -> float:
    """Fraction of samples on the segment between the two symbols' boundaries that are purple."""
    ax, ay, ar = a
    bx, by, br = b
    d = math.hypot(bx - ax, by - ay)
    margin = 0.15 * r0
    start = ar + margin
    end = d - br - margin
    if end - start < 3.0:
        return 0.0
    ux, uy = (bx - ax) / d, (by - ay) / d
    t = np.arange(start, end, 1.0)
    hits = np.zeros(t.shape, bool)
    for off in (-1.5, -0.75, 0.0, 0.75, 1.5):
        xs = ax + ux * t - uy * off
        ys = ay + uy * t + ux * off
        hits |= _sample(mask, xs, ys)
    return float(hits.mean())


def _hamiltonian(n: int, adj: dict[int, list[tuple[float, int]]], start: int, finish: int | None, budget: int = 200_000) -> list[int] | None:
    """DFS for a path visiting every node once from start (to finish), best-support edges first."""
    best: list[int] | None = None
    steps = 0

    def dfs(path: list[int], seen: set[int]) -> bool:
        nonlocal best, steps
        steps += 1
        if steps > budget:
            return False
        node = path[-1]
        if len(path) == n:
            if finish is None or node == finish:
                best = list(path)
                return True
            return False
        for _, nxt in adj.get(node, []):
            if nxt in seen:
                continue
            if nxt == finish and len(path) != n - 1:
                continue
            path.append(nxt)
            seen.add(nxt)
            if dfs(path, seen):
                return True
            path.pop()
            seen.discard(nxt)
        return False

    dfs([start], {start})
    return best


def _greedy(adj: dict[int, list[tuple[float, int]]], start: int) -> list[int]:
    path, seen = [start], {start}
    while True:
        nxt = [j for _, j in adj.get(path[-1], []) if j not in seen]
        if not nxt:
            return path
        path.append(nxt[0])
        seen.add(nxt[0])


def order_by_lines(
    mask: np.ndarray, nodes: list[tuple[str, float, float, float]], r0: float, cfg: Settings
) -> tuple[list[int] | None, list[int], dict[tuple[int, int], float]]:
    """Returns (full_order or None, best partial walk, edge supports)."""
    clean = _clean_for_lines(mask, nodes, r0)
    n = len(nodes)
    support: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            support[(i, j)] = line_support(clean, nodes[i][1:], nodes[j][1:], r0)
    kinds = [k for k, *_ in nodes]
    start = kinds.index("start") if "start" in kinds else None
    finish = kinds.index("finish") if "finish" in kinds else None
    partial: list[int] = []
    edges: dict[tuple[int, int], float] = {}
    # leg lines are often cut where they would hide map detail: relax the
    # threshold step by step if no complete course is found
    for thr in (cfg.leg_support_min_ratio, cfg.leg_support_min_ratio * 0.75, cfg.leg_support_min_ratio * 0.5):
        edges = {e: s for e, s in support.items() if s >= thr}
        adj: dict[int, list[tuple[float, int]]] = {}
        for (i, j), s in edges.items():
            adj.setdefault(i, []).append((s, j))
            adj.setdefault(j, []).append((s, i))
        for k in adj:
            adj[k].sort(key=lambda t: -t[0])
        starts = [start] if start is not None else [i for i in range(n) if i != finish and len(adj.get(i, [])) == 1]
        for st in starts:
            full = _hamiltonian(n, adj, st, finish)
            if full is not None:
                return full, full, edges
            walk = _greedy(adj, st)
            if len(walk) > len(partial):
                partial = walk
    return None, partial, edges


# ---------------------------------------------------------------- OCR fallback


def ocr_available() -> bool:
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        return False
    return shutil.which("tesseract") is not None


def _digit_mask(mask: np.ndarray, nodes: list[tuple[str, float, float, float]], r0: float) -> np.ndarray:
    """Purple mask without course symbols and leg lines, leaving mostly text."""
    clean = _clean_for_lines(mask, nodes, r0).astype(np.uint8)
    # wider ring removal than for line support: remnants look like digits
    thick_ring = max(3, int(round(0.4 * r0)))
    for kind, x, y, r in nodes:
        if kind == "control":
            cv2.circle(clean, (int(round(x)), int(round(y))), int(round(r)), 0, thick_ring)
    thick = max(3, int(round(0.2 * r0)))
    support_mask = clean.astype(bool)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a, b = nodes[i], nodes[j]
            if line_support(support_mask, a[1:], b[1:], r0) < 0.3:
                continue
            d = math.hypot(b[1] - a[1], b[2] - a[2])
            ux, uy = (b[1] - a[1]) / d, (b[2] - a[2]) / d
            p1 = (int(round(a[1] + ux * a[3])), int(round(a[2] + uy * a[3])))
            p2 = (int(round(b[1] - ux * b[3])), int(round(b[2] - uy * b[3])))
            cv2.line(clean, p1, p2, 0, thick)
    return clean


def _sequence_support(clean: np.ndarray, nodes: list[tuple[str, float, float, float]], r0: float, seq: list[int]) -> float:
    """Mean leg-line support along a course order."""
    if len(seq) < 2:
        return 0.0
    return float(np.mean([line_support(clean, nodes[a][1:], nodes[b][1:], r0) for a, b in zip(seq, seq[1:])]))


@dataclass
class ControlLabel:
    """Text printed next to a control circle, e.g. "7", "1-126" or "6/10-131"."""

    text: str
    orders: list[int]  # course positions (several when the control is visited more than once)
    code: int | None  # control code after the dash, if printed
    box: tuple[int, int, int, int]  # x0, y0, x1, y1


_LABEL_RE = re.compile(r"^(\d{1,2}(?:/\d{1,2})*)(?:-(\d{1,4}))?$")


def parse_label(text: str) -> tuple[list[int], int | None] | None:
    """``"1-126"`` -> ``([1], 126)``, ``"6/10-131"`` -> ``([6, 10], 131)``, ``"7"`` -> ``([7], None)``."""
    m = _LABEL_RE.match(text.strip())
    if not m:
        return None
    orders = [int(v) for v in m.group(1).split("/")]
    if any(v < 1 for v in orders):
        return None
    return orders, (int(m.group(2)) if m.group(2) else None)


def _text_lines(text: np.ndarray, r0: float) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
    """Group text-sized components into single lines of text.

    Components join when they overlap vertically and the horizontal gap is
    small relative to the text height; small pieces such as "-" and "/" join
    the line they sit in. Returns (mask crop, box) per line.
    """
    n, lab, st, _ = cv2.connectedComponentsWithStats(text.astype(np.uint8), 8)
    comps = [k for k in range(1, n) if st[k, cv2.CC_STAT_AREA] >= 3 and max(st[k, 2], st[k, 3]) <= 2.5 * r0]
    parent = {k: k for k in comps}

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, a in enumerate(comps):
        ax, ay, aw, ah = st[a, :4]
        for b in comps[i + 1 :]:
            bx, by, bw, bh = st[b, :4]
            big = max(ah, bh)
            if big < 0.4 * r0:
                continue
            overlap = min(ay + ah, by + bh) - max(ay, by)
            if overlap < 0.4 * min(ah, bh):
                continue
            gap = max(bx - (ax + aw), ax - (bx + bw))
            if gap <= 0.45 * big:
                parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for k in comps:
        groups.setdefault(find(k), []).append(k)
    out = []
    for g in groups.values():
        if max(st[k, 3] for k in g) < 0.5 * r0:
            continue
        x0 = int(min(st[k, 0] for k in g))
        y0 = int(min(st[k, 1] for k in g))
        x1 = int(max(st[k, 0] + st[k, 2] for k in g))
        y1 = int(max(st[k, 1] + st[k, 3] for k in g))
        out.append((np.isin(lab[y0:y1, x0:x1], g), (x0, y0, x1, y1)))
    return out


def _ocr_line(pytesseract, crop: np.ndarray) -> str:
    img = np.where(crop, 0, 255).astype(np.uint8)
    scale = 48.0 / max(1, img.shape[0])
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    img = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    try:
        txt = pytesseract.image_to_string(img, config="--psm 7 -c tessedit_char_whitelist=0123456789-/")
    except Exception:  # noqa: BLE001 - OCR is best effort
        return ""
    return "".join(txt.split())


def read_labels(
    mask: np.ndarray, nodes: list[tuple[str, float, float, float]], r0: float, cfg: Settings
) -> dict[int, ControlLabel]:
    """Read the label printed next to each control circle. Returns node index -> label.

    Each text line is OCR'd as a whole and parsed as ``N``, ``N-CODE`` or
    ``N/M-CODE``; labels are matched to the nearest circle (each circle gets at
    most one label, each label at most one circle).
    """
    if not ocr_available():
        return {}
    import pytesseract

    text = _digit_mask(mask, nodes, r0)
    candidates: list[tuple[float, int, ControlLabel]] = []
    for crop, box in _text_lines(text, r0):
        raw = _ocr_line(pytesseract, crop)
        parsed = parse_label(raw)
        if parsed is None:
            continue
        label = ControlLabel(raw, parsed[0], parsed[1], box)
        x0, y0, x1, y1 = box
        for idx, (kind, x, y, _r) in enumerate(nodes):
            if kind != "control":
                continue
            dx = max(x0 - x, 0, x - x1)
            dy = max(y0 - y, 0, y - y1)
            dist = math.hypot(dx, dy)
            if dist <= cfg.label_max_distance_ratio * r0:
                candidates.append((dist, idx, label))
    found: dict[int, ControlLabel] = {}
    used: set[int] = set()
    for _dist, idx, label in sorted(candidates, key=lambda t: t[0]):
        if idx in found or id(label) in used:
            continue
        found[idx] = label
        used.add(id(label))
    log.info("labels read: %s", {i: lbl.text for i, lbl in sorted(found.items())})
    return found


def order_from_labels(
    labels: dict[int, ControlLabel], nodes: list[tuple[str, float, float, float]], r0: float, mask: np.ndarray
) -> tuple[list[int] | None, str | None]:
    """Course order (node indices, repeats allowed) from the printed numbers.

    Up to three unread labels are filled in from the leg lines: the missing
    numbers are tried on the unlabelled circles and the assignment with the
    strongest leg lines wins.
    """
    kinds = [k for k, *_ in nodes]
    ctrl = [i for i, k in enumerate(kinds) if k == "control"]
    by_number: dict[int, int] = {}
    for idx, label in labels.items():
        for num in label.orders:
            if num in by_number:
                return None, f"control number {num} was read twice"
            by_number[num] = idx
    if not by_number:
        return None, None
    unlabeled = [i for i in ctrl if i not in labels]
    total = max(max(by_number), len(by_number) + len(unlabeled))
    missing = [v for v in range(1, total + 1) if v not in by_number]
    note = None
    if missing:
        if len(missing) != len(unlabeled) or len(missing) > 3:
            return None, f"control numbers {missing} could not be read"
        clean = _clean_for_lines(mask, nodes, r0)
        start = kinds.index("start") if "start" in kinds else None
        finish = kinds.index("finish") if "finish" in kinds else None

        def sequence(assign: dict[int, int]) -> list[int]:
            seq = [assign.get(v, by_number.get(v)) for v in range(1, total + 1)]
            return ([start] if start is not None else []) + seq + ([finish] if finish is not None else [])

        best: tuple[float, dict[int, int]] | None = None
        for perm in itertools.permutations(unlabeled):
            assign = dict(zip(missing, perm, strict=True))
            seq = sequence(assign)
            score = sum(line_support(clean, nodes[a][1:], nodes[b][1:], r0) for a, b in zip(seq, seq[1:]))
            if best is None or score > best[0]:
                best = (score, assign)
        assert best is not None
        seq = sequence(best[1])
        inferred = set(best[1].values())
        for a, b in zip(seq, seq[1:]):
            if (a in inferred or b in inferred) and line_support(clean, nodes[a][1:], nodes[b][1:], r0) < 0.3:
                return None, f"control numbers {missing} could not be read"
        by_number.update(best[1])
        note = f"control numbers {missing} were not readable and were inferred from the leg lines"
    order = [by_number[v] for v in range(1, total + 1)]
    if "start" in kinds:
        order = [kinds.index("start")] + order
    if "finish" in kinds:
        order.append(kinds.index("finish"))
    return order, note


# ---------------------------------------------------------------- main entry


def detect_controls(mask: np.ndarray, cfg: Settings, r0_hint: float | None = None) -> Detection:
    """Run the full scale-free detection on a purple mask."""
    raw = mask.astype(bool)
    mask = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
    messages: list[str] = []
    r0_est, wide = estimate_r0(mask, cfg)
    r0 = r0_hint or r0_est
    if r0 is None:
        messages.append("Could not determine control circle size; mark one control circle manually.")
        return Detection(None, None, messages=messages, candidates=wide)
    ppm = 2 * r0 / cfg.control_diameter_mm

    # refine with a narrow radius window
    lo = max(1, int(math.floor((1 - cfg.refine_radius_ratio) * r0)))
    hi = int(math.ceil((1 + cfg.refine_radius_ratio) * r0))
    refined = _hough(mask, lo, hi, cfg, min_dist=max(2.0, 0.25 * r0))
    refined += [c for c in wide if lo <= c.r <= hi]
    circles = _verify(mask, refined, cfg)
    circles = [c for c in _nms(circles, dist=r0) if lo - 1 <= c.r <= hi + 1]
    if circles:
        # drop weak circles relative to the typical control circle in this map
        ref = float(np.median([c.support for c in circles]))
        circles = [c for c in circles if c.support >= cfg.relative_support_min_ratio * ref]

    # finish: concentric pair; test every centre we have seen
    all_hough = _hough(mask, max(1, int(0.6 * r0)), int(1.5 * r0) + 1, cfg, min_dist=2.0)
    # the gap between the finish rings is narrow: use the unclosed mask
    finish = find_finish(raw, circles + wide + all_hough, r0, cfg)
    if finish is not None:
        circles = [c for c in circles if math.hypot(c.x - finish.x, c.y - finish.y) > 0.5 * r0]

    start = find_start(mask, r0, cfg, exclude=[(c.x, c.y, c.r) for c in circles])
    if start is not None:
        # circles fitted to the triangle outline are not controls
        circles = [c for c in circles if math.hypot(c.x - start[0], c.y - start[1]) > start[2] + 0.5 * c.r]

    nodes: list[tuple[str, float, float, float]] = []
    if start:
        nodes.append(("start", *start))
    nodes += [("control", c.x, c.y, c.r) for c in circles]
    if finish:
        nodes.append(("finish", finish.x, finish.y, finish.r))

    det = Detection(r0, ppm, controls=circles, start=start, finish=finish, nodes=nodes, candidates=wide)
    if not start:
        messages.append("Start triangle not found.")
    if not finish:
        messages.append("Finish (double circle) not found.")

    if len(nodes) >= 2:
        # unclosed mask: closing fills the small holes of digits ("0" -> "8")
        labels = read_labels(raw, nodes, r0, cfg)
        det.codes = {i: lbl.code for i, lbl in labels.items() if lbl.code is not None}
        label_order, label_note = order_from_labels(labels, nodes, r0, mask) if labels else (None, None)
        full, partial, edges = order_by_lines(mask, nodes, r0, cfg)
        log.info("leg graph: %d nodes, %d edges, full order=%s", len(nodes), len(edges), full is not None)
        log.info("order from printed numbers: %s (%s)", label_order, label_note)
        if label_order is not None and label_order == full:
            label_note = None  # leg lines confirm the inferred numbers
        if label_order is not None and full is not None and label_order != full:
            # both complete but different: trust the one the leg lines support better
            clean = _clean_for_lines(mask, nodes, r0)
            if _sequence_support(clean, nodes, r0, full) > _sequence_support(clean, nodes, r0, label_order):
                label_order = None
                messages.append("Printed control numbers disagree with the leg lines; used the leg lines.")
            else:
                messages.append("Leg lines disagree with the printed control numbers; used the numbers.")
        if label_order is not None:
            det.order, det.ordered = label_order, True
            if label_note:
                messages.append(label_note[0].upper() + label_note[1:] + "; please verify.")
        elif full is not None:
            det.order, det.ordered = full, True
        else:
            det.order = partial
            if label_note:
                messages.append(f"Printed numbers: {label_note}.")
            messages.append(
                f"Leg lines connect only {len(partial)} of {len(nodes)} course points; please order the rest manually."
            )
    det.messages = messages
    log.info("controls: r0=%.2f, %d circles, start=%s, finish=%s", r0, len(circles), bool(start), bool(finish))
    return det
