import math

import cv2
import numpy as np
import pytest

from omap_router.config import TerrainClass
from omap_router.models import EditPolygon
from omap_router.pipeline import lines, routing, terrain


def _grid(labels, cfg, barrier=None, edits=(), ppm=6.0, cell_px=1):
    barrier = np.zeros(labels.shape, bool) if barrier is None else barrier
    forb = terrain.auto_forbidden(labels, barrier, ppm, cfg)
    forb, bar = terrain.apply_edits(forb, barrier, list(edits))
    return terrain.cost_grid(terrain.fine_cost(labels, cfg), forb, bar, cell_px)


def _route(grid, a, b):
    return routing.route_leg(grid, routing.Snapper(grid), a, b)


def _assert_path_passable(grid, res):
    """Invariant: every cell along the (simplified) path is passable."""
    assert res.ok
    assert routing.path_clear(grid, res.path_xy)


# ---------------------------------------------------------------- lines


def test_thick_line_is_barrier_thin_is_not(cfg):
    lab = np.full((200, 300), int(TerrainClass.WHITE), np.uint8)
    thick = np.zeros(lab.shape, np.uint8)
    cv2.line(thick, (20, 50), (280, 50), 1, 4)
    thin = np.zeros(lab.shape, np.uint8)
    cv2.line(thin, (20, 150), (280, 150), 1, 1)
    lab[thick > 0] = TerrainClass.BLACK
    lab[thin > 0] = TerrainClass.BLACK
    bar = lines.barrier_mask(lab, px_per_mm=6.0, cfg=cfg)
    assert bar[48:53, 30:270].any(axis=0).mean() > 0.95
    assert not bar[145:156].any()


def test_small_text_blob_removed(cfg):
    lab = np.full((100, 100), int(TerrainClass.WHITE), np.uint8)
    lab[40:46, 40:45] = TerrainClass.BLACK  # compact symbol, < 2 mm
    assert not lines.barrier_mask(lab, 6.0, cfg).any()


def test_building_outline_not_barrier(cfg):
    lab = np.full((120, 120), int(TerrainClass.WHITE), np.uint8)
    lab[28:92, 28:92] = TerrainClass.BLACK  # 2 px black outline
    lab[30:90, 30:90] = TerrainClass.GREY
    assert not lines.barrier_mask(lab, 6.0, cfg).any()


# ---------------------------------------------------------------- terrain


def test_thin_blue_lines_are_passable(cfg):
    lab = np.full((50, 50), int(TerrainClass.WHITE), np.uint8)
    lab[:, 25] = TerrainClass.BLUE
    lab[30:, :] = TerrainClass.BLUE
    water = terrain.water_mask(lab, 6.0, cfg)
    assert not water[:20, 25].any()
    assert water[35:45, 5:45].all()


def test_narrow_passage_stays_open_with_coarse_cells(cfg):
    lab = np.full((60, 60), int(TerrainClass.WHITE), np.uint8)
    lab[:, 20:40] = TerrainClass.GREY
    lab[28:31, 20:40] = TerrainClass.WHITE  # 3 px alley through the building
    g = _grid(lab, cfg, cell_px=4)
    res = _route(g, (5, 29), (55, 29))
    _assert_path_passable(g, res)


def test_thin_wall_stays_closed_with_coarse_cells(cfg):
    lab = np.full((60, 60), int(TerrainClass.WHITE), np.uint8)
    bar = np.zeros(lab.shape, bool)
    bar[:, 30] = True
    g = _grid(lab, cfg, barrier=bar, cell_px=4)
    res = _route(g, (5, 29), (55, 29))
    assert not res.ok
    assert res.message


def test_diagonal_wall_does_not_leak(cfg):
    lab = np.full((40, 40), int(TerrainClass.WHITE), np.uint8)
    bar = np.eye(40, dtype=bool)  # one pixel diagonal wall, 8-connected only
    g = _grid(lab, cfg, barrier=bar)
    res = _route(g, (30, 5), (5, 30))
    assert not res.ok


def test_forbid_and_allow_polygons(cfg):
    lab = np.full((100, 100), int(TerrainClass.WHITE), np.uint8)
    lab[:, 45:55] = TerrainClass.GREY  # building wall across the map
    blocked = _grid(lab, cfg)
    assert not _route(blocked, (10, 50), (90, 50)).ok
    door = EditPolygon(id="a", kind="allow", points=[(40, 45), (60, 45), (60, 55), (40, 55)])
    opened = _grid(lab, cfg, edits=[door])
    res = _route(opened, (10, 50), (90, 50))
    _assert_path_passable(opened, res)
    # forbid polygon across open terrain forces a detour
    lab2 = np.full((100, 100), int(TerrainClass.WHITE), np.uint8)
    wall = EditPolygon(id="b", kind="forbid", points=[(48, 0), (52, 0), (52, 80), (48, 80)])
    g = _grid(lab2, cfg, edits=[wall])
    res = _route(g, (10, 20), (90, 20))
    _assert_path_passable(g, res)
    assert max(y for _, y in res.path_xy) >= 80


# ---------------------------------------------------------------- routing


def test_prefers_road_when_moderately_longer(cfg):
    # road cost 1.0 vs open land 1.2 => a road up to 20 % longer wins. The
    # 8-connected grid overestimates oblique lengths by up to ~8 %, so test a
    # 10 % detour at ~29 degrees.
    h, w = 200, 400
    lab = np.full((h, w), int(TerrainClass.WHITE), np.uint8)
    a, b = (20.0, 150.0), (380.0, 150.0)
    direct = math.dist(a, b)
    # road: a -> apex -> b, 10 % longer than the direct line
    half = direct / 2 * 1.10
    apex_y = 150 - math.sqrt(half**2 - (direct / 2) ** 2)
    road = np.zeros((h, w), np.uint8)
    pts = np.array([a, (200, apex_y), b], np.float64)
    cv2.polylines(road, [np.rint(pts).astype(np.int32)], False, 1, 9)
    lab[road > 0] = TerrainClass.BEIGE
    g = _grid(lab, cfg)
    res = _route(g, a, b)
    _assert_path_passable(g, res)
    # sample the path densely and check it stays on the road
    on = []
    for (x0, y0), (x1, y1) in zip(res.path_xy, res.path_xy[1:]):
        for t in np.linspace(0, 1, 20):
            x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
            on.append(road[int(y), int(x)] > 0)
    assert np.mean(on) > 0.9


def test_cuts_across_when_road_much_longer(cfg):
    h, w = 300, 400
    lab = np.full((h, w), int(TerrainClass.WHITE), np.uint8)
    a, b = (20.0, 250.0), (380.0, 250.0)
    road = np.zeros((h, w), np.uint8)
    cv2.polylines(road, [np.array([a, (200, 10), b], np.int32)], False, 1, 9)
    lab[road > 0] = TerrainClass.BEIGE
    g = _grid(lab, cfg)
    res = _route(g, a, b)
    assert min(y for _, y in res.path_xy) > 150


def test_snap_skips_tiny_pockets(cfg):
    lab = np.full((80, 80), int(TerrainClass.WHITE), np.uint8)
    ring = np.zeros(lab.shape, np.uint8)
    cv2.circle(ring, (40, 40), 6, 1, 2)
    lab[ring > 0] = TerrainClass.GREY  # control centre enclosed by a small symbol
    g = _grid(lab, cfg)
    s = routing.Snapper(g, min_component_cells=200)
    r, c = s.snap(40, 40)
    assert math.hypot(r - 40, c - 40) > 6
    res = routing.route_leg(g, s, (40, 40), (5, 5))
    _assert_path_passable(g, res)


@pytest.mark.parametrize("seed", range(3))
def test_random_maps_paths_never_cross_forbidden(seed, cfg):
    rng = np.random.default_rng(seed)
    lab = np.full((150, 150), int(TerrainClass.WHITE), np.uint8)
    for _ in range(25):
        x, y = rng.integers(0, 140, 2)
        cls = rng.choice([TerrainClass.GREY, TerrainClass.OLIVE, TerrainClass.GREEN, TerrainClass.BEIGE])
        lab[y : y + rng.integers(3, 20), x : x + rng.integers(3, 20)] = cls
    g = _grid(lab, cfg, cell_px=int(rng.integers(1, 4)))
    snap = routing.Snapper(g)
    for _ in range(5):
        a = tuple(rng.uniform(0, 150, 2))
        b = tuple(rng.uniform(0, 150, 2))
        res = routing.route_leg(g, snap, a, b)
        if res.ok:
            _assert_path_passable(g, res)
