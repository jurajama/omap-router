"""Orchestrates the pipeline stages and caches their outputs per project.

User edits (controls, polygons, colour samples) live in ``project.json`` and are
applied on top of the cached automatic results, so re-running an automatic
stage never loses them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from contextlib import contextmanager

import cv2
import numpy as np

from .. import storage
from ..config import Settings, get_settings
from ..models import Calibration, Control, Leg, Project
from . import controls as controls_mod
from . import lines, overprint, preprocess, render, routing, segment, terrain

log = logging.getLogger(__name__)


@contextmanager
def _timed(stage: str):
    t0 = time.perf_counter()
    yield
    log.info("stage %s: %.2fs", stage, time.perf_counter() - t0)


def _key(*parts: object) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def _image_sha(project: Project) -> str:
    return hashlib.sha256(storage.image_path(project).read_bytes()).hexdigest()


def _offset(project: Project) -> tuple[int, int]:
    return (project.crop[0], project.crop[1]) if project.crop else (0, 0)


def _cfg_dump(cfg: Settings) -> dict:
    d = cfg.model_dump()
    d.pop("data_dir", None)
    return d


def px_per_mm_from_r0(r0: float, cfg: Settings) -> float:
    return 2.0 * r0 / cfg.control_diameter_mm


def m_per_px(cal: Calibration, cfg: Settings) -> float:
    ppm = cal.px_per_mm or cfg.default_px_per_mm
    return cal.scale / 1000.0 / ppm


# ------------------------------------------------------------------ stages


def _segmentation(project: Project, cfg: Settings) -> tuple[np.ndarray, str]:
    """Stages 8.1 + 8.2 (cached). Returns (labels, cache key)."""
    key = _key("seg", _image_sha(project), project.crop, project.color_samples, _cfg_dump(cfg))
    cached = storage.cache_load(project.id, "segment", key)
    if cached is not None:
        return cached["labels"], key
    img = storage.load_image_bgr(project)
    with _timed("preprocess"):
        rgb = preprocess.preprocess(img, project.crop, cfg)
    with _timed("segment"):
        refs = segment.reference_colors(rgb, cfg, project.color_samples, _offset(project))
        labels = segment.segment(rgb, refs, cfg)
    storage.cache_save(project.id, "segment", key, {"labels": labels})
    return labels, key


def _purple_for_detection(labels: np.ndarray) -> np.ndarray:
    return overprint.purple_mask(labels)


def _ensure_scale(project: Project, labels: np.ndarray, cfg: Settings) -> None:
    """Fill calibration.r0 / px_per_mm from the circle consensus if unknown."""
    cal = project.calibration
    if cal.r0 is not None and cal.r0_source in ("auto", "user"):
        cal.px_per_mm = px_per_mm_from_r0(cal.r0, cfg)
        return
    with _timed("r0"):
        mask = cv2.morphologyEx(
            _purple_for_detection(labels).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
        ).astype(bool)
        r0, verified = controls_mod.estimate_r0(mask, cfg)
    log.info("r0 estimate: %s from %d verified circles", r0, len(verified))
    if r0 is None:
        cal.r0 = None
        cal.px_per_mm = cfg.default_px_per_mm
        cal.r0_source = "default"
        project.messages.append(
            "Control circle size not found; using a default resolution. Mark one control circle to calibrate."
        )
    else:
        cal.r0 = r0
        cal.px_per_mm = px_per_mm_from_r0(r0, cfg)
        cal.r0_source = "auto"


def _terrain(project: Project, cfg: Settings) -> dict[str, np.ndarray]:
    """Stages 8.3 + 8.5 + automatic part of 8.6 (cached)."""
    labels, seg_key = _segmentation(project, cfg)
    _ensure_scale(project, labels, cfg)
    ppm = project.calibration.px_per_mm or cfg.default_px_per_mm
    key = _key("terrain", seg_key, round(ppm, 4))
    cached = storage.cache_load(project.id, "terrain", key)
    if cached is not None:
        return cached
    with _timed("overprint"):
        purple = overprint.purple_mask(labels)
        tlabels = overprint.remove_purple(labels, purple, cfg.purple_dilate_px)
        hatch = overprint.hatch_mask(purple, ppm, cfg) if cfg.hatch_enabled else np.zeros(purple.shape, bool)
    with _timed("lines"):
        barrier = lines.barrier_mask(tlabels, ppm, cfg)
    with _timed("forbidden"):
        if cfg.outside_map_enabled:
            outside = terrain.outside_map_mask(tlabels, ppm, cfg)
        else:
            outside = np.zeros(tlabels.shape, bool)
        forb = terrain.auto_forbidden(tlabels, barrier, ppm, cfg, hatch, outside)
    log.info("hatched out-of-bounds: %d px, outside map: %d px", int(hatch.sum()), int(outside.sum()))
    out = {
        "labels": labels,
        "terrain_labels": tlabels,
        "purple": purple,
        "hatch": hatch,
        "outside": outside,
        "barrier": barrier,
        "forbidden": forb,
    }
    storage.cache_save(project.id, "terrain", key, out)
    return out


def _grid(project: Project, t: dict[str, np.ndarray], cfg: Settings) -> tuple[terrain.CostGrid, np.ndarray]:
    """User edits applied on top of the automatic masks. Returns (grid, fine forbidden)."""
    ppm = project.calibration.px_per_mm or cfg.default_px_per_mm
    forb, bar = terrain.apply_edits(t["forbidden"], t["barrier"], project.edits, _offset(project))
    cost = terrain.fine_cost(t["terrain_labels"], cfg)
    grid = terrain.cost_grid(cost, forb, bar, terrain.cell_size_px(ppm, cfg))
    return grid, forb


def _full_shape(project: Project) -> tuple[int, int]:
    return project.height, project.width


def _render_layers(project: Project, t: dict[str, np.ndarray], grid: terrain.CostGrid, forb: np.ndarray) -> list[str]:
    d = storage.layers_dir(project.id)
    full, off = _full_shape(project), _offset(project)
    with _timed("render"):
        render.write_png(d / "labels.png", render.labels_rgba(t["labels"]), full, off)
        render.write_png(d / "forbidden.png", render.mask_rgba(forb, (230, 0, 0), 170), full, off)
        render.write_png(d / "purple.png", render.mask_rgba(t["purple"], (200, 0, 200), 255), full, off)
        render.write_png(d / "hatch.png", render.mask_rgba(t["hatch"], (255, 120, 0), 200), full, off)
        render.write_png(d / "outside.png", render.mask_rgba(t["outside"], (90, 90, 90), 170), full, off)
        render.write_png(d / "barriers.png", render.mask_rgba(t["barrier"], (0, 60, 255), 255), full, off)
        render.write_png(d / "cost.png", render.cost_rgba(grid.cost, grid.cell_px, forb.shape), full, off)
    return list(render.LAYER_NAMES)


# ------------------------------------------------------------------ public API


def analyze(project: Project, cfg: Settings | None = None) -> tuple[Project, list[str]]:
    """Run 8.1–8.6 and write the layer PNGs."""
    cfg = cfg or get_settings()
    project.messages = []
    t = _terrain(project, cfg)
    grid, forb = _grid(project, t, cfg)
    layers = _render_layers(project, t, grid, forb)
    storage.save_project(project)
    return project, layers


def detect_controls(project: Project, cfg: Settings | None = None) -> Project:
    """Run 8.4 and replace the control list."""
    cfg = cfg or get_settings()
    project.messages = []
    labels, _ = _segmentation(project, cfg)
    hint = project.calibration.r0 if project.calibration.r0_source == "user" else None
    with _timed("controls"):
        det = controls_mod.detect_controls(_purple_for_detection(labels), cfg, r0_hint=hint)
    project.messages.extend(det.messages)
    cal = project.calibration
    if det.r0 is not None:
        if cal.r0_source != "user":
            cal.r0 = det.r0
            cal.r0_source = "auto"
        cal.px_per_mm = px_per_mm_from_r0(cal.r0, cfg)
    ox, oy = _offset(project)
    order_pos = {node: i for i, node in enumerate(det.order or [])}
    ctrls: list[Control] = []
    for idx, (kind, x, y, r) in enumerate(det.nodes):
        ctrls.append(
            Control(
                id=idx + 1,
                order=order_pos.get(idx),
                kind=kind,  # type: ignore[arg-type]
                x=float(x) + ox,
                y=float(y) + oy,
                r=float(r),
                source="auto",
            )
        )
    project.controls = ctrls
    project.legs = []
    log.info("detected %d course points (r0=%s)", len(ctrls), cal.r0)
    storage.save_project(project)
    return project


def ordered_controls(project: Project) -> list[Control]:
    return sorted((c for c in project.controls if c.order is not None), key=lambda c: c.order)  # type: ignore[arg-type, return-value]


def route(project: Project, cfg: Settings | None = None) -> Project:
    """Compute least-cost legs between consecutive ordered controls."""
    cfg = cfg or get_settings()
    t = _terrain(project, cfg)
    grid, _ = _grid(project, t, cfg)
    ppm = project.calibration.px_per_mm or cfg.default_px_per_mm
    min_cells = int(cfg.snap_min_component_mm2 * (ppm / grid.cell_px) ** 2)
    snapper = routing.Snapper(grid, min_cells)
    ox, oy = _offset(project)
    mpp = m_per_px(project.calibration, cfg)
    seq = ordered_controls(project)
    legs: list[Leg] = []
    with _timed("routing"):
        for a, b in zip(seq, seq[1:]):
            res = routing.route_leg(grid, snapper, (a.x - ox, a.y - oy), (b.x - ox, b.y - oy))
            path = [(x + ox, y + oy) for x, y in res.path_xy]
            length = routing.path_length(path)
            legs.append(
                Leg(
                    from_id=a.id,
                    to_id=b.id,
                    path=path,
                    length_px=length,
                    length_m=length * mpp,
                    cost=res.cost if math.isfinite(res.cost) else -1.0,
                    ok=res.ok,
                    message=res.message,
                )
            )
    log.info("routed %d legs (%d unreachable)", len(legs), sum(not leg.ok for leg in legs))
    project.legs = legs
    storage.save_project(project)
    return project


def update_leg_lengths(project: Project, cfg: Settings | None = None) -> None:
    cfg = cfg or get_settings()
    mpp = m_per_px(project.calibration, cfg)
    for leg in project.legs:
        leg.length_m = leg.length_px * mpp
