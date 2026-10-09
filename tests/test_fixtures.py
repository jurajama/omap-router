"""Fixture tests: synthetic map always, plus real map crops under ``tests/fixtures/<name>/``.

A real fixture directory contains ``image.png`` and optionally
``forbidden.png`` (white = forbidden), ``passages.png`` (white = narrow
passages that must stay open) and ``controls.json``
(``{"r0": float, "course": [{"kind": "start", "x": .., "y": ..}, ...]}``).
"""

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
from synthetic import iou

from omap_router.config import Settings
from omap_router.pipeline import controls, lines, overprint, preprocess, segment, terrain

FIXTURES = Path(__file__).parent / "fixtures"
REAL = sorted(d for d in FIXTURES.glob("*") if (d / "image.png").exists()) if FIXTURES.exists() else []


def _analyse(img_bgr, cfg: Settings):
    rgb = preprocess.preprocess(img_bgr, None, cfg)
    labels = segment.segment(rgb, segment.reference_colors(rgb, cfg), cfg)
    purple = overprint.purple_mask(labels)
    det = controls.detect_controls(purple, cfg)
    ppm = det.px_per_mm or cfg.default_px_per_mm
    tl = overprint.remove_purple(labels, purple, cfg.purple_dilate_px)
    bar = lines.barrier_mask(tl, ppm, cfg)
    forb = terrain.auto_forbidden(
        tl, bar, ppm, cfg, overprint.hatch_mask(purple, ppm, cfg), terrain.outside_map_mask(tl, ppm, cfg)
    )
    return det, forb


def _check_controls(det, r0: float, course: list[tuple[str, float, float]]):
    assert det.r0 is not None and abs(det.r0 - r0) / r0 <= 0.1
    found = 0
    for kind, x, y in course:
        if any(k == kind and math.hypot(nx - x, ny - y) < 0.5 * r0 for k, nx, ny, _ in det.nodes):
            found += 1
    assert found / len(course) >= 0.9
    false_pos = sum(
        1 for k, nx, ny, _ in det.nodes if not any(math.hypot(nx - x, ny - y) < 0.5 * r0 for _, x, y in course)
    )
    assert false_pos <= 1


def test_synthetic_fixture(synth, cfg):
    det, forb = _analyse(synth.image, cfg)
    _check_controls(det, synth.r0, synth.course)
    assert iou(forb, synth.forbidden) >= 0.85
    # the 6 px passage between the two top-left buildings stays open
    assert not forb[110, 201:206].all()


@pytest.mark.parametrize("fixture", REAL, ids=[d.name for d in REAL])
def test_real_fixture(fixture, cfg):
    img = cv2.imread(str(fixture / "image.png"))
    det, forb = _analyse(img, cfg)
    if (fixture / "controls.json").exists():
        spec = json.loads((fixture / "controls.json").read_text())
        course = [(c["kind"], c["x"], c["y"]) for c in spec["course"]]
        _check_controls(det, spec["r0"], course)
        if det.ordered:
            got = [det.nodes[i][0] for i in det.order]
            assert got == [k for k, *_ in course]
    if (fixture / "forbidden.png").exists():
        gt = cv2.imread(str(fixture / "forbidden.png"), cv2.IMREAD_GRAYSCALE) > 127
        assert iou(forb, gt) >= 0.85
    if (fixture / "passages.png").exists():
        pas = cv2.imread(str(fixture / "passages.png"), cv2.IMREAD_GRAYSCALE) > 127
        n, lab = cv2.connectedComponents(pas.astype(np.uint8))
        for i in range(1, n):
            assert not forb[lab == i].all(), f"passage {i} closed"
