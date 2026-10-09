import math

import cv2
import numpy as np
import pytest
from synthetic import draw_course

from omap_router.pipeline import controls, overprint, preprocess, segment


def _mask_from(img_bgr: np.ndarray) -> np.ndarray:
    # purple drawn on white: any clearly coloured pixel is overprint
    return (img_bgr.astype(int).sum(axis=2) < 600)


def _random_course(rng, r0, w, h, n_controls=6):
    pts = []
    margin = int(3 * r0)
    while len(pts) < n_controls + 2:
        x, y = rng.uniform(margin, w - margin), rng.uniform(margin, h - margin)
        if all(math.hypot(x - a, y - b) > 5 * r0 for a, b in pts):
            pts.append((x, y))
    kinds = ["start"] + ["control"] * n_controls + ["finish"]
    return [(k, x, y) for k, (x, y) in zip(kinds, pts)]


@pytest.mark.parametrize("seed", range(6))
def test_r0_found_without_size_hint(seed, cfg):
    rng = np.random.default_rng(seed)
    r0 = float(rng.uniform(8, 60))
    w = h = int(max(600, 16 * r0 * 2.2))
    img = np.full((h, w, 3), 255, np.uint8)
    course = _random_course(rng, r0, w, h, n_controls=5)
    draw_course(img, course, r0, numbers=False)
    est, _ = controls.estimate_r0(_mask_from(img), cfg)
    assert est is not None
    assert abs(est - r0) / r0 < 0.1


def test_full_detection_and_order(synth, cfg):
    rgb = preprocess.preprocess(synth.image, None, cfg)
    labels = segment.segment(rgb, segment.reference_colors(rgb, cfg), cfg)
    mask = overprint.purple_mask(labels)
    det = controls.detect_controls(mask, cfg)
    assert det.r0 == pytest.approx(synth.r0, rel=0.1)
    assert det.ordered, det.messages
    got = [det.nodes[i] for i in det.order]
    assert [k for k, *_ in got] == [k for k, *_ in synth.course]
    for (k, x, y, _r), (_, ex, ey) in zip(got, synth.course):
        assert math.hypot(x - ex, y - ey) < 0.25 * synth.r0, (k, x, y, ex, ey)


def test_no_circles_asks_user(cfg):
    img = np.full((300, 300, 3), 255, np.uint8)
    cv2.line(img, (10, 10), (290, 290), (185, 70, 165), 2)
    det = controls.detect_controls(_mask_from(img), cfg)
    assert det.r0 is None
    assert det.messages


def test_filled_discs_rejected(cfg):
    img = np.full((400, 400, 3), 255, np.uint8)
    for x, y in [(80, 80), (200, 80), (320, 80), (80, 250)]:
        cv2.circle(img, (x, y), 20, (185, 70, 165), -1)
    r0, verified = controls.estimate_r0(_mask_from(img), cfg)
    assert r0 is None
    assert not verified


def test_line_support():
    m = np.zeros((100, 200), bool)
    m[50, 20:180] = True
    assert controls.line_support(m, (10, 50, 5), (190, 50, 5), 10) > 0.9
    assert controls.line_support(m, (10, 20, 5), (190, 20, 5), 10) == 0.0
