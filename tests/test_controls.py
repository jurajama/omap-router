import math

import cv2
import numpy as np
import pytest
from synthetic import draw_course

from omap_router.config import Settings
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


# ---------------------------------------------------------------- printed labels "N-CODE"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("7", ([7], None)),
        ("1-126", ([1], 126)),
        ("6/10-131", ([6, 10], 131)),
        ("11-133", ([11], 133)),
        ("-12", None),
        ("1--2", None),
        ("0-100", None),
        ("", None),
    ],
)
def test_parse_label(text, expected):
    assert controls.parse_label(text) == expected


def _labelled_course(r0: float = 18.0):
    """S, 1-101, 2/4-102, 3-103, 5-104, F: control 102 is visited twice."""
    img = np.full((720, 960, 3), 255, np.uint8)
    pts = {"S": (120.0, 600.0), "A": (160.0, 220.0), "B": (480.0, 330.0), "C": (800.0, 160.0), "D": (760.0, 560.0), "F": (420.0, 620.0)}
    seq = ["S", "A", "B", "C", "B", "D", "F"]
    kinds = {"S": "start", "F": "finish"}
    course = [(kinds.get(k, "control"), *pts[k]) for k in seq]
    draw_course(img, course, r0, numbers=False)
    labels = {"A": "1-101", "B": "2/4-102", "C": "3-103", "D": "5-104"}
    for k, text in labels.items():
        x, y = pts[k]
        cv2.putText(img, text, (int(x + 1.3 * r0), int(y - 0.9 * r0)), cv2.FONT_HERSHEY_SIMPLEX, r0 / 20, (185, 70, 165), 2, cv2.LINE_AA)
    return img, pts, seq


def _purple(img):
    rgb = preprocess.preprocess(img, None, Settings())
    return overprint.purple_mask(segment.segment(rgb, segment.reference_colors(rgb, Settings()), Settings()))


@pytest.mark.skipif(not controls.ocr_available(), reason="tesseract not installed")
def test_labels_with_codes_and_revisit(cfg):
    img, pts, seq = _labelled_course()
    det = controls.detect_controls(_purple(img), cfg)
    assert det.ordered, det.messages
    got = [det.nodes[i] for i in det.order]
    assert len(got) == len(seq)
    for (_k, x, y, _r), name in zip(got, seq, strict=True):
        assert math.hypot(x - pts[name][0], y - pts[name][1]) < 6, name
    # the revisited circle is the same node at positions 2 and 4
    assert det.order[2] == det.order[4]
    codes = {name: det.codes.get(det.order[i]) for i, name in enumerate(seq) if name in "ABCD"}
    assert codes == {"A": 101, "B": 102, "C": 103, "D": 104}


def test_order_from_labels_fills_unread_number_from_leg_lines(cfg):
    img, pts, seq = _labelled_course()
    mask = _purple(img)
    nodes = [("start", *pts["S"], 22.0)] + [("control", *pts[k], 18.0) for k in "ABCD"] + [("finish", *pts["F"], 21.0)]
    lbl = controls.ControlLabel
    labels = {1: lbl("1-101", [1], 101, (0, 0, 1, 1)), 2: lbl("2/4-102", [2, 4], 102, (0, 0, 1, 1)), 4: lbl("5-104", [5], 104, (0, 0, 1, 1))}
    order, note = controls.order_from_labels(labels, nodes, 18.0, mask)  # "3-103" unread
    assert order == [0, 1, 2, 3, 2, 4, 5]
    assert note and "3" in note
    labels[3] = lbl("1-103", [1], 103, (0, 0, 1, 1))  # misread: number 1 twice
    assert controls.order_from_labels(labels, nodes, 18.0, mask)[0] is None
