import cv2
import numpy as np
import pytest
from synthetic import iou

from omap_router.config import TerrainClass
from omap_router.pipeline import overprint, preprocess, segment


def _labels(img_bgr, cfg, samples=None):
    rgb = preprocess.preprocess(img_bgr, None, cfg)
    return segment.segment(rgb, segment.reference_colors(rgb, cfg, samples), cfg)


def test_building_and_olive_iou(synth, cfg):
    labels = _labels(synth.image, cfg)
    purple = overprint.purple_mask(labels)
    terrain = overprint.remove_purple(labels, purple, cfg.purple_dilate_px)
    # ground truth: buildings are grey rectangles; olive is the top-right block
    gt_olive = np.zeros(labels.shape, bool)
    gt_olive[40:231, 600:861] = True
    assert iou(terrain == TerrainClass.OLIVE, gt_olive) >= 0.85
    gt_build = synth.forbidden & ~gt_olive
    gt_build[560:, :] = False  # lake
    gt_build[335:406, 250:401] = False  # purple hatching
    walls = np.zeros_like(gt_build)
    cv2.line(walls.view(np.uint8), (500, 120), (580, 250), 1, 6)
    gt_build &= ~walls
    build = (terrain == TerrainClass.GREY) | (terrain == TerrainClass.BLACK)
    build &= ~walls
    assert iou(build & gt_build | (terrain == TerrainClass.GREY), gt_build) >= 0.85


def test_purple_detected(synth, cfg):
    labels = _labels(synth.image, cfg)
    purple = labels == TerrainClass.PURPLE
    # every control centre is surrounded by a purple ring
    for kind, x, y in synth.course:
        if kind != "control":
            continue
        ring = purple[int(y) - 22 : int(y) + 23, int(x) - 22 : int(x) + 23]
        assert ring.sum() > 50


def test_user_samples_override(cfg):
    # an odd shade that is far from every default reference colour
    img = np.zeros((40, 80, 3), np.uint8)
    img[:, :40] = (90, 200, 250)  # BGR: unusual yellow
    img[:, 40:] = (255, 255, 255)
    rgb = preprocess.preprocess(img, None, cfg)
    refs = segment.reference_colors(rgb, cfg, {"YELLOW": [(10, 20)]})
    labels = segment.segment(rgb, refs, cfg)
    assert (labels[:, 5:35] == TerrainClass.YELLOW).mean() > 0.95
    assert (labels[:, 45:75] == TerrainClass.WHITE).mean() > 0.95


def test_unknown_majority_fill():
    lab = np.full((5, 5), int(TerrainClass.BEIGE), np.uint8)
    lab[2, 2] = TerrainClass.UNKNOWN
    out = segment.majority_fill(lab, 3)
    assert out[2, 2] == TerrainClass.BEIGE


def test_remove_purple_takes_neighbour_label():
    lab = np.full((9, 9), int(TerrainClass.GREY), np.uint8)
    lab[:, 4] = TerrainClass.PURPLE
    out = overprint.remove_purple(lab, lab == TerrainClass.PURPLE)
    assert (out == TerrainClass.GREY).all()


# ---------------------------------------------------------------- purple hatching


def _course_only(r0=18.0):
    from synthetic import draw_course

    img = np.full((500, 700, 3), 255, np.uint8)
    course = [("start", 100.0, 100.0), ("control", 300.0, 120.0), ("control", 500.0, 300.0), ("finish", 200.0, 400.0)]
    draw_course(img, course, r0)
    cv2.putText(img, "1234567890", (80, 470), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (185, 70, 165), 3)
    return img


@pytest.mark.parametrize("spacing_mm", [1.0, 1.6, 2.4])
def test_hatch_detected(cfg, spacing_mm):
    from synthetic import draw_hatch

    img = _course_only()
    ppm = 6.0
    draw_hatch(img, (380, 60, 560, 200), ppm, spacing_mm)
    purple = np.abs(img.astype(int) - (185, 70, 165)).sum(axis=2) < 120
    hatch = overprint.hatch_mask(purple, ppm, cfg)
    gt = np.zeros(hatch.shape, bool)
    gt[60:201, 380:561] = True
    assert iou(hatch, gt) >= 0.8
    # course symbols, leg lines and digits are not hatching
    assert not hatch[:, :360].any()
    assert not hatch[230:, :].any()


def test_no_hatch_on_plain_course(cfg):
    img = _course_only()
    purple = np.abs(img.astype(int) - (185, 70, 165)).sum(axis=2) < 120
    assert not overprint.hatch_mask(purple, 6.0, cfg).any()
