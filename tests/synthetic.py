"""Synthetic ISSprOM-like map generator for tests (no real map data in the repo)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

# RGB
WHITE = (255, 255, 255)
BEIGE = (235, 204, 172)
GREY = (145, 141, 138)
OLIVE = (170, 158, 55)
BLUE = (0, 175, 235)
GREEN = (145, 210, 120)
YELLOW = (245, 188, 85)
BLACK = (30, 25, 15)
PURPLE = (165, 70, 185)


def bgr(c: tuple[int, int, int]) -> tuple[int, int, int]:
    return (c[2], c[1], c[0])


@dataclass
class SyntheticMap:
    image: np.ndarray  # BGR uint8
    forbidden: np.ndarray  # bool, ground truth
    r0: float
    course: list[tuple[str, float, float]] = field(default_factory=list)  # kind, x, y in course order
    road: np.ndarray | None = None  # bool


def _boundary_point(kind: str, x: float, y: float, toward: tuple[float, float], r0: float) -> tuple[float, float]:
    d = math.hypot(toward[0] - x, toward[1] - y)
    ux, uy = (toward[0] - x) / d, (toward[1] - y) / d
    rad = {"start": r0 * 1.25, "finish": r0 * 7 / 6, "control": r0}[kind] + 0.15 * r0
    return x + ux * rad, y + uy * rad


def draw_hatch(img: np.ndarray, box: tuple[int, int, int, int], ppm: float, spacing_mm: float = 1.6) -> None:
    """Diagonal purple grid clipped to ``box`` (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = box
    layer = np.zeros(img.shape[:2], np.uint8)
    step = max(4, int(round(spacing_mm * ppm)))
    stroke = max(1, int(round(0.2 * ppm)))
    span = (x1 - x0) + (y1 - y0)
    for d in range(-span, span, step):
        cv2.line(layer, (x0 + d, y0), (x0 + d + span, y0 + span), 1, stroke, cv2.LINE_AA)
        cv2.line(layer, (x0 + d, y1), (x0 + d + span, y1 - span), 1, stroke, cv2.LINE_AA)
    clip = np.zeros_like(layer)
    clip[y0 : y1 + 1, x0 : x1 + 1] = 1
    img[(layer > 0) & (clip > 0)] = bgr(PURPLE)


def draw_course(img: np.ndarray, course: list[tuple[str, float, float]], r0: float, numbers: bool = True) -> None:
    ppm = 2 * r0 / 6.0
    stroke = max(1, int(round(0.35 * ppm)))
    col = bgr(PURPLE)
    n = 0
    for kind, x, y in course:
        c = (int(round(x)), int(round(y)))
        if kind == "start":
            side = 7.0 * ppm
            rad = side / math.sqrt(3)
            pts = [(x + rad * math.cos(math.radians(a)), y + rad * math.sin(math.radians(a))) for a in (-90, 30, 150)]
            cv2.polylines(img, [np.rint(pts).astype(np.int32)], True, col, stroke, cv2.LINE_AA)
        elif kind == "finish":
            cv2.circle(img, c, int(round(2.5 * ppm)), col, stroke, cv2.LINE_AA)
            cv2.circle(img, c, int(round(3.5 * ppm)), col, stroke, cv2.LINE_AA)
        else:
            n += 1
            cv2.circle(img, c, int(round(r0)), col, stroke, cv2.LINE_AA)
            if numbers:
                cv2.putText(img, str(n), (c[0] + int(1.1 * r0), c[1] - int(0.9 * r0)), cv2.FONT_HERSHEY_SIMPLEX, r0 / 22, col, stroke, cv2.LINE_AA)
    for (ka, xa, ya), (kb, xb, yb) in zip(course, course[1:]):
        p1 = _boundary_point(ka, xa, ya, (xb, yb), r0)
        p2 = _boundary_point(kb, xb, yb, (xa, ya), r0)
        cv2.line(img, (int(round(p1[0])), int(round(p1[1]))), (int(round(p2[0])), int(round(p2[1]))), col, stroke, cv2.LINE_AA)


def make_map(seed: int = 0, r0: float = 18.0, noise: float = 4.0) -> SyntheticMap:
    rng = np.random.default_rng(seed)
    h, w = 700, 900
    ppm = 2 * r0 / 6.0
    img = np.full((h, w, 3), bgr(WHITE), np.uint8)
    forb = np.zeros((h, w), np.uint8)
    road = np.zeros((h, w), np.uint8)

    # terrain
    cv2.rectangle(img, (0, 560), (w, h), bgr(BLUE), -1)  # lake
    cv2.rectangle(forb, (0, 560), (w, h), 1, -1)
    cv2.rectangle(img, (600, 40), (860, 230), bgr(OLIVE), -1)
    cv2.rectangle(forb, (600, 40), (860, 230), 1, -1)
    cv2.rectangle(img, (60, 380), (250, 520), bgr(GREEN), -1)
    cv2.rectangle(img, (300, 420), (520, 520), bgr(YELLOW), -1)
    # roads
    cv2.rectangle(img, (0, 280), (w, 320), bgr(BEIGE), -1)
    cv2.rectangle(road, (0, 280), (w, 320), 1, -1)
    cv2.rectangle(img, (420, 0), (455, 560), bgr(BEIGE), -1)
    cv2.rectangle(road, (420, 0), (455, 560), 1, -1)
    # buildings with black outline, a narrow passage (6 px) between two of them
    for x0, y0, x1, y1 in [(80, 60, 200, 160), (206, 60, 330, 160), (480, 360, 580, 400), (620, 360, 760, 520), (90, 190, 180, 250)]:
        cv2.rectangle(img, (x0, y0), (x1, y1), bgr(GREY), -1)
        cv2.rectangle(img, (x0, y0), (x1, y1), bgr(BLACK), 1)
        cv2.rectangle(forb, (x0, y0), (x1, y1), 1, -1)
    # thick wall (impassable) and thin lines (passable)
    wall_w = max(2, int(round(0.6 * ppm)))
    cv2.line(img, (500, 120), (580, 250), bgr(BLACK), wall_w)
    cv2.line(forb, (500, 120), (580, 250), 1, wall_w)
    cv2.line(img, (260, 340), (400, 400), bgr(BLACK), 1)
    # magnetic north lines (thin blue, passable)
    for x in (150, 450, 750):
        cv2.line(img, (x, 0), (x, 560), bgr(BLUE), 1)

    # purple cross-hatched out-of-bounds area (forbidden)
    draw_hatch(img, (250, 335, 400, 405), ppm)
    cv2.rectangle(forb, (250, 335), (400, 405), 1, -1)

    course = [
        ("start", 300.0, 220.0),
        ("control", 120.0, 300.0),
        ("control", 360.0, 470.0),
        ("control", 560.0, 300.0),
        ("control", 700.0, 290.0),
        ("finish", 520.0, 470.0),
    ]
    draw_course(img, course, r0)

    img = cv2.GaussianBlur(img, (3, 3), 0.6)
    if noise > 0:
        img = np.clip(img.astype(np.float32) + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return SyntheticMap(img, forb.astype(bool), r0, course, road.astype(bool))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 1.0
