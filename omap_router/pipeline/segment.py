"""Stage 8.2: nearest reference colour classification in Lab space."""

from __future__ import annotations

import cv2
import numpy as np

from ..config import Settings, TerrainClass

N_CLASSES = len(TerrainClass)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 (..., 3) -> CIE Lab float32 (L in 0..100)."""
    arr = np.asarray(rgb, dtype=np.float32).reshape(-1, 1, 3) / 255.0
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    return lab.reshape(np.shape(rgb))


def reference_colors(
    rgb: np.ndarray,
    cfg: Settings,
    samples: dict[str, list[tuple[int, int]]] | None = None,
    offset: tuple[int, int] = (0, 0),
) -> list[tuple[TerrainClass, np.ndarray]]:
    """Reference (class, RGB) list.

    Defaults come from config. A class with user samples uses only the samples:
    each clicked point contributes the mean of a ``sample_patch_px`` square patch
    of the preprocessed image. ``offset`` is the crop origin (x, y) so that
    sample coordinates stay in original image space.
    """
    refs: dict[TerrainClass, list[np.ndarray]] = {
        TerrainClass[name]: [np.array(c, np.float32) for c in colors]
        for name, colors in cfg.reference_colors.items()
    }
    h, w = rgb.shape[:2]
    half = cfg.sample_patch_px // 2
    for name, points in (samples or {}).items():
        try:
            cls = TerrainClass[name.upper()]
        except KeyError:
            continue
        if cls == TerrainClass.UNKNOWN:
            continue
        means = []
        for x, y in points:
            cx, cy = int(round(x)) - offset[0], int(round(y)) - offset[1]
            if not (0 <= cx < w and 0 <= cy < h):
                continue
            patch = rgb[max(0, cy - half) : cy + half + 1, max(0, cx - half) : cx + half + 1]
            means.append(patch.reshape(-1, 3).mean(axis=0).astype(np.float32))
        if means:
            refs[cls] = means
    return [(cls, c) for cls, colors in refs.items() for c in colors]


def majority_fill(labels: np.ndarray, iterations: int) -> np.ndarray:
    """Replace UNKNOWN by the 3x3 majority of known neighbours."""
    labels = labels.copy()
    unknown_val = int(TerrainClass.UNKNOWN)
    for _ in range(iterations):
        unknown = labels == unknown_val
        if not unknown.any():
            break
        counts = np.zeros((N_CLASSES - 1,) + labels.shape, np.float32)
        for c in range(N_CLASSES - 1):
            counts[c] = cv2.boxFilter((labels == c).astype(np.float32), -1, (3, 3), normalize=False)
        best = counts.argmax(axis=0)
        has = counts.max(axis=0) > 0
        fill = unknown & has
        labels[fill] = best[fill]
    return labels


def segment(rgb: np.ndarray, refs: list[tuple[TerrainClass, np.ndarray]], cfg: Settings) -> np.ndarray:
    """Returns labels uint8 (H, W) of ``TerrainClass`` values."""
    lab = rgb_to_lab(rgb)
    ref_cls = np.array([int(c) for c, _ in refs], np.uint8)
    ref_lab = rgb_to_lab(np.stack([c for _, c in refs]).astype(np.uint8)[None, :, :])[0]
    h, w = rgb.shape[:2]
    flat = lab.reshape(-1, 3)
    best_d = np.full(flat.shape[0], np.inf, np.float32)
    best_i = np.zeros(flat.shape[0], np.int32)
    for i, ref in enumerate(ref_lab):
        d = np.sum((flat - ref) ** 2, axis=1)
        better = d < best_d
        best_d[better] = d[better]
        best_i[better] = i
    labels = ref_cls[best_i]
    labels[np.sqrt(best_d) > cfg.max_delta_e] = TerrainClass.UNKNOWN
    labels = labels.reshape(h, w)
    return majority_fill(labels, cfg.unknown_fill_iterations)
