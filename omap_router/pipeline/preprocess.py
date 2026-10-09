"""Stage 8.1: crop, denoise and white balance."""

from __future__ import annotations

import cv2
import numpy as np

from ..config import Settings


def apply_crop(img: np.ndarray, crop: tuple[int, int, int, int] | None) -> np.ndarray:
    if crop is None:
        return img
    x0, y0, x1, y1 = crop
    h, w = img.shape[:2]
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"empty crop {crop}")
    return img[y0:y1, x0:x1]


def white_balance(rgb: np.ndarray, percentile: float) -> np.ndarray:
    """Scale channels so that the brightest ``100 - percentile`` % maps to white."""
    gray = rgb.mean(axis=2)
    thr = np.percentile(gray, percentile)
    bright = gray >= thr
    if bright.sum() < 10:
        return rgb
    ref = rgb[bright].reshape(-1, 3).mean(axis=0)
    gain = 255.0 / np.maximum(ref, 1.0)
    out = rgb.astype(np.float32) * gain[None, None, :]
    return np.clip(out, 0, 255).astype(np.uint8)


def preprocess(img_bgr: np.ndarray, crop: tuple[int, int, int, int] | None, cfg: Settings) -> np.ndarray:
    """Returns the cropped, filtered, white balanced image as RGB uint8."""
    img = apply_crop(img_bgr, crop)
    img = cv2.bilateralFilter(
        np.ascontiguousarray(img), cfg.bilateral_d_px, cfg.bilateral_sigma_color, cfg.bilateral_sigma_space
    )
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return white_balance(rgb, cfg.white_percentile)
