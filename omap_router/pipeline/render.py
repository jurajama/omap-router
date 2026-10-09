"""Layer PNGs: transparent RGBA overlays in original-image pixel space."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..config import CLASS_DISPLAY_COLORS, TerrainClass

LAYER_NAMES = ("labels", "forbidden", "outside", "purple", "hatch", "barriers", "cost")


def _place(rgba: np.ndarray, full_shape: tuple[int, int], offset: tuple[int, int]) -> np.ndarray:
    """Pad a cropped overlay back to the original image size."""
    h, w = full_shape
    if rgba.shape[:2] == (h, w) and offset == (0, 0):
        return rgba
    out = np.zeros((h, w, 4), np.uint8)
    x0, y0 = offset
    ch, cw = rgba.shape[:2]
    out[y0 : y0 + ch, x0 : x0 + cw] = rgba[: h - y0, : w - x0]
    return out


def labels_rgba(labels: np.ndarray) -> np.ndarray:
    pal = np.zeros((256, 4), np.uint8)
    for cls in TerrainClass:
        pal[int(cls), :3] = CLASS_DISPLAY_COLORS[cls]
        pal[int(cls), 3] = 255
    return pal[labels]


def mask_rgba(mask: np.ndarray, color: tuple[int, int, int], alpha: int = 255) -> np.ndarray:
    out = np.zeros(mask.shape + (4,), np.uint8)
    out[mask, :3] = color
    out[mask, 3] = alpha
    return out


def cost_rgba(cost: np.ndarray, cell_px: int, fine_shape: tuple[int, int]) -> np.ndarray:
    """Passable cost as a green (cheap) -> red (expensive) ramp; forbidden dark."""
    finite = np.isfinite(cost)
    out = np.zeros(cost.shape + (4,), np.uint8)
    if finite.any():
        lo, hi = float(cost[finite].min()), float(cost[finite].max())
        t = np.zeros(cost.shape, np.float32)
        if hi > lo:
            t[finite] = (cost[finite] - lo) / (hi - lo)
        out[..., 0] = (255 * t).astype(np.uint8)
        out[..., 1] = (255 * (1 - t)).astype(np.uint8)
        out[..., 2] = 40
        out[..., 3] = np.where(finite, 255, 0)
    out[~finite] = (40, 0, 40, 255)
    if cell_px > 1:
        out = np.repeat(np.repeat(out, cell_px, axis=0), cell_px, axis=1)
    return out[: fine_shape[0], : fine_shape[1]]


def write_png(path: Path, rgba: np.ndarray, full_shape: tuple[int, int], offset: tuple[int, int]) -> None:
    img = _place(rgba, full_shape, offset)
    bgra = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    tmp = path.with_suffix(".tmp.png")
    cv2.imwrite(str(tmp), bgra)
    tmp.replace(path)
