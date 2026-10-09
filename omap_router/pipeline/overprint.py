"""Stage 8.3: course overprint (purple) mask and removal from the terrain labels."""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from ..config import TerrainClass


def purple_mask(labels: np.ndarray) -> np.ndarray:
    return labels == TerrainClass.PURPLE


def remove_purple(labels: np.ndarray, purple: np.ndarray, dilate_px: int = 0) -> np.ndarray:
    """Replace overprint pixels with the label of the nearest non-purple pixel.

    ``dilate_px`` grows the mask first so anti-aliased fringes (purple blended
    with the background, often classified as grey/brown) are replaced too.
    """
    mask = purple
    if dilate_px > 0 and mask.any():
        k = 2 * dilate_px + 1
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    if not mask.any() or mask.all():
        return labels.copy()
    _, (ri, ci) = ndimage.distance_transform_edt(mask, return_indices=True)
    out = labels.copy()
    out[mask] = labels[ri[mask], ci[mask]]
    return out
