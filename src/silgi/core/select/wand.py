"""Magic wand: select pixels similar in colour to a clicked point."""

from __future__ import annotations

import cv2
import numpy as np

from .._types import BoolMask, Image
from ..color import to_float


def color_distance(image: Image, color: np.ndarray) -> np.ndarray:
    """Largest per-channel difference to ``color`` (0..1), ignoring alpha."""
    f = to_float(image)
    n = 3 if f.shape[2] >= 3 else 1
    return np.asarray(np.abs(f[:, :, :n] - color[:n]).max(axis=2), dtype=np.float32)


def magic_wand(
    image: Image,
    x: int,
    y: int,
    tolerance: float = 0.12,
    *,
    contiguous: bool = True,
) -> BoolMask:
    """Select pixels within ``tolerance`` (0..1, per channel) of the colour at (x, y).

    With ``contiguous`` only the connected region containing the clicked pixel
    is selected (8-connectivity); otherwise every matching pixel is.
    """
    h, w = image.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError("seed point is outside the image")
    seed = to_float(image[y : y + 1, x : x + 1])[0, 0]
    within = color_distance(image, seed) <= tolerance + 1e-6
    if not contiguous:
        return within
    _, labels = cv2.connectedComponents(within.astype(np.uint8), connectivity=8)
    return np.asarray(labels == labels[y, x], dtype=bool)
