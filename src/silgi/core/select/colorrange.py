"""Colour range: select every pixel perceptually close to a picked colour.

Distances are measured in CIELAB (ΔE76), so the tolerance behaves the same for
dark and light colours. Great for single-colour logos and text.
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from .._types import BoolMask, Image
from ..color import to_float


def to_lab(image: Image) -> npt.NDArray[np.float32]:
    f = to_float(image)
    rgb = np.repeat(f[:, :, :1], 3, axis=2) if f.shape[2] < 3 else f[:, :, :3]
    return np.asarray(cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2Lab), dtype=np.float32)


def color_range(
    image: Image,
    color: tuple[float, float, float],
    tolerance: float = 20.0,
    *,
    softness: float = 0.0,
) -> BoolMask | npt.NDArray[np.float32]:
    """Pixels within ``tolerance`` ΔE of ``color`` (RGB in 0..1).

    With ``softness`` > 0 a float mask is returned that fades from 1 at
    ``tolerance`` to 0 at ``tolerance + softness``.
    """
    lab = to_lab(image)
    ref = cv2.cvtColor(np.array([[color]], np.float32), cv2.COLOR_RGB2Lab)[0, 0]
    dist = np.sqrt(((lab - ref) ** 2).sum(axis=2))
    if softness <= 0:
        return np.asarray(dist <= tolerance, dtype=bool)
    return np.asarray(np.clip((tolerance + softness - dist) / softness, 0.0, 1.0), dtype=np.float32)


def sample_color(image: Image, x: int, y: int, radius: int = 1) -> tuple[float, float, float]:
    """Average RGB (0..1) around a clicked point."""
    f = to_float(image)
    h, w = f.shape[:2]
    patch = f[
        max(0, y - radius) : min(h, y + radius + 1), max(0, x - radius) : min(w, x + radius + 1)
    ]
    mean = patch.reshape(-1, f.shape[2]).mean(axis=0)
    if mean.size < 3:
        return float(mean[0]), float(mean[0]), float(mean[0])
    return float(mean[0]), float(mean[1]), float(mean[2])
