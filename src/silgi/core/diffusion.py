"""Fast inpainting by diffusion: Telea's fast marching method and Navier–Stokes.

Both propagate surrounding colours smoothly into the hole. They are very fast
and ideal for thin text strokes over smooth backgrounds, but they blur texture.

References: Telea, "An Image Inpainting Technique Based on the Fast Marching
Method", 2004; Bertalmío, Bertozzi & Sapiro, "Navier–Stokes, Fluid Dynamics,
and Image and Video Inpainting", 2001.
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

from ._types import BoolMask, FloatImage, Image, MaskLike
from .roi import run_in_roi

DiffusionMethod = Literal["telea", "ns"]
_FLAGS = {"telea": cv2.INPAINT_TELEA, "ns": cv2.INPAINT_NS}


def fill(
    roi: FloatImage, mask: BoolMask, *, method: DiffusionMethod = "telea", radius: float = 3.0
) -> FloatImage:
    """Inpaint ``mask`` inside a float ROI. Works per channel in float32 so
    16-bit and alpha channels are handled exactly like 8-bit RGB."""
    if method not in _FLAGS:
        raise ValueError(f"unknown diffusion method {method!r}")
    m8 = mask.astype(np.uint8) * 255
    out = np.empty_like(roi)
    for c in range(roi.shape[2]):
        # OpenCV's thresholds assume an 8-bit-like value range
        chan = np.ascontiguousarray(roi[:, :, c] * np.float32(255.0))
        res = cv2.inpaint(chan, m8, float(radius), _FLAGS[method])
        out[:, :, c] = res * np.float32(1.0 / 255.0)
    return np.clip(out, 0.0, 1.0)


def remove(
    image: Image,
    mask: MaskLike,
    *,
    method: DiffusionMethod = "telea",
    radius: float = 3.0,
) -> Image:
    """Remove the masked area by diffusion. Pixels outside ``mask`` are untouched."""
    margin = int(np.ceil(radius)) + 4
    return run_in_roi(
        image, mask, lambda r, m, _box: fill(r, m, method=method, radius=radius), margin=margin
    )
