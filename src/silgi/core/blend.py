"""Edge finishing: inward feathering and Poisson seam blending.

Both operate strictly inside the mask, so the untouched-pixels guarantee holds.
To blend over a wider band, grow the mask first.
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
import numpy.typing as npt

from . import poisson
from ._types import BoolMask, FloatImage

SeamMode = Literal["none", "poisson"]


def inward_ramp(mask: BoolMask, radius: float) -> npt.NDArray[np.float32]:
    """Weights that are 0 outside ``mask`` and rise to 1 at ``radius`` px inside."""
    if radius <= 0:
        return mask.astype(np.float32)
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    return np.asarray(np.clip(dist / np.float32(radius + 1.0), 0.0, 1.0), dtype=np.float32)


def feather(original: FloatImage, filled: FloatImage, mask: BoolMask, radius: float) -> FloatImage:
    """Fade ``filled`` into ``original`` over ``radius`` px inside the mask edge."""
    w = inward_ramp(mask, radius)[..., None]
    return (original + w * (filled - original)).astype(np.float32)


def seam(original: FloatImage, filled: FloatImage, mask: BoolMask) -> FloatImage:
    """Re-solve the fill in the gradient domain so it meets the surroundings
    without a visible step in brightness or colour."""
    return poisson.blend(original, filled, mask, boundary_gradients=False)


def finish(
    original: FloatImage,
    filled: FloatImage,
    mask: BoolMask,
    *,
    feather_px: float = 0.0,
    seam_mode: SeamMode = "none",
) -> FloatImage:
    out = filled
    if seam_mode == "poisson":
        out = seam(original, out, mask)
    elif seam_mode != "none":
        raise ValueError(f"unknown seam mode {seam_mode!r}")
    if feather_px > 0:
        out = feather(original, out, mask, feather_px)
    return out
