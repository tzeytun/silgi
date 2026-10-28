"""Clone stamp and healing brush.

Both copy pixels from a source area at a fixed offset into the masked area.
*Clone* copies them verbatim. *Heal* copies only their texture (gradients) and
solves a Poisson equation so brightness and colour adapt to the surroundings
(Pérez et al. 2003), which hides the seam.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from . import poisson
from ._types import BoolMask, Box, FloatImage, Image, MaskLike
from .roi import mask_bbox, normalize_mask, run_in_roi

HealMode = Literal["clone", "heal"]


def remove(
    image: Image,
    mask: MaskLike,
    *,
    offset: tuple[int, int],
    mode: HealMode = "heal",
) -> Image:
    """Replace the masked area with pixels from ``offset`` = (dy, dx) away.

    The pixel at (y, x) takes its value from (y + dy, x + dx). The source must
    lie inside the image.
    """
    if mode not in ("clone", "heal"):
        raise ValueError(f"unknown mode {mode!r}")
    dy, dx = int(offset[0]), int(offset[1])
    hard, _ = normalize_mask(mask, image.shape)
    box = mask_bbox(hard)
    if box is None:
        return image.copy()
    h, w = image.shape[:2]
    src_box: Box = (box[0] + dy, box[1] + dy, box[2] + dx, box[3] + dx)
    if src_box[0] < 0 or src_box[2] < 0 or src_box[1] > h or src_box[3] > w:
        raise ValueError("source area falls outside the image")
    if dy == 0 and dx == 0:
        raise ValueError("offset must not be zero")

    def _fill(roi: FloatImage, m: BoolMask, rb: Box) -> FloatImage:
        if mode == "clone":
            out = roi.copy()
            ys, xs = np.nonzero(m)
            out[ys, xs] = roi[ys + dy, xs + dx]
            return out
        # the guide is the whole ROI shifted by the offset, so gradients across
        # the mask boundary also come from the source area
        rh, rw = m.shape
        guide = roi.copy()
        ty0, ty1 = max(0, -dy), min(rh, rh - dy)
        tx0, tx1 = max(0, -dx), min(rw, rw - dx)
        guide[ty0:ty1, tx0:tx1] = roi[ty0 + dy : ty1 + dy, tx0 + dx : tx1 + dx]
        return poisson.blend(roi, guide, m)

    # one extra pixel around the source so boundary gradients are available
    src_ctx: Box = (src_box[0] - 1, src_box[1] + 1, src_box[2] - 1, src_box[3] + 1)
    return run_in_roi(image, mask, _fill, margin=1, include=src_ctx)
