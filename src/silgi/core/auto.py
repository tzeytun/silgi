"""Choosing a removal method automatically from the mask and its surroundings."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ._types import Image, MaskLike
from .color import luminance, to_float
from .roi import expand_box, mask_bbox, normalize_mask

#: Masks at most this many pixels wide count as "thin" (text strokes, lines).
THIN_PX = 6.0
#: Median local standard deviation (0..1 scale) below which surroundings are smooth.
SMOOTH_STD = 0.02


@dataclass(frozen=True)
class Decision:
    method: str
    reason: str
    max_width: float = 0.0
    texture: float = 0.0


def mask_width(mask: np.ndarray) -> float:
    """Widest point of the mask in pixels (twice the largest inscribed radius)."""
    d = cv2.distanceTransform(np.pad(mask, 1).astype(np.uint8), cv2.DIST_L2, 5)
    return float(2.0 * np.max(d))


def surround_texture(image: Image, mask: np.ndarray, ring_px: int = 6) -> float:
    """Median local standard deviation of luminance in a ring around the mask."""
    box = mask_bbox(mask)
    if box is None:
        return 0.0
    y0, y1, x0, x1 = expand_box(box, ring_px + 3, image.shape)
    m = mask[y0:y1, x0:x1]
    k = np.ones((2 * ring_px + 1, 2 * ring_px + 1), np.uint8)
    ring = cv2.dilate(m.astype(np.uint8), k).astype(bool) & ~m
    if not ring.any():
        return 0.0
    lum = luminance(to_float(image[y0:y1, x0:x1]))
    mean = cv2.blur(lum, (5, 5))
    sq = cv2.blur(lum * lum, (5, 5))
    std = np.sqrt(np.maximum(sq - mean * mean, 0.0))
    return float(np.median(std[ring]))


def choose(
    image: Image,
    mask: MaskLike,
    *,
    has_watermark: bool = False,
    thin_px: float = THIN_PX,
    smooth_std: float = SMOOTH_STD,
) -> Decision:
    """Pick the most suitable method and explain why in one sentence."""
    if has_watermark:
        return Decision(
            "alpha",
            "A watermark image was provided, so the original pixels can be recovered "
            "by reversing the blend instead of being guessed.",
        )
    hard, _ = normalize_mask(mask, image.shape)
    if not hard.any():
        return Decision("none", "Nothing is selected.")
    width = mask_width(hard)
    texture = surround_texture(image, hard)
    if width <= thin_px and texture <= smooth_std:
        return Decision(
            "telea",
            f"The selection is thin (at most {width:.0f} px wide) on a smooth background, "
            "so fast diffusion fills it cleanly.",
            width,
            texture,
        )
    if width <= thin_px:
        why = f"The selection is thin ({width:.0f} px) but the background is textured"
    else:
        why = f"The selection is {width:.0f} px wide"
    return Decision(
        "patchmatch",
        f"{why}, so texture is copied from the surroundings with PatchMatch.",
        width,
        texture,
    )
