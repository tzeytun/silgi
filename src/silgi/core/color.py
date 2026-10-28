"""Pixel format conversions.

All removal methods work on float32 images in [0, 1] in the image's *native*
encoding (whatever colour space its ICC profile describes). Converting to sRGB
and back would alter pixels outside the mask, so conversion to sRGB happens only
for on-screen display (:func:`to_display_rgb8`).
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image as PILImage
from PIL import ImageCms

from ._types import FloatImage, Image

_MAX = {np.dtype(np.uint8): 255.0, np.dtype(np.uint16): 65535.0}


def max_value(dtype: npt.DTypeLike) -> float:
    """Return the full-scale value for an integer dtype, or 1.0 for float."""
    dt = np.dtype(dtype)
    if dt in _MAX:
        return _MAX[dt]
    if dt.kind == "f":
        return 1.0
    raise TypeError(f"unsupported image dtype: {dt}")


def as_hwc(image: Image) -> Image:
    """View a HxW image as HxWx1; HxWxC is returned unchanged."""
    if image.ndim == 2:
        return image[:, :, None]
    if image.ndim == 3 and 1 <= image.shape[2] <= 4:
        return image
    raise ValueError(f"expected HxW or HxWxC (C<=4) image, got shape {image.shape}")


def to_float(image: Image) -> FloatImage:
    """Convert an image to float32 in [0, 1], HxWxC."""
    img = as_hwc(image)
    scale = max_value(img.dtype)
    out = img.astype(np.float32)
    if scale != 1.0:
        out *= np.float32(1.0 / scale)
    return out


def from_float(image: FloatImage, dtype: npt.DTypeLike) -> Image:
    """Convert a float image in [0, 1] back to ``dtype`` with rounding and clipping."""
    dt = np.dtype(dtype)
    if dt.kind == "f":
        return np.clip(image, 0.0, 1.0).astype(dt)
    scale = np.float32(max_value(dt))
    out = np.clip(np.rint(image * scale), 0, scale)
    return out.astype(dt)


def to_8bit(image: Image) -> npt.NDArray[np.uint8]:
    """Reduce any supported image to 8 bits per channel (rounded, not truncated)."""
    if image.dtype == np.uint8:
        return image
    return from_float(to_float(image).reshape(image.shape), np.uint8)


def luminance(image: FloatImage) -> npt.NDArray[np.float32]:
    """Rec. 709 luma of a float HxWxC image (alpha ignored)."""
    c = image.shape[2]
    if c < 3:
        return image[:, :, 0].copy()
    lum = 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]
    return np.asarray(lum, dtype=np.float32)


_SRGB = ImageCms.createProfile("sRGB")


def to_display_rgb8(pixels: Image, icc_profile: bytes | None = None) -> npt.NDArray[np.uint8]:
    """Convert pixels to 8-bit sRGB (RGB or RGBA) for display only.

    The ICC profile, when present and valid, is honoured. Failures fall back to
    treating the pixels as sRGB, which is the common case anyway.
    """
    img8 = as_hwc(to_8bit(pixels))
    c = img8.shape[2]
    if c == 1:
        rgb = np.repeat(img8, 3, axis=2)
        alpha = None
    elif c == 2:
        rgb = np.repeat(img8[:, :, :1], 3, axis=2)
        alpha = img8[:, :, 1:2]
    else:
        rgb = img8[:, :, :3]
        alpha = img8[:, :, 3:4] if c == 4 else None

    if icc_profile and c >= 3:
        try:
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc_profile))
            pil = PILImage.fromarray(np.ascontiguousarray(rgb), "RGB")
            converted: Any = ImageCms.profileToProfile(pil, src, _SRGB, outputMode="RGB")
            rgb = np.asarray(converted, dtype=np.uint8)
        except (ImageCms.PyCMSError, OSError, ValueError):
            pass
    if alpha is not None:
        return np.ascontiguousarray(np.concatenate([rgb, alpha], axis=2))
    return np.ascontiguousarray(rgb)
