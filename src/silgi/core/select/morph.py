"""Mask operations: combine, grow, shrink, feather, smooth, load and save."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image as PILImage

from .._types import BoolMask

CombineMode = Literal["replace", "add", "subtract", "intersect"]


def _disk(radius: int) -> npt.NDArray[np.uint8]:
    k = 2 * radius + 1
    return np.asarray(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), dtype=np.uint8)


def combine(current: BoolMask, new: BoolMask, mode: CombineMode = "add") -> BoolMask:
    """Merge a new selection into the current one."""
    if current.shape != new.shape:
        raise ValueError("masks must have the same shape")
    if mode == "replace":
        return new.copy()
    if mode == "add":
        return current | new
    if mode == "subtract":
        return current & ~new
    if mode == "intersect":
        return current & new
    raise ValueError(f"unknown combine mode {mode!r}")


def grow(mask: BoolMask, px: int) -> BoolMask:
    """Dilate by ``px`` pixels (round structuring element)."""
    if px <= 0:
        return mask.copy()
    out = cv2.dilate(mask.astype(np.uint8), _disk(px))
    return np.asarray(out, dtype=bool)


def shrink(mask: BoolMask, px: int) -> BoolMask:
    """Erode by ``px`` pixels. The image border does not erode the mask."""
    if px <= 0:
        return mask.copy()
    out = cv2.erode(mask.astype(np.uint8), _disk(px), borderType=cv2.BORDER_REPLICATE)
    return np.asarray(out, dtype=bool)


def smooth(mask: BoolMask, px: int) -> BoolMask:
    """Round off jagged edges: blur and re-threshold at 50 %."""
    if px <= 0:
        return mask.copy()
    k = 2 * px + 1
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (k, k), px / 2.0)
    return np.asarray(blurred >= 0.5, dtype=bool)


def feather(mask: BoolMask, px: float) -> npt.NDArray[np.float32]:
    """Soft weights fading from 0 at the edge to 1 at ``px`` inside.

    Feathering only goes inward, so a feathered selection never touches pixels
    that were not selected.
    """
    if px <= 0:
        return mask.astype(np.float32)
    dist = cv2.distanceTransform(np.pad(mask, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]
    return np.asarray(np.clip(dist / np.float32(px + 1.0), 0.0, 1.0), dtype=np.float32)


def load_mask(src: str | Path | bytes, shape: tuple[int, int] | None = None) -> BoolMask:
    """Read a mask PNG. White (or opaque, for RGBA) means selected."""
    data = src if isinstance(src, bytes) else Path(src).read_bytes()
    pil = PILImage.open(io.BytesIO(data))
    if pil.mode in ("RGBA", "LA", "PA") or "transparency" in pil.info:
        arr = np.asarray(pil.convert("RGBA"))[:, :, 3]
        # fully opaque masks carry the selection in their colour instead
        if arr.min() == arr.max():
            arr = np.asarray(pil.convert("L"))
    else:
        arr = np.asarray(pil.convert("L"))
    mask = np.asarray(arr >= 128, dtype=bool)
    if shape is not None and mask.shape != tuple(shape):
        raise ValueError(f"mask is {mask.shape[1]}x{mask.shape[0]}, image is {shape[1]}x{shape[0]}")
    return mask


def mask_to_png(mask: npt.NDArray[np.bool_] | npt.NDArray[np.float32]) -> bytes:
    """Encode a mask as an 8-bit grayscale PNG (white = selected)."""
    if mask.dtype == np.bool_:
        arr = mask.astype(np.uint8) * 255
    else:
        arr = np.clip(np.rint(mask * 255), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", arr)
    if not ok:  # pragma: no cover
        raise RuntimeError("PNG encoding failed")
    return bytes(buf.tobytes())


def save_mask(path: str | Path, mask: npt.NDArray[np.bool_] | npt.NDArray[np.float32]) -> None:
    Path(path).write_bytes(mask_to_png(mask))
