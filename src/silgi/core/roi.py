"""Region-of-interest processing and the untouched-pixels guarantee.

Every removal method runs through :func:`run_in_roi`. It crops the mask's
bounding box plus a context margin, hands a float32 copy to the method and
writes the method's output back **only where the mask is set**. Pixels outside
the mask are never assigned, so they stay bit-identical to the input.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from ._types import BoolMask, Box, FloatImage, Image, MaskLike
from .color import as_hwc, from_float, to_float

FillFn = Callable[[FloatImage, BoolMask, Box], FloatImage]


def normalize_mask(
    mask: MaskLike, shape: tuple[int, ...]
) -> tuple[BoolMask, npt.NDArray[np.float32] | None]:
    """Split a mask into a hard selection and optional soft weights.

    Accepts bool, uint8 (0..255), uint16 or float (0..1) masks. Returns the
    boolean selection (``mask > 0``) and per-pixel weights in [0, 1], or
    ``None`` when every selected pixel has full weight.
    """
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]
    if mask.ndim != 2 or mask.shape != tuple(shape[:2]):
        raise ValueError(f"mask shape {mask.shape} does not match image shape {shape[:2]}")
    if mask.dtype == np.bool_:
        return mask, None
    if mask.dtype == np.uint8:
        weights = mask.astype(np.float32) / 255.0
    elif mask.dtype == np.uint16:
        weights = mask.astype(np.float32) / 65535.0
    elif mask.dtype.kind == "f":
        weights = np.clip(np.nan_to_num(mask.astype(np.float32)), 0.0, 1.0)
    else:
        raise TypeError(f"unsupported mask dtype: {mask.dtype}")
    hard = weights > 0
    if np.all(weights[hard] >= 1.0):
        return hard, None
    return hard, weights


def mask_bbox(mask: BoolMask) -> Box | None:
    """Bounding box of the set pixels, or ``None`` for an empty mask."""
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def expand_box(box: Box, margin: int, shape: tuple[int, ...]) -> Box:
    y0, y1, x0, x1 = box
    h, w = shape[:2]
    return max(0, y0 - margin), min(h, y1 + margin), max(0, x0 - margin), min(w, x1 + margin)


def union_box(a: Box, b: Box) -> Box:
    return min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3])


def clip_box(box: Box, shape: tuple[int, ...]) -> Box:
    h, w = shape[:2]
    y0, y1, x0, x1 = box
    return max(0, y0), min(h, y1), max(0, x0), min(w, x1)


def paste(
    image: Image,
    filled: FloatImage,
    box: Box,
    hard: BoolMask,
    weights: npt.NDArray[np.float32] | None = None,
) -> Image:
    """Return a copy of ``image`` with ``filled`` written inside ``hard`` only.

    ``filled`` covers ``box`` and is float in [0, 1]. With soft ``weights`` the
    written value is ``original + w * (filled - original)``.
    """
    y0, y1, x0, x1 = box
    out = image.copy()
    region = as_hwc(out)[y0:y1, x0:x1]
    sel = hard[y0:y1, x0:x1]
    values = np.nan_to_num(filled, nan=0.0, posinf=1.0, neginf=0.0)
    if weights is not None:
        original = to_float(image[y0:y1, x0:x1])
        w = weights[y0:y1, x0:x1, None]
        values = original + w * (values - original)
    converted = from_float(values.astype(np.float32), image.dtype)
    region[sel] = converted[sel]
    return out


def run_in_roi(
    image: Image,
    mask: MaskLike,
    fill: FillFn,
    *,
    margin: int,
    include: Box | None = None,
) -> Image:
    """Apply ``fill`` to the mask's region of interest and paste back.

    ``fill`` receives the ROI as float32 HxWxC in [0, 1], the ROI mask and the
    ROI box in image coordinates, and must return a float32 array of the same
    shape. ``include`` forces an extra
    box (e.g. a source region) into the ROI.
    """
    hard, weights = normalize_mask(mask, image.shape)
    box = mask_bbox(hard)
    if box is None:
        return image.copy()
    roi_box = expand_box(box, margin, image.shape)
    if include is not None:
        roi_box = union_box(roi_box, clip_box(include, image.shape))
    y0, y1, x0, x1 = roi_box
    roi = to_float(image[y0:y1, x0:x1])
    filled = fill(roi, hard[y0:y1, x0:x1].copy(), roi_box)
    if filled.shape != roi.shape:
        raise RuntimeError(f"fill returned shape {filled.shape}, expected {roi.shape}")
    return paste(image, filled, roi_box, hard, weights)


def roi_box_for(mask: MaskLike, shape: tuple[int, ...], margin: int) -> Box | None:
    """The ROI :func:`run_in_roi` would use for ``mask`` (for previews and tests)."""
    hard, _ = normalize_mask(mask, shape)
    box = mask_bbox(hard)
    return None if box is None else expand_box(box, margin, shape)
