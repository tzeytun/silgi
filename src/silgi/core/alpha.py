"""Reverse alpha blending: recover the original pixels under a known watermark.

A semi-transparent watermark ``W`` with opacity ``α`` turns an original pixel
``J`` into ``I = α·W + (1 − α)·J``. When ``W`` and ``α`` are known this can be
inverted exactly:

    J = (I − α·W) / (1 − α)

This recovers real image content rather than inventing it. Where ``α`` is
close to 1 the original is effectively gone (and noise is amplified by
``1 / (1 − α)``), so those pixels are filled by inpainting instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from . import diffusion, patchmatch
from ._types import BoolMask, Box, FloatImage, Image, MaskLike
from .align import gradient_magnitude, locate, shift_watermark
from .color import luminance, to_float
from .progress import NULL_PROGRESS, Progress
from .roi import expand_box, mask_bbox, normalize_mask, run_in_roi

Fallback = Literal["patchmatch", "telea"]
F32 = npt.NDArray[np.float32]


@dataclass(frozen=True)
class Placement:
    """Where the watermark's top-left corner sits (may be fractional)."""

    top: float
    left: float
    score: float | None = None


def as_watermark(watermark: Any) -> tuple[FloatImage, F32]:
    """Normalise a watermark to (rgb float HxWx3, alpha float HxW).

    Accepts an RGBA array (uint8 / uint16 / float), a ``(rgb, alpha)`` pair or
    any object with ``.rgb`` and ``.alpha`` attributes.
    """
    if hasattr(watermark, "rgb") and hasattr(watermark, "alpha"):
        rgb, a = watermark.rgb, watermark.alpha
        return to_float(np.asarray(rgb))[:, :, :3], np.asarray(a, np.float32)
    if isinstance(watermark, tuple):
        rgb, a = watermark
        return to_float(np.asarray(rgb))[:, :, :3], to_float(np.asarray(a))[:, :, 0]
    arr = np.asarray(watermark)
    if arr.ndim != 3 or arr.shape[2] not in (2, 4):
        raise ValueError("watermark must be an RGBA (or gray+alpha) image with transparency")
    f = to_float(arr)
    if f.shape[2] == 2:
        return np.repeat(f[:, :, :1], 3, axis=2), f[:, :, 1].copy()
    return f[:, :, :3].copy(), f[:, :, 3].copy()


def _color_channels(n_channels: int) -> int:
    return 1 if n_channels <= 2 else 3


def _match_channels(rgb: FloatImage, n_color: int) -> FloatImage:
    if n_color == 3:
        return rgb
    return luminance(rgb)[..., None]


def reverse(
    observed: FloatImage, rgb: FloatImage, alpha: F32, threshold: float = 0.85
) -> tuple[FloatImage, BoolMask]:
    """Invert ``I = αW + (1 − α)J``. Returns J and the mask of pixels with α ≥ threshold."""
    a = np.minimum(alpha, np.float32(threshold))[..., None]
    j = (observed - a * rgb) / (1.0 - a)
    return np.clip(j, 0.0, 1.0).astype(np.float32), alpha >= threshold


def _placed(
    shape: tuple[int, ...], rgb: FloatImage, alpha: F32, top: float, left: float
) -> tuple[Box, FloatImage, F32]:
    """Watermark arrays positioned in image coordinates, clipped to the image."""
    ti, li = int(np.floor(top)), int(np.floor(left))
    fy, fx = top - ti, left - li
    if abs(fy) > 1e-3 or abs(fx) > 1e-3:
        rgb, alpha = shift_watermark(rgb, alpha, fy, fx)
        ti, li = ti - 1, li - 1
    h, w = alpha.shape
    y0, x0 = max(ti, 0), max(li, 0)
    y1, x1 = min(ti + h, shape[0]), min(li + w, shape[1])
    if y0 >= y1 or x0 >= x1:
        raise ValueError("watermark position lies outside the image")
    sy, sx = y0 - ti, x0 - li
    return (
        (y0, y1, x0, x1),
        rgb[sy : sy + (y1 - y0), sx : sx + (x1 - x0)],
        alpha[sy : sy + (y1 - y0), sx : sx + (x1 - x0)],
    )


def alignment_energy(
    image_f: FloatImage, rgb: FloatImage, alpha: F32, top: float, left: float
) -> float:
    """How much watermark edge is left after reversing at this placement (lower is better)."""
    n_color = _color_channels(image_f.shape[2])
    box, wrgb, wa = _placed(image_f.shape, rgb, alpha, top, left)
    y0, y1, x0, x1 = box
    obs = image_f[y0:y1, x0:x1, :n_color]
    j, _ = reverse(obs, _match_channels(wrgb, n_color), wa, 0.95)
    edges = gradient_magnitude(np.ascontiguousarray(wa)) > 1e-3
    if not edges.any():
        return 0.0
    g = gradient_magnitude(np.ascontiguousarray(luminance(j) if n_color == 3 else j[:, :, 0]))
    return float(np.mean(g[edges] ** 2))


def refine_placement(
    image_f: FloatImage, rgb: FloatImage, alpha: F32, top: float, left: float
) -> Placement:
    """Pick the placement near (top, left) that leaves the least watermark edge."""
    ty, tx = round(top), round(left)
    best = (np.inf, ty, tx)
    energies: dict[tuple[int, int], float] = {}
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            try:
                e = alignment_energy(image_f, rgb, alpha, ty + dy, tx + dx)
            except ValueError:
                continue
            energies[(dy, dx)] = e
            if e < best[0]:
                best = (e, ty + dy, tx + dx)
    e0, by, bx = best
    # sub-pixel step from a parabola through the energy, kept only if it helps
    oy = ox = 0.0
    cy, cx = by - ty, bx - tx
    if (cy - 1, cx) in energies and (cy + 1, cx) in energies:
        m1, p1 = energies[(cy - 1, cx)], energies[(cy + 1, cx)]
        den = m1 - 2 * e0 + p1
        oy = float(np.clip(0.5 * (m1 - p1) / den, -0.5, 0.5)) if den > 1e-12 else 0.0
    if (cy, cx - 1) in energies and (cy, cx + 1) in energies:
        m1, p1 = energies[(cy, cx - 1)], energies[(cy, cx + 1)]
        den = m1 - 2 * e0 + p1
        ox = float(np.clip(0.5 * (m1 - p1) / den, -0.5, 0.5)) if den > 1e-12 else 0.0
    if abs(oy) > 0.05 or abs(ox) > 0.05:
        e_sub = alignment_energy(image_f, rgb, alpha, by + oy, bx + ox)
        if e_sub < e0:
            return Placement(by + oy, bx + ox)
    return Placement(float(by), float(bx))


def find_placement(
    image: Image,
    watermark: Any,
    *,
    region: Box | None = None,
    refine: bool = True,
) -> Placement:
    """Locate a known watermark by gradient NCC, optionally refined by energy."""
    rgb, a = as_watermark(watermark)
    m = locate(image, a, region=region)
    if not refine:
        return Placement(m.top, m.left, m.score)
    p = refine_placement(to_float(image), rgb, a, m.top, m.left)
    return Placement(p.top, p.left, m.score)


def remove(
    image: Image,
    mask: MaskLike | None = None,
    *,
    watermark: Any,
    position: tuple[float, float] | None = None,
    search_region: Box | None = None,
    refine: bool = True,
    alpha_threshold: float = 0.85,
    fallback: Fallback = "patchmatch",
    quality: patchmatch.Quality = "balanced",
    progress: Progress = NULL_PROGRESS,
) -> Image:
    """Undo a known watermark. Only pixels covered by both the watermark and
    ``mask`` (when given) change.

    ``position`` is the watermark's top-left corner; when omitted it is found
    automatically, searching near the mask if one is given.
    """
    rgb, a = as_watermark(watermark)
    shape = image.shape
    user: BoolMask | None = None
    if mask is not None:
        user, _ = normalize_mask(mask, shape)
        if not user.any():
            return image.copy()

    if position is None:
        region = search_region
        if region is None and user is not None:
            ub = mask_bbox(user)
            assert ub is not None
            region = expand_box(ub, max(a.shape) // 2 + 8, shape)
        place = find_placement(image, (rgb, a), region=region, refine=refine)
    else:
        place = Placement(*position)

    box, wrgb, wa = _placed(shape, rgb, a, place.top, place.left)
    y0, y1, x0, x1 = box
    sel = np.zeros(shape[:2], bool)
    sel[y0:y1, x0:x1] = wa > (1.0 / 1024.0)
    if user is not None:
        sel &= user
    if not sel.any():
        return image.copy()

    n_color = _color_channels(1 if image.ndim == 2 else shape[2])
    wcol = _match_channels(wrgb, n_color)
    lost = np.zeros(shape[:2], bool)
    lost[y0:y1, x0:x1] = wa >= alpha_threshold
    lost &= sel
    margin = patchmatch.default_margin(lost, 7) if lost.any() else 1
    progress.report(0.05, "reversing blend")

    def _fill(roi: FloatImage, roi_mask: BoolMask, rb: Box) -> FloatImage:
        out = roi.copy()
        oy, ox = y0 - rb[0], x0 - rb[2]
        h, w = wa.shape
        obs = roi[oy : oy + h, ox : ox + w, :n_color]
        j, _ = reverse(obs, wcol, wa, alpha_threshold)
        out[oy : oy + h, ox : ox + w, :n_color] = j
        hole = lost[rb[0] : rb[1], rb[2] : rb[3]]
        if hole.any():
            if fallback == "patchmatch":
                try:
                    out = patchmatch.fill(
                        out, hole, params=patchmatch.PRESETS[quality], progress=progress
                    )
                except patchmatch.NoSourceError:
                    out = diffusion.fill(out, hole)
            else:
                out = diffusion.fill(out, hole)
        return out

    result = run_in_roi(image, sel, _fill, margin=margin, include=box)
    progress.report(1.0, "done")
    return result
