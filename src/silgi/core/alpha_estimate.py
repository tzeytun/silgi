"""Estimating an unknown watermark from several images that share it.

A classical take on Dekel et al., "On the Effectiveness of Visible Watermarks"
(CVPR 2017), without any learning:

1. **Localise.** Image content varies between photos but the watermark does
   not, so the per-pixel *median* of the image gradients across the set keeps
   only the watermark's edges. Their strongest cluster is the watermark box.
2. **Reconstruct.** Integrating the median gradients (a Poisson solve with a
   zero boundary) gives the watermark's shape; it is used to align each image
   to sub-region precision with normalised cross-correlation.
3. **Solve for α and W.** For every image k and pixel, ``I_k = αW + (1−α)J_k``.
   We estimate the unknown background ``J_k`` by diffusing its surroundings
   into the watermark area and solve the resulting linear model across all
   images by least squares, first for α and a colour, then for α with the
   colour fixed (most watermarks are a single colour).
4. **Refine.** A single global gain on α is tuned so that reversing the blend
   leaves the least watermark edge energy across the set.

Compared with Dekel et al. this skips their iterative multi-image matting
optimisation and image-level priors; it needs roughly ten or more images with
varied content and the watermark at the same size.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np
import numpy.typing as npt
from scipy import fft

from . import diffusion
from ._types import Box, FloatImage, Image
from .alpha import reverse
from .color import luminance, to_float
from .progress import NULL_PROGRESS, Progress

F32 = npt.NDArray[np.float32]

MIN_IMAGES = 3


@dataclass
class Estimate:
    """An estimated watermark and where it sits in each input image."""

    rgb: FloatImage
    alpha: F32
    box: Box  # (y0, y1, x0, x1) in the coordinates of the input images
    offsets: list[tuple[int, int]] = field(default_factory=list)
    single_color: bool = True

    def position(self, index: int) -> tuple[float, float]:
        dy, dx = self.offsets[index] if self.offsets else (0, 0)
        return float(self.box[0] + dy), float(self.box[2] + dx)

    def rgba8(self) -> npt.NDArray[np.uint8]:
        rgba = np.dstack([self.rgb, self.alpha])
        return np.asarray(np.clip(np.rint(rgba * 255), 0, 255), dtype=np.uint8)


def _rgb(image: Image) -> FloatImage:
    f = to_float(image)
    if f.shape[2] >= 3:
        return np.ascontiguousarray(f[:, :, :3])
    return np.repeat(f[:, :, :1], 3, axis=2)


def _fwd_grad(gray: F32) -> tuple[F32, F32]:
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, :-1] = gray[:, 1:] - gray[:, :-1]
    gy[:-1, :] = gray[1:, :] - gray[:-1, :]
    return gx, gy


def poisson_from_gradients(gx: F32, gy: F32) -> F32:
    """Integrate a gradient field with zero Dirichlet boundary (DST solver)."""
    h, w = gx.shape
    div = np.zeros((h, w), np.float64)
    div[:, 1:] += gx[:, 1:] - gx[:, :-1]
    div[:, 0] += gx[:, 0]
    div[1:, :] += gy[1:, :] - gy[:-1, :]
    div[0, :] += gy[0, :]
    inner = div[1:-1, 1:-1]
    if inner.size == 0:
        return np.zeros((h, w), np.float32)
    ih, iw = inner.shape
    d = fft.dstn(inner, type=1)
    ky = 2.0 * np.cos(np.pi * np.arange(1, ih + 1) / (ih + 1)) - 2.0
    kx = 2.0 * np.cos(np.pi * np.arange(1, iw + 1) / (iw + 1)) - 2.0
    d /= ky[:, None] + kx[None, :]
    sol = fft.idstn(d, type=1)
    out = np.zeros((h, w), np.float32)
    out[1:-1, 1:-1] = sol
    return out


def _median_gradients(grays: Sequence[F32]) -> tuple[F32, F32]:
    gxs, gys = zip(*(_fwd_grad(g) for g in grays), strict=True)
    return np.median(np.stack(gxs), axis=0).astype(np.float32), np.median(
        np.stack(gys), axis=0
    ).astype(np.float32)


def _detect_box(mag: F32, pad: int) -> Box:
    scaled = np.clip(mag / max(float(np.max(mag)), 1e-6) * 255, 0, 255).astype(np.uint8)
    _, binary = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = max(5, int(min(mag.shape) * 0.03) | 1)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if n <= 1:
        raise ValueError("no consistent watermark found across the images")
    strength = [float(mag[labels == i].sum()) for i in range(1, n)]
    best = 1 + int(np.argmax(strength))
    x, y, w, h = (int(v) for v in stats[best, :4])
    hh, ww = mag.shape
    return max(0, y - pad), min(hh, y + h + pad), max(0, x - pad), min(ww, x + w + pad)


def _align(gray: F32, template_mag: F32, box: Box, search: int) -> tuple[int, int]:
    y0, y1, x0, x1 = box
    h, w = gray.shape
    sy0, sy1 = max(0, y0 - search), min(h, y1 + search)
    sx0, sx1 = max(0, x0 - search), min(w, x1 + search)
    gx, gy = _fwd_grad(np.ascontiguousarray(gray[sy0:sy1, sx0:sx1]))
    mag = np.hypot(gx, gy).astype(np.float32)
    if mag.shape[0] < template_mag.shape[0] or mag.shape[1] < template_mag.shape[1]:
        return 0, 0
    score = cv2.matchTemplate(mag, template_mag, cv2.TM_CCOEFF_NORMED)
    _, _, _, loc = cv2.minMaxLoc(np.nan_to_num(score, nan=-1.0))
    return sy0 + int(loc[1]) - y0, sx0 + int(loc[0]) - x0


def _crop(img: FloatImage, box: Box, off: tuple[int, int]) -> FloatImage:
    y0, y1, x0, x1 = box
    dy, dx = off
    h, w = img.shape[:2]
    ys = np.clip(np.arange(y0 + dy, y1 + dy), 0, h - 1)
    xs = np.clip(np.arange(x0 + dx, x1 + dx), 0, w - 1)
    return img[np.ix_(ys, xs)]


def _edge_energy(
    crops: Sequence[FloatImage], rgb: FloatImage, alpha: F32, edges: npt.NDArray[np.bool_]
) -> float:
    total = 0.0
    for c in crops:
        j, _ = reverse(c, rgb, alpha, 0.95)
        gx, gy = _fwd_grad(luminance(j))
        total += float(np.mean((gx * gx + gy * gy)[edges]))
    return total / max(len(crops), 1)


def estimate(
    images: Sequence[Image],
    *,
    region: Box | None = None,
    progress: Progress = NULL_PROGRESS,
) -> Estimate:
    """Estimate the shared watermark (colour and opacity) from ``images``.

    All images must have the same size and carry the watermark at the same
    scale and roughly the same position. ``region`` restricts the search.
    """
    if len(images) < MIN_IMAGES:
        raise ValueError(f"need at least {MIN_IMAGES} images (10 or more work best)")
    shapes = {im.shape[:2] for im in images}
    if len(shapes) != 1:
        raise ValueError("all images must have the same width and height")
    rgbs = [_rgb(im) for im in images]
    oy, ox = 0, 0
    if region is not None:
        ry0, ry1, rx0, rx1 = region
        rgbs = [r[ry0:ry1, rx0:rx1] for r in rgbs]
        oy, ox = ry0, rx0
    grays = [luminance(r) for r in rgbs]
    progress.report(0.1, "finding the shared watermark")

    mgx, mgy = _median_gradients(grays)
    mag = np.hypot(mgx, mgy).astype(np.float32)
    box = _detect_box(mag, pad=6)
    y0, y1, x0, x1 = box
    shape_w = poisson_from_gradients(mgx[y0:y1, x0:x1], mgy[y0:y1, x0:x1])
    tgx, tgy = _fwd_grad(shape_w)
    tmag = np.hypot(tgx, tgy).astype(np.float32)
    progress.check()

    # align every image to the reconstructed shape
    search = max(8, (y1 - y0) // 4)
    offsets = [_align(g, tmag, box, search) for g in grays]
    crops = [_crop(r, box, o) for r, o in zip(rgbs, offsets, strict=True)]
    progress.report(0.35, "aligning images")

    # refined shape and support from aligned crops
    cgx, cgy = _median_gradients([luminance(c) for c in crops])
    shape_w = poisson_from_gradients(cgx, cgy)
    cmag = np.hypot(cgx, cgy)
    support = (np.abs(shape_w) > 0.08 * max(float(np.abs(shape_w).max()), 1e-6)) | (
        cmag > 0.25 * max(float(cmag.max()), 1e-6)
    )
    closed = cv2.morphologyEx(support.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    support = cv2.dilate(closed, np.ones((3, 3), np.uint8)).astype(bool)
    support[[0, -1], :] = False
    support[:, [0, -1]] = False
    if not support.any():
        raise ValueError("no consistent watermark found across the images")

    # background guesses: diffuse the surroundings into the watermark area
    backgrounds = []
    for i, c in enumerate(crops):
        progress.check()
        backgrounds.append(diffusion.fill(c, support, method="telea", radius=5.0))
        progress.report(0.35 + 0.4 * (i + 1) / len(crops), "estimating backgrounds")
    x = np.stack(backgrounds).astype(np.float64)  # K,h,w,3
    obs = np.stack(crops).astype(np.float64)
    y = obs - x

    # least squares with a shared α:  y = A − α x
    xm = x - x.mean(axis=0)
    ym = y - y.mean(axis=0)
    num = -(ym * xm).sum(axis=(0, 3))
    den = (xm * xm).sum(axis=(0, 3)) + 1e-9
    a = np.clip(num / den, 0.0, 0.99)
    a[~support] = 0.0
    big_a = y.mean(axis=0) + a[..., None] * x.mean(axis=0)
    w_rgb = np.where(a[..., None] > 0.05, big_a / np.maximum(a, 1e-6)[..., None], 0.0)
    w_rgb = np.clip(w_rgb, 0.0, 1.0)

    confident = a > 0.15
    single = True
    if confident.sum() >= 20:
        med = np.median(w_rgb[confident], axis=0)
        spread = float(np.median(np.abs(w_rgb[confident] - med)))
        single = spread < 0.06
    else:
        med = np.ones(3)
    if single:
        w_rgb = np.broadcast_to(med, w_rgb.shape).copy()
    else:
        w_rgb[~confident] = med

    # α again with the colour fixed:  y = α (W − x)
    d = w_rgb[None] - x
    a = np.clip((y * d).sum(axis=(0, 3)) / ((d * d).sum(axis=(0, 3)) + 1e-9), 0.0, 0.99)
    a[~support] = 0.0
    alpha = a.astype(np.float32)
    rgb = w_rgb.astype(np.float32)
    progress.report(0.85, "refining")

    # one global gain on α, chosen by remaining edge energy (golden section)
    edges = cv2.dilate(support.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    lo, hi = 0.6, 1.4
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    c1, c2 = hi - phi * (hi - lo), lo + phi * (hi - lo)

    def energy(g: float) -> float:
        return _edge_energy(crops, rgb, np.clip(alpha * g, 0, 0.99).astype(np.float32), edges)

    e1, e2 = energy(c1), energy(c2)
    for _ in range(14):
        if e1 < e2:
            hi, c2, e2 = c2, c1, e1
            c1 = hi - phi * (hi - lo)
            e1 = energy(c1)
        else:
            lo, c1, e1 = c1, c2, e2
            c2 = lo + phi * (hi - lo)
            e2 = energy(c2)
    gain = (lo + hi) / 2.0
    alpha = np.clip(alpha * gain, 0.0, 0.99).astype(np.float32)
    progress.report(1.0, "done")

    return Estimate(
        rgb=rgb,
        alpha=alpha,
        box=(y0 + oy, y1 + oy, x0 + ox, x1 + ox),
        offsets=offsets,
        single_color=single,
    )
