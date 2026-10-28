"""Locating a known watermark in an image.

Semi-transparent watermarks barely change colours, but they do create edges
exactly where their alpha changes. We therefore match the image's gradient
magnitude against the watermark alpha's gradient magnitude using normalised
cross-correlation, then refine the peak to sub-pixel precision with a
parabola fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from ._types import Box, FloatImage, Image
from .color import luminance, to_float


@dataclass(frozen=True)
class Match:
    top: float
    left: float
    score: float
    scale: float = 1.0


def gradient_magnitude(gray: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.asarray(cv2.magnitude(gx, gy), dtype=np.float32)


def _parabola_offset(m1: float, m0: float, p1: float) -> float:
    den = m1 - 2.0 * m0 + p1
    if abs(den) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (m1 - p1) / den, -0.5, 0.5))


def subpixel_peak(score: npt.NDArray[np.float32], y: int, x: int) -> tuple[float, float]:
    """Refine an integer peak location with a separable parabola fit."""
    h, w = score.shape
    dy = dx = 0.0
    if 0 < y < h - 1:
        dy = _parabola_offset(float(score[y - 1, x]), float(score[y, x]), float(score[y + 1, x]))
    if 0 < x < w - 1:
        dx = _parabola_offset(float(score[y, x - 1]), float(score[y, x]), float(score[y, x + 1]))
    return y + dy, x + dx


def match_gradients(
    image_grad: npt.NDArray[np.float32], template_grad: npt.NDArray[np.float32]
) -> tuple[npt.NDArray[np.float32], int, int, float]:
    """Run NCC; returns the score map and its integer maximum."""
    raw = cv2.matchTemplate(image_grad, template_grad, cv2.TM_CCOEFF_NORMED)
    score = np.asarray(np.nan_to_num(raw, nan=-1.0), dtype=np.float32)
    _, best, _, loc = cv2.minMaxLoc(score)
    return score, int(loc[1]), int(loc[0]), float(best)


def locate(
    image: Image,
    alpha: npt.NDArray[np.float32],
    *,
    region: Box | None = None,
    scales: tuple[float, ...] = (1.0,),
) -> Match:
    """Find where a watermark with coverage ``alpha`` sits in ``image``.

    ``region`` limits the search to a box in image coordinates. With several
    ``scales`` the template is resized and the best-scoring scale wins.
    """
    img = to_float(image)
    gray = luminance(img)
    y0, x0 = 0, 0
    if region is not None:
        y0, y1, x0, x1 = region
        gray = gray[y0:y1, x0:x1]
    grad = gradient_magnitude(np.ascontiguousarray(gray))
    best: Match | None = None
    for s in scales:
        a = alpha if s == 1.0 else cv2.resize(alpha, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if a.shape[0] > grad.shape[0] or a.shape[1] > grad.shape[1] or min(a.shape) < 3:
            continue
        tgrad = gradient_magnitude(np.ascontiguousarray(a.astype(np.float32)))
        if float(tgrad.std()) == 0.0:
            continue
        score, py, px, val = match_gradients(grad, tgrad)
        fy, fx = subpixel_peak(score, py, px)
        if best is None or val > best.score:
            best = Match(top=y0 + fy, left=x0 + fx, score=val, scale=s)
    if best is None:
        raise ValueError("watermark is larger than the search region or has no edges")
    return best


def shift_watermark(
    rgb: FloatImage, alpha: npt.NDArray[np.float32], dy: float, dx: float
) -> tuple[FloatImage, npt.NDArray[np.float32]]:
    """Shift a watermark by a sub-pixel offset (premultiplied, bilinear).

    The output is one pixel larger on each side to hold the shifted edge.
    """
    h, w = alpha.shape
    pad = 1
    m = np.array([[1, 0, dx + pad], [0, 1, dy + pad]], dtype=np.float32)
    size = (w + 2 * pad, h + 2 * pad)
    pre = rgb * alpha[..., None]
    a2 = cv2.warpAffine(alpha, m, size, flags=cv2.INTER_LINEAR, borderValue=0)
    p2 = cv2.warpAffine(pre, m, size, flags=cv2.INTER_LINEAR, borderValue=0)
    if p2.ndim == 2:
        p2 = p2[..., None]
    rgb2 = np.where(a2[..., None] > 1e-6, p2 / np.maximum(a2, 1e-6)[..., None], 0.0)
    return np.clip(rgb2, 0, 1).astype(np.float32), np.clip(a2, 0, 1).astype(np.float32)
