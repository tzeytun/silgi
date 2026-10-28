"""Image quality metrics used by tests and benchmarks (PSNR, SSIM)."""

from __future__ import annotations

import cv2
import numpy as np

from ._types import Image
from .color import to_float


def psnr(a: Image, b: Image, mask: Image | None = None) -> float:
    """Peak signal-to-noise ratio in dB, computed on [0, 1]-normalised values.

    With ``mask`` only the selected pixels count. Identical inputs give ``inf``.
    """
    fa, fb = to_float(a), to_float(b)
    if fa.shape != fb.shape:
        raise ValueError(f"shape mismatch: {fa.shape} vs {fb.shape}")
    diff = (fa.astype(np.float64) - fb.astype(np.float64)) ** 2
    if mask is not None:
        sel = np.asarray(mask) > 0
        if not sel.any():
            return float("inf")
        diff = diff[sel]
    mse = float(diff.mean())
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def ssim(a: Image, b: Image) -> float:
    """Mean structural similarity (Wang et al. 2004), Gaussian window σ=1.5.

    Channels are averaged. Values are computed on [0, 1]-normalised data.
    """
    fa, fb = to_float(a).astype(np.float64), to_float(b).astype(np.float64)
    if fa.shape != fb.shape:
        raise ValueError(f"shape mismatch: {fa.shape} vs {fb.shape}")
    c1, c2 = (0.01) ** 2, (0.03) ** 2
    scores = []
    for ch in range(fa.shape[2]):
        x, y = fa[:, :, ch], fb[:, :, ch]

        def blur(z: np.ndarray) -> np.ndarray:
            return cv2.GaussianBlur(z, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT)

        mx, my = blur(x), blur(y)
        sxx = blur(x * x) - mx * mx
        syy = blur(y * y) - my * my
        sxy = blur(x * y) - mx * my
        num = (2 * mx * my + c1) * (2 * sxy + c2)
        den = (mx * mx + my * my + c1) * (sxx + syy + c2)
        scores.append(float(np.mean(num / den)))
    return float(np.mean(scores))
