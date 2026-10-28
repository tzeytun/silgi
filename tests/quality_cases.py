"""Synthetic inpainting cases with known ground truth.

Shared by the quality-regression test and ``benchmarks/bench_quality.py``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np

from silgi.core import engine
from silgi.core.metrics import psnr, ssim
from silgi.samples import SCENES, scene, to_uint8

SIZE = 192
METHODS = ("telea", "ns", "patchmatch")


def masks() -> dict[str, np.ndarray]:
    blob = np.zeros((SIZE, SIZE), np.uint8)
    cv2.ellipse(blob, (SIZE // 2, SIZE // 2), (40, 22), 15, 0, 360, 1, -1)
    strokes = np.zeros((SIZE, SIZE), np.uint8)
    for i in range(4):
        y = 50 + i * 28
        cv2.line(strokes, (30, y), (SIZE - 30, y + 10), 1, 3)
    return {"blob": blob.astype(bool), "strokes": strokes.astype(bool)}


@dataclass(frozen=True)
class Score:
    psnr: float
    ssim: float


def cases() -> Iterator[tuple[str, str, str]]:
    for name in SCENES:
        for mask_name in ("blob", "strokes"):
            for method in METHODS:
                yield name, mask_name, method


def evaluate(scene_name: str, mask_name: str, method: str) -> Score:
    gt = to_uint8(scene(scene_name, SIZE, SIZE, seed=11))
    mask = masks()[mask_name]
    damaged = gt.copy()
    damaged[mask] = (255, 255, 255)
    out = engine.remove(damaged, mask, method).image  # type: ignore[arg-type]
    ys, xs = np.nonzero(mask)
    sl = (slice(max(ys.min() - 8, 0), ys.max() + 9), slice(max(xs.min() - 8, 0), xs.max() + 9))
    p = psnr(out, gt, mask)
    return Score(psnr=min(p, 99.0), ssim=ssim(out[sl], gt[sl]))
