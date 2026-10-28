"""Find similar: locate other copies of a selected logo (tiled watermarks).

The selected area is used as a template and matched across the image at
several scales with normalised cross-correlation on gradient magnitude, which
is robust to semi-transparent marks over changing backgrounds. Peaks are
extracted greedily and overlapping hits are merged (non-maximum suppression).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from .._types import BoolMask, Image
from ..align import gradient_magnitude
from ..color import luminance, to_float

MAX_SIDE = 2400
DEFAULT_SCALES = (0.8, 0.9, 1.0, 1.1, 1.25)


@dataclass(frozen=True)
class Hit:
    x: int
    y: int
    w: int
    h: int
    score: float
    scale: float
    is_source: bool = False


def _iou(a: Hit, b: Hit) -> float:
    x0, y0 = max(a.x, b.x), max(a.y, b.y)
    x1, y1 = min(a.x + a.w, b.x + b.w), min(a.y + a.h, b.y + b.h)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union else 0.0


def find_similar(
    image: Image,
    box: tuple[int, int, int, int],
    *,
    scales: Sequence[float] = DEFAULT_SCALES,
    threshold: float = 0.6,
    max_results: int = 100,
) -> list[Hit]:
    """Find copies of the region ``box`` = (x, y, w, h). Best matches first.

    The source region itself is included, flagged with ``is_source``.
    """
    bx, by, bw, bh = box
    if bw < 4 or bh < 4:
        raise ValueError("select an area of at least 4x4 pixels")
    lum = luminance(to_float(image))
    h0, w0 = lum.shape
    f = min(1.0, MAX_SIDE / max(h0, w0))
    if f < 1.0:
        lum = np.asarray(
            cv2.resize(lum, (round(w0 * f), round(h0 * f)), interpolation=cv2.INTER_AREA),
            dtype=np.float32,
        )
    grad = gradient_magnitude(np.ascontiguousarray(lum))
    tx, ty = round(bx * f), round(by * f)
    tw, th = max(3, round(bw * f)), max(3, round(bh * f))
    template = grad[ty : ty + th, tx : tx + tw]
    if template.size == 0 or float(template.std()) == 0.0:
        raise ValueError("the selected area has no detail to match")

    hits: list[Hit] = []
    for s in scales:
        t = (
            template
            if s == 1.0
            else cv2.resize(template, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        )
        if t.shape[0] >= grad.shape[0] or t.shape[1] >= grad.shape[1] or min(t.shape) < 3:
            continue
        score: npt.NDArray[np.float32] = np.asarray(
            np.nan_to_num(cv2.matchTemplate(grad, t, cv2.TM_CCOEFF_NORMED), nan=-1.0),
            dtype=np.float32,
        )
        th_, tw_ = t.shape
        for _ in range(max_results):
            _, val, _, loc = cv2.minMaxLoc(score)
            if val < threshold:
                break
            px, py = loc
            hits.append(
                Hit(
                    round(px / f),
                    round(py / f),
                    round(tw_ / f),
                    round(th_ / f),
                    float(val),
                    float(s),
                )
            )
            score[
                max(0, py - th_ // 2) : py + th_ // 2 + 1, max(0, px - tw_ // 2) : px + tw_ // 2 + 1
            ] = -1.0

    hits.sort(key=lambda h: -h.score)
    kept: list[Hit] = []
    src = Hit(bx, by, bw, bh, 1.0, 1.0)
    for hit in hits:
        if all(_iou(hit, k) < 0.3 for k in kept):
            kept.append(
                Hit(
                    hit.x, hit.y, hit.w, hit.h, hit.score, hit.scale, is_source=_iou(hit, src) > 0.5
                )
            )
        if len(kept) >= max_results:
            break
    return kept


def hits_to_mask(
    shape: tuple[int, int],
    hits: Sequence[Hit],
    source_box: tuple[int, int, int, int],
    source_mask: BoolMask | None = None,
) -> BoolMask:
    """Paint every hit into a mask, reusing the source selection's shape if given."""
    out = np.zeros(shape, bool)
    bx, by, bw, bh = source_box
    tmpl = None
    if source_mask is not None:
        tmpl = source_mask[by : by + bh, bx : bx + bw].astype(np.uint8)
    for hit in hits:
        y1, x1 = min(shape[0], hit.y + hit.h), min(shape[1], hit.x + hit.w)
        if y1 <= hit.y or x1 <= hit.x:
            continue
        if tmpl is None or tmpl.size == 0:
            out[hit.y : y1, hit.x : x1] = True
            continue
        scaled = cv2.resize(tmpl, (hit.w, hit.h), interpolation=cv2.INTER_NEAREST).astype(bool)
        out[hit.y : y1, hit.x : x1] |= scaled[: y1 - hit.y, : x1 - hit.x]
    return out
