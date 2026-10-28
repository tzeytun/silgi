"""Rasterising geometric selections: rectangle, ellipse, (polygon) lasso, brush."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
import numpy.typing as npt

from .._types import BoolMask

Point = tuple[float, float]  # (x, y) in image pixels

_SHIFT = 4  # OpenCV fixed-point bits for sub-pixel accuracy
_ONE = 1 << _SHIFT


def _fixed(points: Sequence[Point]) -> npt.NDArray[np.int32]:
    return np.array([[round(x * _ONE), round(y * _ONE)] for x, y in points], np.int32)


def rectangle(shape: tuple[int, int], x0: float, y0: float, x1: float, y1: float) -> BoolMask:
    """Pixels whose centres lie inside the rectangle (corners in any order)."""
    h, w = shape
    xa, xb = sorted((x0, x1))
    ya, yb = sorted((y0, y1))
    m = np.zeros((h, w), bool)
    c0, c1 = max(0, int(np.ceil(xa - 0.5))), min(w, int(np.ceil(xb - 0.5)))
    r0, r1 = max(0, int(np.ceil(ya - 0.5))), min(h, int(np.ceil(yb - 0.5)))
    if c1 > c0 and r1 > r0:
        m[r0:r1, c0:c1] = True
    return m


def ellipse(shape: tuple[int, int], x0: float, y0: float, x1: float, y1: float) -> BoolMask:
    """Ellipse inscribed in the given box."""
    h, w = shape
    cx, cy = (x0 + x1) / 2.0 - 0.5, (y0 + y1) / 2.0 - 0.5
    rx, ry = abs(x1 - x0) / 2.0, abs(y1 - y0) / 2.0
    if rx <= 0 or ry <= 0:
        return np.zeros((h, w), bool)
    yy, xx = np.ogrid[0:h, 0:w]
    return np.asarray(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0, dtype=bool)


def polygon(shape: tuple[int, int], points: Sequence[Point]) -> BoolMask:
    """Filled polygon (lasso and polygon lasso). Self-intersections use even-odd fill."""
    m = np.zeros(shape, np.uint8)
    if len(points) >= 3:
        pts = _fixed([(x - 0.5, y - 0.5) for x, y in points])
        cv2.fillPoly(m, [pts], 1, lineType=cv2.LINE_8, shift=_SHIFT)
    return m.astype(bool)


def brush_stroke(
    shape: tuple[int, int],
    points: Sequence[Point],
    size: float,
    hardness: float = 1.0,
) -> npt.NDArray[np.float32]:
    """A round brush dragged through ``points``.

    Returns weights in [0, 1]: 1 in the core, fading out over the soft part of
    the brush when ``hardness`` < 1.
    """
    h, w = shape
    radius = max(size / 2.0, 0.5)
    hard_r = radius * float(np.clip(hardness, 0.0, 1.0))
    m = np.zeros((h, w), np.uint8)
    if not points:
        return m.astype(np.float32)
    pts = _fixed([(x - 0.5, y - 0.5) for x, y in points])
    r_fixed = max(1, round(radius * _ONE))
    if len(pts) == 1:
        cv2.circle(m, tuple(int(v) for v in pts[0]), r_fixed, 1, -1, cv2.LINE_8, _SHIFT)
    else:
        cv2.polylines(m, [pts], False, 1, max(1, round(2 * radius)), cv2.LINE_8, _SHIFT)
        for p in (pts[0], pts[-1]):
            cv2.circle(m, tuple(int(v) for v in p), r_fixed, 1, -1, cv2.LINE_8, _SHIFT)
    if hard_r >= radius - 1e-6:
        return m.astype(np.float32)
    # distance from the stroke's skeleton gives a soft falloff
    core = np.ones((h, w), np.uint8)
    if len(pts) == 1:
        cv2.circle(core, tuple(int(v) for v in pts[0]), 0, 0, -1, cv2.LINE_8, _SHIFT)
    else:
        cv2.polylines(core, [pts], False, 0, 1, cv2.LINE_8, _SHIFT)
    dist = cv2.distanceTransform(core, cv2.DIST_L2, 5)
    soft = np.clip((radius - dist) / max(radius - hard_r, 1e-6), 0.0, 1.0)
    return np.asarray(soft * m, dtype=np.float32)
