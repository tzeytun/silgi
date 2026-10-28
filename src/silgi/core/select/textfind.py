"""Classical text detection (no OCR, no learning).

Three cues are combined:

* **MSER glyphs** — maximally stable extremal regions that look like
  characters: sensible size and fill, and a roughly *constant stroke width*
  (measured with a distance transform along the region's ridge).
* **Top-hat glyphs** — white and black top-hat transforms at a few stroke
  scales separate thin bright (or dark) strokes from a varying background.
  This catches semi-transparent watermark text that MSER misses because the
  background changes under the glyph. The same glyph tests apply.
* **Morphological gradient** — text is dense in short, strong edges; lines
  with too little gradient energy are rejected.

Glyphs are linked into lines only when they agree in height, vertical
position, spacing and stroke width, which is what separates text from
texture. The user then accepts or rejects each box.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from .._types import Image
from ..color import luminance, to_float

MAX_SIDE = 2000
MIN_GLYPHS = 3


@dataclass(frozen=True)
class TextBox:
    x: int
    y: int
    w: int
    h: int
    score: float

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


@dataclass(frozen=True)
class _Glyph:
    x: int
    y: int
    w: int
    h: int
    stroke: float  # median stroke width in px


def _stroke_stats(glyph: npt.NDArray[np.uint8]) -> tuple[float, float]:
    """(median stroke width, coefficient of variation) of a binary glyph."""
    dist = cv2.distanceTransform(np.pad(glyph, 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
    ridge = (dist > 0) & (dist >= cv2.dilate(dist, np.ones((3, 3), np.uint8)) - 1e-3)
    vals = dist[ridge]
    if vals.size < 3:
        return 0.0, 1.0
    return 2.0 * float(np.median(vals)), float(np.std(vals)) / max(float(np.mean(vals)), 1e-6)


def _glyph(mask: npt.NDArray[np.uint8], x: int, y: int, img_h: int) -> _Glyph | None:
    bh, bw = mask.shape
    if bh < 6 or bw < 2 or bh > img_h * 0.5:
        return None
    if not 0.08 <= bw / bh <= 3.0:
        return None
    fill = float(mask.sum()) / float(bw * bh)
    if not 0.12 <= fill <= 0.85:
        return None
    stroke, cv = _stroke_stats(mask)
    if cv > 0.65 or stroke <= 0 or stroke > 0.6 * bh:
        return None
    return _Glyph(x, y, bw, bh, stroke)


def _mser_glyphs(gray: npt.NDArray[np.uint8]) -> list[_Glyph]:
    h, w = gray.shape
    area = h * w
    # min_diversity=0: OpenCV 5's default prunes away nearly every region;
    # the nested near-duplicates this keeps are merged by _dedupe.
    mser = cv2.MSER.create(
        delta=4,
        min_area=max(12, area // 200000),
        max_area=max(400, area // 50),
        max_variation=0.5,
        min_diversity=0.0,
    )
    regions, boxes = mser.detectRegions(gray)
    out: list[_Glyph] = []
    for region, (x, y, bw, bh) in zip(regions, boxes, strict=True):
        if bh < 6 or bw < 2 or bh > h * 0.5:
            continue
        pts = np.asarray(region).reshape(-1, 2)
        m = np.zeros((int(bh), int(bw)), np.uint8)
        m[pts[:, 1] - y, pts[:, 0] - x] = 1
        g = _glyph(m, int(x), int(y), h)
        if g is not None:
            out.append(g)
    return out


def _tophat_glyphs(gray: npt.NDArray[np.uint8]) -> list[_Glyph]:
    """Top-hat glyphs on a small pyramid (x1, x1/2, x1/4) so large text is found too."""
    out: list[_Glyph] = []
    for f in (1, 2, 4):
        g8 = (
            gray
            if f == 1
            else cv2.resize(
                gray, (gray.shape[1] // f, gray.shape[0] // f), interpolation=cv2.INTER_AREA
            )
        )
        h, w = g8.shape
        for k in (15, 31):
            if k * 2 > min(h, w):
                break
            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            for op in (cv2.MORPH_TOPHAT, cv2.MORPH_BLACKHAT):
                th = cv2.morphologyEx(g8, op, se)
                level, _ = cv2.threshold(th, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                binary = (th > max(level, 14)).astype(np.uint8)
                n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
                for i in range(1, n):
                    x, y, bw, bh, area = (int(v) for v in stats[i])
                    if area < 12:
                        continue
                    mask = (labels[y : y + bh, x : x + bw] == i).astype(np.uint8)
                    g = _glyph(mask, x, y, h)
                    if g is not None and g.stroke < k * 0.6:
                        out.append(_Glyph(x * f, y * f, bw * f, bh * f, g.stroke * f))
    return out


def _dedupe(glyphs: list[_Glyph]) -> list[_Glyph]:
    """Drop glyphs mostly contained in a larger kept glyph (nested detections)."""
    kept: list[_Glyph] = []
    for g in sorted(set(glyphs), key=lambda g: (-(g.w * g.h), g.y, g.x)):
        dup = False
        for k in kept:
            ix = max(0, min(g.x + g.w, k.x + k.w) - max(g.x, k.x))
            iy = max(0, min(g.y + g.h, k.y + k.h) - max(g.y, k.y))
            if ix * iy >= 0.7 * g.w * g.h:
                dup = True
                break
        if not dup:
            kept.append(g)
    return kept


def _compatible(a: _Glyph, b: _Glyph) -> bool:
    """Could ``a`` and ``b`` be neighbouring characters on one text line?"""
    hmin, hmax = min(a.h, b.h), max(a.h, b.h)
    if hmax > 1.8 * hmin:
        return False
    overlap = min(a.y + a.h, b.y + b.h) - max(a.y, b.y)
    if overlap < 0.5 * hmin:
        return False
    gap = max(a.x, b.x) - min(a.x + a.w, b.x + b.w)
    if gap > 1.0 * hmax:
        return False
    s0, s1 = min(a.stroke, b.stroke), max(a.stroke, b.stroke)
    return s1 <= 2.2 * s0


def _lines(glyphs: list[_Glyph]) -> list[list[_Glyph]]:
    parent = list(range(len(glyphs)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    order = sorted(range(len(glyphs)), key=lambda i: (glyphs[i].x, glyphs[i].y))
    for oi, i in enumerate(order):
        gi = glyphs[i]
        for j in order[oi + 1 :]:
            gj = glyphs[j]
            if gj.x > gi.x + gi.w + 1.8 * max(gi.h, gj.h):
                break
            if _compatible(gi, gj):
                parent[find(i)] = find(j)
    groups: dict[int, list[_Glyph]] = {}
    for i, g in enumerate(glyphs):
        groups.setdefault(find(i), []).append(g)
    return list(groups.values())


def _robust_cv(values: npt.NDArray[np.float64]) -> float:
    """Spread relative to the median (MAD-based, robust to fragments)."""
    med = float(np.median(values))
    if med <= 0:
        return 1.0
    return float(1.4826 * np.median(np.abs(values - med)) / med)


def _gradient_integral(gray: npt.NDArray[np.uint8]) -> npt.NDArray[np.int32]:
    """Integral image of strong morphological-gradient pixels."""
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, se)
    _, binary = cv2.threshold(grad, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return np.asarray(cv2.integral(binary.astype(np.uint8)), dtype=np.int32)


def find_text(image: Image, *, max_results: int = 50, min_score: float = 0.3) -> list[TextBox]:
    """Return candidate text boxes (x, y, w, h in image pixels), best first."""
    lum = luminance(to_float(image))
    h0, w0 = lum.shape
    scale = min(1.0, MAX_SIDE / max(h0, w0))
    if scale < 1.0:
        lum = np.asarray(
            cv2.resize(lum, (round(w0 * scale), round(h0 * scale)), interpolation=cv2.INTER_AREA),
            dtype=np.float32,
        )
    gray = np.clip(np.rint(lum * 255), 0, 255).astype(np.uint8)
    h, w = gray.shape

    glyphs = _dedupe(_mser_glyphs(gray) + _tophat_glyphs(gray))
    integral = _gradient_integral(gray)
    results: list[TextBox] = []
    for line in _lines(glyphs):
        if len(line) < MIN_GLYPHS:
            continue
        heights = np.array([g.h for g in line], np.float64)
        widths = np.array([g.w for g in line], np.float64)
        strokes = np.array([g.stroke for g in line], np.float64)
        h_cv, s_cv = _robust_cv(heights), _robust_cv(strokes)
        if h_cv > 0.5 or s_cv > 0.65:
            continue
        if len(line) >= 5 and _robust_cv(widths) < 0.05 and h_cv < 0.05:
            continue  # identical repeated cells: a pattern, not text
        x0, y0 = min(g.x for g in line), min(g.y for g in line)
        x1, y1 = max(g.x + g.w for g in line), max(g.y + g.h for g in line)
        bw, bh = x1 - x0, y1 - y0
        if bw < 1.2 * bh:
            continue
        edges = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
        # edge pixels per unit area scale like 1/stroke width; normalise so
        # large and small text are judged alike
        density = float(edges) / float(bw * bh) * float(np.median(strokes))
        if density < 0.3:
            continue
        regularity = 1.0 - min(1.0, (h_cv + s_cv) / 1.15)
        score = 0.45 * min(len(line) / 6.0, 1.0) + 0.35 * regularity + 0.2 * min(density / 1.0, 1.0)
        if score < min_score:
            continue
        pad = max(2, int(bh * 0.15))
        px0, py0 = max(0, x0 - pad), max(0, y0 - pad)
        px1, py1 = min(w, x1 + pad), min(h, y1 + pad)
        results.append(
            TextBox(
                round(px0 / scale),
                round(py0 / scale),
                round((px1 - px0) / scale),
                round((py1 - py0) / scale),
                float(score),
            )
        )
    results.sort(key=lambda b: (-b.score, b.y, b.x))
    return results[:max_results]


def boxes_to_mask(shape: tuple[int, int], boxes: list[TextBox]) -> npt.NDArray[np.bool_]:
    m = np.zeros(shape, bool)
    for b in boxes:
        m[b.y : b.y + b.h, b.x : b.x + b.w] = True
    return m
