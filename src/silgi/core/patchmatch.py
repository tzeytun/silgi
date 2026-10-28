"""Exemplar-based image completion with multiscale PatchMatch.

The hole is filled by copying patches from the rest of the region of interest:

1. Build a coarse-to-fine pyramid. At the coarsest level the hole is only a few
   patches wide and is initialised by diffusion.
2. At each level alternate (Wexler et al. 2007):
   * **search** — for every patch overlapping the hole find the most similar
     fully-known patch with PatchMatch (Barnes et al. 2009): propagation of good
     matches from neighbours plus random search at shrinking radii;
   * **vote** — every hole pixel becomes the weighted average of the pixels
     that the overlapping matched patches put there.
3. Upsample the nearest-neighbour field and the result to the next level.

Determinism: the search runs as red/black half-sweeps, so each pixel only reads
neighbours that are not being written in the same sweep, and all random
numbers come from a counter-based hash of (seed, level, iteration, pixel). The
output is therefore identical regardless of the number of threads.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import cv2
import numpy as np
import numpy.typing as npt
from numba import njit, prange

from . import diffusion
from ._types import BoolMask, Box, FloatImage, Image, MaskLike
from .progress import NULL_PROGRESS, Progress
from .roi import mask_bbox, normalize_mask, run_in_roi

Quality = Literal["fast", "balanced", "best"]


class NoSourceError(ValueError):
    """There are no fully-known patches to copy from."""


@dataclass(frozen=True)
class Params:
    patch_size: int = 7
    em_iters_coarse: int = 10
    em_iters_fine: int = 2
    pm_iters: int = 4
    seed: int = 0


PRESETS: dict[str, Params] = {
    "fast": Params(patch_size=5, em_iters_coarse=6, em_iters_fine=1, pm_iters=2),
    "balanced": Params(patch_size=7, em_iters_coarse=10, em_iters_fine=2, pm_iters=4),
    "best": Params(patch_size=9, em_iters_coarse=16, em_iters_fine=4, pm_iters=6),
}

_K1 = np.uint64(0x9E3779B97F4A7C15)
_K2 = np.uint64(0xC2B2AE3D27D4EB4F)
_K3 = np.uint64(0x165667B19E3779F9)


# ----------------------------------------------------------------- numba kernels


@njit(cache=True, nogil=True, inline="always")
def _mix(z: np.uint64) -> np.uint64:  # pragma: no cover - compiled
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


@njit(cache=True, nogil=True, inline="always")
def _ssd(img, py, px, qy, qx, r, cutoff):  # type: ignore[no-untyped-def]  # pragma: no cover
    c_n = img.shape[2]
    s = 0.0
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            for c in range(c_n):
                d = img[py + dy, px + dx, c] - img[qy + dy, qx + dx, c]
                s += d * d
        if s >= cutoff:
            return s
    return s


@njit(parallel=True, cache=True, nogil=True)
def _compute_dist(img, ay, ax, nnf_y, nnf_x, dist, r):  # type: ignore[no-untyped-def]  # pragma: no cover
    for i in prange(ay.size):
        y = ay[i]
        x = ax[i]
        dist[y, x] = _ssd(img, y, x, nnf_y[y, x], nnf_x[y, x], r, np.inf)


@njit(parallel=True, cache=True, nogil=True)
def _random_init(ay, ax, vy, vx, nnf_y, nnf_x, seed):  # type: ignore[no-untyped-def]  # pragma: no cover
    nv = np.uint64(vy.size)
    s = np.uint64(seed)
    for i in prange(ay.size):
        h = _mix(s * _K1 + np.uint64(i) * _K2 + _K3)
        j = np.int64(h % nv)
        nnf_y[ay[i], ax[i]] = vy[j]
        nnf_x[ay[i], ax[i]] = vx[j]


@njit(parallel=True, cache=True, nogil=True)
def _upsample_nnf(ay, ax, cy_nnf, cx_nnf, valid, vy, vx, nnf_y, nnf_x, seed):  # type: ignore[no-untyped-def]  # pragma: no cover
    hc, wc = cy_nnf.shape
    h, w = valid.shape
    nv = np.uint64(vy.size)
    s = np.uint64(seed)
    for i in prange(ay.size):
        y = ay[i]
        x = ax[i]
        found = False
        by = 0
        bx = 0
        for k in range(9):
            oy = k // 3 - 1
            ox = k % 3 - 1
            cy = min(max(y // 2 + oy, 0), hc - 1)
            cx = min(max(x // 2 + ox, 0), wc - 1)
            sy = cy_nnf[cy, cx]
            if sy < 0:
                continue
            fy = 2 * sy + (y - 2 * cy)
            fx = 2 * cx_nnf[cy, cx] + (x - 2 * cx)
            if fy >= 0 and fy < h and fx >= 0 and fx < w and valid[fy, fx]:
                by = fy
                bx = fx
                found = True
                break
        if not found:
            hsh = _mix(s * _K1 + np.uint64(i) * _K2 + _K3)
            j = np.int64(hsh % nv)
            by = vy[j]
            bx = vx[j]
        nnf_y[y, x] = by
        nnf_x[y, x] = bx


@njit(parallel=True, cache=True, nogil=True)
def _sweep(img, valid, ay, ax, nnf_y, nnf_x, dist, r, seed, max_radius):  # type: ignore[no-untyped-def]  # pragma: no cover
    """One PatchMatch half-sweep over pixels of a single parity."""
    h, w = valid.shape
    s = np.uint64(seed)
    for i in prange(ay.size):
        y = ay[i]
        x = ax[i]
        by = nnf_y[y, x]
        bx = nnf_x[y, x]
        bd = dist[y, x]
        # propagation: neighbours' matches shifted by the same offset
        for k in range(4):
            ny = y
            nx = x
            if k == 0:
                ny = y - 1
            elif k == 1:
                ny = y + 1
            elif k == 2:
                nx = x - 1
            else:
                nx = x + 1
            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                continue
            sy = nnf_y[ny, nx]
            if sy < 0:
                continue
            cy = sy + (y - ny)
            cx = nnf_x[ny, nx] + (x - nx)
            if cy < 0 or cy >= h or cx < 0 or cx >= w or not valid[cy, cx]:
                continue
            if cy == by and cx == bx:
                continue
            d = _ssd(img, y, x, cy, cx, r, bd)
            if d < bd:
                by = cy
                bx = cx
                bd = d
        # random search around the current best at exponentially shrinking radii
        hsh = _mix(s ^ (np.uint64(y) * _K1 + np.uint64(x) * _K2))
        rad = max_radius
        while rad >= 1:
            span = np.uint64(2 * rad + 1)
            hsh = _mix(hsh + _K3)
            cy = by + np.int64(hsh % span) - rad
            hsh = _mix(hsh + _K3)
            cx = bx + np.int64(hsh % span) - rad
            if cy >= 0 and cy < h and cx >= 0 and cx < w and valid[cy, cx]:
                d = _ssd(img, y, x, cy, cx, r, bd)
                if d < bd:
                    by = cy
                    bx = cx
                    bd = d
            rad = rad // 2
        nnf_y[y, x] = by
        nnf_x[y, x] = bx
        dist[y, x] = bd


@njit(parallel=True, cache=True, nogil=True)
def _vote(img, hy, hx, nnf_y, nnf_x, weight, r, out):  # type: ignore[no-untyped-def]  # pragma: no cover
    """Each hole pixel becomes the weighted mean of what overlapping patches propose."""
    h, w, c_n = img.shape
    for i in prange(hy.size):
        y = hy[i]
        x = hx[i]
        a0 = 0.0
        a1 = 0.0
        a2 = 0.0
        a3 = 0.0
        ws = 0.0
        for dy in range(-r, r + 1):
            py = y - dy
            if py < r or py >= h - r:
                continue
            for dx in range(-r, r + 1):
                px = x - dx
                if px < r or px >= w - r:
                    continue
                sy = nnf_y[py, px]
                if sy < 0:
                    continue
                sx = nnf_x[py, px]
                wt = weight[py, px]
                ty = sy + dy
                tx = sx + dx
                a0 += wt * img[ty, tx, 0]
                if c_n > 1:
                    a1 += wt * img[ty, tx, 1]
                if c_n > 2:
                    a2 += wt * img[ty, tx, 2]
                if c_n > 3:
                    a3 += wt * img[ty, tx, 3]
                ws += wt
        if ws > 0.0:
            out[y, x, 0] = a0 / ws
            if c_n > 1:
                out[y, x, 1] = a1 / ws
            if c_n > 2:
                out[y, x, 2] = a2 / ws
            if c_n > 3:
                out[y, x, 3] = a3 / ws


# ------------------------------------------------------------------- pyramid


@dataclass
class _Level:
    img: FloatImage
    hole: BoolMask
    src: BoolMask  # where source patch centres are allowed (user restriction)


def _down2(level: _Level) -> _Level:
    """Halve resolution. Coarse pixels are known only if all four children are."""
    img, hole, src = level.img, level.hole, level.src
    h, w = hole.shape
    hh, ww = (h + 1) // 2, (w + 1) // 2
    pad = ((0, 2 * hh - h), (0, 2 * ww - w))
    known = np.pad((~hole).astype(np.float32), pad, mode="edge")
    im = np.pad(img, (*pad, (0, 0)), mode="edge")
    srcp = np.pad(src, pad, mode="edge")
    c = img.shape[2]
    kb = known.reshape(hh, 2, ww, 2).sum(axis=(1, 3))
    vb = (im * known[..., None]).reshape(hh, 2, ww, 2, c).sum(axis=(1, 3))
    out = np.where(kb[..., None] > 0, vb / np.maximum(kb, 1)[..., None], 0.0).astype(np.float32)
    return _Level(out, kb < 4, srcp.reshape(hh, 2, ww, 2).any(axis=(1, 3)))


def _valid_sources(level: _Level, r: int) -> BoolMask:
    h, w = level.hole.shape
    k = 2 * r + 1
    known = (~level.hole).astype(np.uint8)
    full = cv2.erode(
        known, np.ones((k, k), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0
    )
    valid = full.astype(bool) & level.src
    valid[:r] = False
    valid[h - r :] = False
    valid[:, :r] = False
    valid[:, w - r :] = False
    return valid


def _active_centres(level: _Level, r: int) -> BoolMask:
    h, w = level.hole.shape
    k = 2 * r + 1
    near = cv2.dilate(level.hole.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    active = np.zeros_like(near)
    active[r : h - r, r : w - r] = near[r : h - r, r : w - r]
    return active


def _hole_thickness(hole: BoolMask) -> float:
    d = cv2.distanceTransform(np.pad(hole, 1).astype(np.uint8), cv2.DIST_L2, 5)
    return 2.0 * float(np.max(d))


def _build_pyramid(base: _Level, r: int) -> list[_Level]:
    k = 2 * r + 1
    levels = [base]
    thickness = _hole_thickness(base.hole)
    while thickness / (2 ** (len(levels) - 1)) > 2 * k and len(levels) < 10:
        nxt = _down2(levels[-1])
        if min(nxt.hole.shape) < 2 * k + 2 or not _valid_sources(nxt, r).any():
            break
        levels.append(nxt)
    return levels


# ------------------------------------------------------------------- driver


def _coords(
    mask: BoolMask, parity: int | None = None
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    ys, xs = np.nonzero(mask)
    if parity is not None:
        sel = ((ys + xs) & 1) == parity
        ys, xs = ys[sel], xs[sel]
    return ys.astype(np.int64), xs.astype(np.int64)


def _weights(dist: npt.NDArray[np.float32], active: BoolMask) -> npt.NDArray[np.float32]:
    d = dist[active]
    sigma2 = float(np.percentile(d, 75)) if d.size else 1.0
    sigma2 = max(sigma2, 1e-6)
    w = np.zeros_like(dist)
    w[active] = np.exp(-d / (2.0 * sigma2))
    w[active] = np.maximum(w[active], 1e-12)
    return w


def fill(
    roi: FloatImage,
    hole: BoolMask,
    *,
    params: Params = PRESETS["balanced"],
    source: BoolMask | None = None,
    progress: Progress = NULL_PROGRESS,
) -> FloatImage:
    """Fill ``hole`` in ``roi`` (float32 HxWxC) by multiscale PatchMatch.

    ``source`` optionally restricts where patches may be copied from.
    """
    if params.patch_size < 3 or params.patch_size % 2 == 0:
        raise ValueError("patch_size must be an odd number >= 3")
    r = params.patch_size // 2
    if min(hole.shape) < params.patch_size:
        raise NoSourceError("region is smaller than one patch")
    src = np.ones_like(hole) if source is None else source.copy()
    img0 = np.ascontiguousarray(roi.astype(np.float32))
    base = _Level(img0.copy(), hole.copy(), src)
    if not _valid_sources(base, r).any():
        raise NoSourceError("no fully known patches to copy from; enlarge the context or source")
    levels = _build_pyramid(base, r)
    n_levels = len(levels)
    weights_total = sum(lv.hole.sum() for lv in levels) or 1
    done = 0.0

    # coarsest initialisation by diffusion
    top = levels[-1]
    img = diffusion.fill(top.img, top.hole, method="telea", radius=3.0)
    nnf_y = np.full(top.hole.shape, -1, np.int32)
    nnf_x = np.full(top.hole.shape, -1, np.int32)

    for li in range(n_levels - 1, -1, -1):
        lv = levels[li]
        h, w = lv.hole.shape
        valid = _valid_sources(lv, r)
        vy, vx = _coords(valid)
        active = _active_centres(lv, r)
        ay, ax = _coords(active)
        ay0, ax0 = _coords(active, 0)
        ay1, ax1 = _coords(active, 1)
        hy, hx = _coords(lv.hole)
        seed = params.seed * 1000003 + li * 7919

        if li == n_levels - 1:
            new_y = np.full((h, w), -1, np.int32)
            new_x = np.full((h, w), -1, np.int32)
            _random_init(ay, ax, vy.astype(np.int32), vx.astype(np.int32), new_y, new_x, seed)
        else:
            # carry the coarse solution up: image by interpolation, NNF by doubling
            up = cv2.resize(
                img, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_LINEAR
            )
            if up.ndim == 2:
                up = up[..., None]
            fine = lv.img.copy()
            fine[lv.hole] = up[:h, :w][lv.hole]
            img = fine
            new_y = np.full((h, w), -1, np.int32)
            new_x = np.full((h, w), -1, np.int32)
            _upsample_nnf(
                ay,
                ax,
                nnf_y,
                nnf_x,
                valid,
                vy.astype(np.int32),
                vx.astype(np.int32),
                new_y,
                new_x,
                seed,
            )
        nnf_y, nnf_x = new_y, new_x
        img = np.ascontiguousarray(img, dtype=np.float32)
        dist = np.full((h, w), np.inf, np.float32)

        n_em = params.em_iters_coarse if li == n_levels - 1 else params.em_iters_fine
        if li > 0 and li < n_levels - 1:
            n_em = max(params.em_iters_fine, (params.em_iters_coarse + params.em_iters_fine) // 2)
        max_radius = max(h, w)
        for em in range(n_em):
            progress.check()
            _compute_dist(img, ay, ax, nnf_y, nnf_x, dist, r)
            for it in range(params.pm_iters):
                s = seed + em * 131 + it * 17
                _sweep(img, valid, ay0, ax0, nnf_y, nnf_x, dist, r, s * 2, max_radius)
                _sweep(img, valid, ay1, ax1, nnf_y, nnf_x, dist, r, s * 2 + 1, max_radius)
            wts = _weights(dist, active)
            out = img.copy()
            _vote(img, hy, hx, nnf_y, nnf_x, wts, r, out)
            img = out
            frac = done + (em + 1) / n_em * lv.hole.sum()
            progress.report(float(frac / weights_total), f"level {n_levels - li}/{n_levels}")
        done += float(lv.hole.sum())
        if li > 0 and progress.wants_preview:
            progress.preview(img.copy())

    result = img0.copy()
    result[hole] = img[hole]
    return result


def default_margin(mask: BoolMask, patch_size: int) -> int:
    box = mask_bbox(mask)
    if box is None:
        return 0
    size = max(box[1] - box[0], box[3] - box[2])
    return int(np.clip(size, 4 * patch_size, 400))


def remove(
    image: Image,
    mask: MaskLike,
    *,
    quality: Quality = "balanced",
    patch_size: int | None = None,
    seed: int = 0,
    source_mask: MaskLike | None = None,
    margin: int | None = None,
    progress: Progress = NULL_PROGRESS,
) -> Image:
    """Fill the masked area with texture copied from its surroundings.

    ``source_mask`` restricts where patches may come from. Pixels outside
    ``mask`` are untouched. Falls back to diffusion when there is not enough
    known context to copy from.
    """
    params = replace(PRESETS[quality], seed=seed)
    if patch_size is not None:
        params = replace(params, patch_size=patch_size)
    hard, _ = normalize_mask(mask, image.shape)
    m = margin if margin is not None else default_margin(hard, params.patch_size)

    src_full: BoolMask | None = None
    include: Box | None = None
    if source_mask is not None:
        src_full, _ = normalize_mask(source_mask, image.shape)
        include = mask_bbox(src_full)

    def _fill(roi: FloatImage, hole: BoolMask, box: Box) -> FloatImage:
        src = None if src_full is None else src_full[box[0] : box[1], box[2] : box[3]]
        try:
            return fill(roi, hole, params=params, source=src, progress=progress.for_box(box))
        except NoSourceError:
            return diffusion.fill(roi, hole)

    return run_in_roi(image, mask, _fill, margin=m, include=include)
