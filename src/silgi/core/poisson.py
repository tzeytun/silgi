"""Gradient-domain (Poisson) blending.

Solves for pixels ``f`` inside a mask Ω so that their gradients match a guide
image ``g`` while matching the original image exactly on Ω's boundary:

    Δf = Δg  in Ω,      f = target  on ∂Ω

This is "seamless cloning" from Pérez, Gangnet & Blake, "Poisson Image
Editing", SIGGRAPH 2003. Unlike ``cv2.seamlessClone`` it works on float data,
so 16-bit images and alpha channels are supported, and it never touches pixels
outside Ω.
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ._types import BoolMask, FloatImage

#: Above this many unknowns the sparse direct solver is replaced by CG.
DIRECT_LIMIT = 250_000
#: Coarse-to-fine initial guesses are only worthwhile for regions at least this big.
MIN_COARSE_SIDE = 64


def _resize(
    a: npt.NDArray[np.float32], size: tuple[int, int], interp: int
) -> npt.NDArray[np.float32]:
    """cv2.resize that keeps a trailing channel axis."""
    out = cv2.resize(a, size, interpolation=interp)
    return np.asarray(out.reshape(size[1], size[0], -1), dtype=np.float32)


def _coarse_start(
    target: FloatImage, guide: FloatImage, mask: BoolMask, boundary_gradients: bool, factor: int = 4
) -> FloatImage:
    """Initial guess for the fine problem from a solve at 1/``factor`` resolution.

    ``f - g`` is harmonic inside the mask, hence smooth: solving it on a coarse
    grid and interpolating leaves only high-frequency error, which a few CG
    iterations remove.
    """
    h, w, _ = target.shape
    size = ((w + factor - 1) // factor, (h + factor - 1) // factor)
    tc = _resize(target, size, cv2.INTER_AREA)
    gc = _resize(guide, size, cv2.INTER_AREA)
    mc = _resize(mask.astype(np.float32)[..., None], size, cv2.INTER_AREA)[..., 0] > 0.5
    solved = blend(tc, gc, mc, boundary_gradients=boundary_gradients)
    # correction field: solution minus guide inside, boundary data (target - guide) outside
    corr = np.where(mc[..., None], solved - gc, tc - gc)
    up = _resize(corr.astype(np.float32), (w, h), cv2.INTER_LINEAR)
    return np.asarray(guide + up, dtype=np.float32)


def blend(
    target: FloatImage,
    guide: FloatImage,
    mask: BoolMask,
    *,
    boundary_gradients: bool = True,
) -> FloatImage:
    """Return ``target`` with Ω (= ``mask``) replaced by the Poisson solution
    guided by ``guide``'s gradients.

    With ``boundary_gradients`` the guide must be meaningful just outside Ω
    too (seamless cloning of a shifted source). Without it only gradients
    inside Ω are used, so a fill that exists only inside Ω adapts smoothly to
    the boundary (seam removal).
    """
    if target.shape != guide.shape or target.shape[:2] != mask.shape:
        raise ValueError("target, guide and mask must have matching shapes")
    h, w, c = target.shape
    ys, xs = np.nonzero(mask)
    n = ys.size
    out = target.copy()
    if n == 0:
        return out

    index = np.full((h, w), -1, np.int64)
    index[ys, xs] = np.arange(n)

    rows: list[npt.NDArray[np.int64]] = []
    cols: list[npt.NDArray[np.int64]] = []
    diag = np.zeros(n, np.float64)
    b = np.zeros((n, c), np.float64)
    g = guide.astype(np.float64)
    t = target.astype(np.float64)
    has_boundary = False

    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ny, nx = ys + dy, xs + dx
        inside = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        p = np.flatnonzero(inside)
        qy, qx = ny[p], nx[p]
        diag[p] += 1.0
        q_idx = index[qy, qx]
        in_omega = q_idx >= 0
        guided = p if boundary_gradients else p[in_omega]
        gq = (qy, qx) if boundary_gradients else (qy[in_omega], qx[in_omega])
        b[guided] += g[ys[guided], xs[guided]] - g[gq]
        rows.append(p[in_omega])
        cols.append(q_idx[in_omega])
        boundary = p[~in_omega]
        if boundary.size:
            has_boundary = True
            b[boundary] += t[qy[~in_omega], qx[~in_omega]]

    if not has_boundary:
        # no Dirichlet boundary (mask covers everything): keep the guide
        out[ys, xs] = guide[ys, xs]
        return out

    r = np.concatenate(rows)
    cc = np.concatenate(cols)
    off = sp.csr_matrix((-np.ones(r.size), (r, cc)), shape=(n, n))
    a = (off + sp.diags(diag)).tocsc()

    if n <= DIRECT_LIMIT:
        solve = spla.factorized(a)
        sol = np.column_stack([solve(b[:, k]) for k in range(c)])
    else:
        pre = sp.diags(1.0 / diag)
        sol = np.empty((n, c), np.float64)
        coarse = min(h, w) >= MIN_COARSE_SIDE
        start = (
            _coarse_start(target, guide, mask, boundary_gradients).astype(np.float64)
            if coarse
            else g
        )
        for k in range(c):
            x0 = start[ys, xs, k]
            x, _ = spla.cg(a, b[:, k], x0=x0, rtol=1e-7, maxiter=120 if coarse else 4000, M=pre)
            sol[:, k] = x
    out[ys, xs] = np.clip(sol, 0.0, 1.0).astype(np.float32)
    return out
