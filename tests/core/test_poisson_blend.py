from __future__ import annotations

import numpy as np
import pytest

from silgi.core import blend, poisson
from silgi.samples import scene


def _hole(h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), bool)
    m[h // 4 : 3 * h // 4, w // 3 : 2 * w // 3] = True
    return m


def test_guide_equal_to_target_is_identity() -> None:
    t = scene("noise", 64, 64)
    m = _hole(64, 64)
    out = poisson.blend(t, t, m)
    assert np.allclose(out, t, atol=1e-5)


def test_poisson_removes_constant_offset() -> None:
    t = scene("gradient", 64, 64)
    m = _hole(64, 64)
    guide = np.clip(t + 0.15, 0, 1).astype(np.float32)
    out = poisson.blend(t, guide, m)
    assert np.abs(out[m] - t[m]).max() < 2e-3
    assert np.array_equal(out[~m], t[~m])


def test_poisson_iterative_path_matches_direct(monkeypatch: pytest.MonkeyPatch) -> None:
    t = scene("noise", 48, 48)
    m = _hole(48, 48)
    guide = np.clip(t * 0.8 + 0.1, 0, 1).astype(np.float32)
    direct = poisson.blend(t, guide, m)
    monkeypatch.setattr(poisson, "DIRECT_LIMIT", 0)
    iterative = poisson.blend(t, guide, m)
    assert np.abs(direct - iterative).max() < 1e-4


def test_poisson_without_boundary_keeps_guide() -> None:
    t = scene("flat", 8, 8)
    g = scene("noise", 8, 8)
    out = poisson.blend(t, g, np.ones((8, 8), bool))
    assert np.array_equal(out, g)


def test_poisson_empty_and_shape_errors() -> None:
    t = scene("flat", 8, 8)
    assert np.array_equal(poisson.blend(t, t, np.zeros((8, 8), bool)), t)
    with pytest.raises(ValueError, match="matching"):
        poisson.blend(t, t[:4], np.zeros((8, 8), bool))


def test_inward_ramp_is_zero_outside_and_one_deep_inside() -> None:
    m = np.zeros((40, 40), bool)
    m[5:35, 5:35] = True
    w = blend.inward_ramp(m, 4)
    assert (w[~m] == 0).all()
    assert w[20, 20] == 1.0
    assert 0 < w[5, 20] < 1
    assert np.array_equal(blend.inward_ramp(m, 0), m.astype(np.float32))


def test_finish_modes() -> None:
    t = scene("gradient", 32, 32)
    f = np.clip(t + 0.1, 0, 1).astype(np.float32)
    m = _hole(32, 32)
    assert np.array_equal(blend.finish(t, f, m), f)
    seamed = blend.finish(t, f, m, seam_mode="poisson", feather_px=2)
    assert np.array_equal(seamed[~m], t[~m])
    with pytest.raises(ValueError, match="seam"):
        blend.finish(t, f, m, seam_mode="bogus")  # type: ignore[arg-type]


def test_two_level_solver_is_close_to_the_direct_solution(monkeypatch: pytest.MonkeyPatch) -> None:
    t = scene("noise", 200, 240)
    g = np.roll(t, 30, axis=1)
    m = np.zeros((200, 240), bool)
    m[40:160, 50:190] = True
    direct = poisson.blend(t, g, m)
    monkeypatch.setattr(poisson, "DIRECT_LIMIT", 0)
    two_level = poisson.blend(t, g, m)
    assert np.array_equal(two_level[~m], t[~m])
    assert np.abs(two_level - direct).mean() < 2e-3
    assert np.abs(two_level - direct).max() < 0.05
