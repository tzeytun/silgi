from __future__ import annotations

import numpy as np
import pytest

from silgi.core.color import from_float, to_float
from silgi.core.roi import mask_bbox, normalize_mask, paste, roi_box_for, run_in_roi


def test_normalize_bool_mask_has_no_weights() -> None:
    m = np.zeros((4, 5), bool)
    m[1, 2] = True
    hard, w = normalize_mask(m, (4, 5, 3))
    assert hard is m
    assert w is None


@pytest.mark.parametrize(
    ("mask", "expect_soft"),
    [
        (np.array([[0, 255]], np.uint8), False),
        (np.array([[0, 128]], np.uint8), True),
        (np.array([[0, 65535]], np.uint16), False),
        (np.array([[0.0, 0.5]], np.float32), True),
        (np.array([[[0], [255]]], np.uint8).reshape(1, 2, 1), False),
    ],
)
def test_normalize_mask_dtypes(mask: np.ndarray, expect_soft: bool) -> None:
    hard, w = normalize_mask(mask, (1, 2))
    assert hard.tolist() == [[False, True]]
    assert (w is not None) == expect_soft


def test_normalize_mask_rejects_bad_shape_and_dtype() -> None:
    with pytest.raises(ValueError, match="does not match"):
        normalize_mask(np.zeros((3, 3), bool), (4, 4))
    with pytest.raises(TypeError):
        normalize_mask(np.zeros((2, 2), np.int32), (2, 2))


def test_mask_bbox() -> None:
    m = np.zeros((10, 10), bool)
    assert mask_bbox(m) is None
    m[2:4, 5:9] = True
    assert mask_bbox(m) == (2, 4, 5, 9)
    assert roi_box_for(m, (10, 10), 3) == (0, 7, 2, 10)
    assert roi_box_for(np.zeros((10, 10), bool), (10, 10), 3) is None


def test_empty_mask_returns_copy() -> None:
    img = np.arange(12, dtype=np.uint8).reshape(3, 4)
    out = run_in_roi(img, np.zeros((3, 4), bool), lambda r, m, b: r * 0, margin=1)
    assert np.array_equal(out, img)
    assert out is not img


def test_run_in_roi_writes_only_inside_mask() -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 65535, (20, 30, 3), dtype=np.uint16)
    m = np.zeros((20, 30), bool)
    m[5:8, 10:12] = True
    seen: list[tuple[int, int, int, int]] = []

    def fill(roi: np.ndarray, mask: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        seen.append(box)
        assert roi.dtype == np.float32
        assert mask.shape == roi.shape[:2]
        return np.zeros_like(roi)

    out = run_in_roi(img, m, fill, margin=2)
    assert seen == [(3, 10, 8, 14)]
    assert np.array_equal(out[~m], img[~m])
    assert (out[m] == 0).all()


def test_fill_with_wrong_shape_raises() -> None:
    img = np.zeros((5, 5), np.uint8)
    m = np.zeros((5, 5), bool)
    m[2, 2] = True
    with pytest.raises(RuntimeError, match="shape"):
        run_in_roi(img, m, lambda r, mk, b: r[:1], margin=1)


def test_soft_weights_blend() -> None:
    img = np.full((1, 3, 1), 100, np.uint8)
    weights = np.array([[0.0, 0.5, 1.0]], np.float32)
    hard = weights > 0
    out = paste(img, np.full((1, 3, 1), 200 / 255, np.float32), (0, 1, 0, 3), hard, weights)
    assert out[0, :, 0].tolist() == [100, 150, 200]


def test_float_roundtrip_is_exact_for_integer_types() -> None:
    for dt, top in ((np.uint8, 255), (np.uint16, 65535)):
        vals = np.arange(0, top + 1, dtype=dt).reshape(1, -1)
        assert np.array_equal(from_float(to_float(vals), dt)[:, :, 0], vals)
