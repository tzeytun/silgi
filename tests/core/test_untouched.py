"""The core guarantee: pixels outside the mask are bit-identical to the input."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from silgi.core import engine
from silgi.core.roi import mask_bbox
from tests.conftest import make_image, random_mask

N_MASKS = 12


def _watermark(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    a = np.zeros((h, w), np.float32)
    a[h // 4 : 3 * h // 4, w // 5 : 4 * w // 5] = rng.uniform(0.2, 0.95)
    rgba = np.dstack([np.full((h, w, 3), rng.uniform(0, 1), np.float32), a])
    return np.rint(rgba * 255).astype(np.uint8)


def _offset_for(mask: np.ndarray, shape: tuple[int, ...]) -> tuple[int, int] | None:
    box = mask_bbox(mask)
    assert box is not None
    h, w = shape[:2]
    for dy, dx in (
        (0, box[3] - box[2] + 2),
        (0, -(box[3] - box[2] + 2)),
        (box[1] - box[0] + 2, 0),
        (-(box[1] - box[0] + 2), 0),
    ):
        if box[0] + dy >= 0 and box[1] + dy <= h and box[2] + dx >= 0 and box[3] + dx <= w:
            return dy, dx
    return None


Runner = Callable[[np.ndarray, np.ndarray, np.random.Generator], np.ndarray | None]


def _run(method: str) -> Runner:
    def run(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
        kwargs: dict[str, object] = {}
        if method == "alpha":
            wm = _watermark(rng, int(rng.integers(8, 30)), int(rng.integers(8, 40)))
            kwargs["watermark"] = wm
            kwargs["watermark_position"] = (
                float(rng.integers(-4, img.shape[0] - 4)),
                float(rng.integers(-4, img.shape[1] - 4)),
            )
        if method in ("clone", "heal"):
            off = _offset_for(mask, img.shape)
            if off is None:
                return None
            kwargs["offset"] = off
        if method == "patchmatch":
            kwargs["quality"] = "fast"
        return engine.remove(img, mask, method, **kwargs).image  # type: ignore[arg-type]

    return run


@pytest.mark.parametrize("method", ["telea", "ns", "patchmatch", "alpha", "clone", "heal", "auto"])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_pixels_outside_mask_are_untouched(method: str, dtype: type, channels: int) -> None:
    rng = np.random.default_rng(hash((method, channels, dtype.__name__)) % (2**32))
    run = _run(method)
    checked = 0
    max_frac = 0.12 if method in ("clone", "heal") else 0.25
    for _ in range(N_MASKS * 4):
        if checked == N_MASKS:
            break
        h, w = int(rng.integers(24, 72)), int(rng.integers(24, 90))
        img = make_image(rng, h, w, channels, dtype)
        mask = random_mask(rng, h, w, max_frac)
        if not mask.any():
            continue
        before = img.copy()
        out = run(img, mask, rng)
        if out is None:
            continue
        checked += 1
        assert np.array_equal(img, before), "input was modified in place"
        assert out.shape == img.shape
        assert out.dtype == img.dtype
        assert np.array_equal(out[~mask], img[~mask])
    assert checked == N_MASKS


@pytest.mark.parametrize("method", ["telea", "patchmatch"])
def test_soft_masks_only_touch_selected_pixels(method: str) -> None:
    rng = np.random.default_rng(5)
    img = make_image(rng, 48, 64, 3)
    soft = np.zeros((48, 64), np.float32)
    soft[10:30, 20:40] = rng.uniform(0.0, 1.0, (20, 20)).astype(np.float32)
    out = engine.remove(img, soft, method).image  # type: ignore[arg-type]
    sel = soft > 0
    assert np.array_equal(out[~sel], img[~sel])


def test_finishing_stays_inside_mask() -> None:
    rng = np.random.default_rng(9)
    img = make_image(rng, 64, 64, 3, np.uint16)
    mask = np.zeros((64, 64), bool)
    mask[20:40, 15:45] = True
    out = engine.remove(img, mask, "patchmatch", feather=3, seam="poisson").image
    assert np.array_equal(out[~mask], img[~mask])
