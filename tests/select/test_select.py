from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from silgi.core.select import (
    brush_stroke,
    color_range,
    combine,
    ellipse,
    feather,
    find_similar,
    find_text,
    grow,
    magic_wand,
    polygon,
    rectangle,
    shrink,
    smooth,
)
from silgi.core.select.colorrange import sample_color
from silgi.core.select.morph import load_mask, mask_to_png, save_mask
from silgi.core.select.similar import hits_to_mask
from silgi.core.select.textfind import boxes_to_mask
from silgi.samples import composite, demo_image, fake_logo, landscape, scene, to_uint8, to_uint16

masks = arrays(np.bool_, st.tuples(st.integers(1, 40), st.integers(1, 40)))
radii = st.integers(0, 6)

# ------------------------------------------------------------- property tests


@settings(max_examples=150, deadline=None)
@given(masks, radii)
def test_grow_contains_and_shrink_is_contained(m: np.ndarray, r: int) -> None:
    assert (grow(m, r) | m).sum() == grow(m, r).sum()
    assert not (shrink(m, r) & ~m).any()


@settings(max_examples=150, deadline=None)
@given(masks, radii)
def test_closing_contains_and_opening_is_contained(m: np.ndarray, r: int) -> None:
    assert not (m & ~shrink(grow(m, r), r)).any()
    assert not (grow(shrink(m, r), r) & ~m).any()


@settings(max_examples=150, deadline=None)
@given(masks, radii)
def test_grow_and_shrink_are_monotonic(m: np.ndarray, r: int) -> None:
    assert grow(m, r).sum() <= grow(m, r + 1).sum()
    assert shrink(m, r).sum() >= shrink(m, r + 1).sum()


@settings(max_examples=150, deadline=None)
@given(masks, st.floats(0, 8))
def test_feather_is_inward_and_bounded(m: np.ndarray, px: float) -> None:
    f = feather(m, px)
    assert f.dtype == np.float32
    assert (f >= 0).all()
    assert (f <= 1).all()
    assert (f[~m] == 0).all()
    assert ((f > 0) == m).all()


@settings(max_examples=150, deadline=None)
@given(masks, masks)
def test_combine_algebra(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        b = np.resize(b, a.shape)
    assert np.array_equal(combine(a, b, "add"), a | b)
    assert np.array_equal(combine(a, b, "subtract"), a & ~b)
    assert np.array_equal(combine(a, b, "intersect"), a & b)
    assert np.array_equal(combine(a, b, "replace"), b)
    assert np.array_equal(combine(combine(a, b, "subtract"), b, "intersect"), np.zeros_like(a))


@settings(max_examples=100, deadline=None)
@given(masks, st.integers(0, 4))
def test_smooth_stays_near_mask(m: np.ndarray, r: int) -> None:
    s = smooth(m, r)
    assert s.shape == m.shape
    assert not (s & ~grow(m, r + 1)).any()


@settings(max_examples=100, deadline=None)
@given(masks)
def test_mask_png_roundtrip(m: np.ndarray) -> None:
    assert np.array_equal(load_mask(mask_to_png(m)), m)


# ------------------------------------------------------------- unit tests


def test_zero_radius_ops_copy() -> None:
    m = np.eye(5, dtype=bool)
    for fn in (grow, shrink, smooth):
        out = fn(m, 0)
        assert np.array_equal(out, m)
        assert out is not m
    assert np.array_equal(feather(m, 0), m.astype(np.float32))


def test_combine_errors() -> None:
    with pytest.raises(ValueError, match="shape"):
        combine(np.zeros((2, 2), bool), np.zeros((3, 3), bool))
    with pytest.raises(ValueError, match="mode"):
        combine(np.zeros((2, 2), bool), np.zeros((2, 2), bool), "xor")  # type: ignore[arg-type]


def test_mask_files(tmp_path: Path) -> None:
    m = np.zeros((8, 10), bool)
    m[2:5, 3:7] = True
    p = tmp_path / "m.png"
    save_mask(p, m)
    assert np.array_equal(load_mask(p, (8, 10)), m)
    with pytest.raises(ValueError, match="image is"):
        load_mask(p, (9, 10))
    soft = mask_to_png(feather(m, 2))
    assert load_mask(soft).shape == (8, 10)
    # an RGBA mask uses alpha; a fully opaque RGBA mask uses its colour
    from PIL import Image

    rgba = np.zeros((8, 10, 4), np.uint8)
    rgba[..., 3] = m * 255
    Image.fromarray(rgba).save(tmp_path / "a.png")
    assert np.array_equal(load_mask(tmp_path / "a.png"), m)
    rgba[..., 3] = 255
    rgba[..., :3] = (m * 255)[..., None]
    Image.fromarray(rgba).save(tmp_path / "b.png")
    assert np.array_equal(load_mask(tmp_path / "b.png"), m)


def test_shapes() -> None:
    r = rectangle((10, 10), 7.2, 8.0, 2.0, 3.0)
    assert r.sum() == 5 * 5
    assert r[3, 2]
    assert not r[2, 2]
    assert not rectangle((10, 10), 3, 3, 3, 8).any()
    e = ellipse((21, 21), 0, 0, 21, 21)
    assert e[10, 10]
    assert not e[0, 0]
    assert abs(e.sum() - np.pi * 10.5**2) < 30
    assert not ellipse((10, 10), 2, 2, 2, 8).any()
    tri = polygon((20, 20), [(0, 0), (20, 0), (0, 20)])
    assert tri[2, 2]
    assert not tri[18, 18]
    assert not polygon((5, 5), [(0, 0), (1, 1)]).any()


def test_brush_hard_and_soft() -> None:
    hard = brush_stroke((40, 40), [(10, 20), (30, 20)], size=8)
    assert hard[20, 20] == 1.0
    assert hard[20, 5] == 0.0
    assert set(np.unique(hard)) <= {0.0, 1.0}
    soft = brush_stroke((40, 40), [(10, 20), (30, 20)], size=10, hardness=0.2)
    assert soft[20, 20] == pytest.approx(1.0)
    assert 0 < soft[24, 20] < 1
    dot = brush_stroke((20, 20), [(10, 10)], size=6, hardness=0.5)
    assert dot[9, 9] > 0.5
    assert dot[0, 0] == 0
    assert brush_stroke((20, 20), [], size=6).sum() == 0
    assert brush_stroke((20, 20), [(5, 5)], size=4).sum() > 0


def test_magic_wand() -> None:
    img = np.zeros((20, 30, 3), np.uint8)
    img[:, :10] = (200, 10, 10)
    img[:, 20:] = (200, 10, 10)
    img[5, 5] = (205, 12, 9)
    cont = magic_wand(img, 2, 2, 0.05)
    assert cont[:, :10].all()
    assert not cont[:, 20:].any()
    glob = magic_wand(img, 2, 2, 0.05, contiguous=False)
    assert glob[:, 20:].all()
    assert not glob[:, 10:20].any()
    assert magic_wand(to_uint16(scene("flat", 10, 10)), 1, 1, 0.0).all()
    assert magic_wand(img[:, :, 0], 0, 0, 0.1).shape == (20, 30)
    with pytest.raises(ValueError, match="outside"):
        magic_wand(img, 99, 0)


def test_color_range() -> None:
    img = np.zeros((10, 20, 3), np.uint8)
    img[:, 10:] = (240, 240, 240)
    sel = color_range(img, (1.0, 1.0, 1.0), tolerance=10)
    assert sel.dtype == bool
    assert sel[:, 10:].all()
    assert not sel[:, :10].any()
    soft = color_range(img, (1.0, 1.0, 1.0), tolerance=2, softness=20)
    assert soft.dtype == np.float32
    assert 0 < soft[0, 15] <= 1
    assert sample_color(img, 15, 5) == pytest.approx((240 / 255,) * 3)
    gray = img[:, :, 0]
    assert color_range(gray, (0.0, 0.0, 0.0), 5)[:, :10].all()
    assert sample_color(gray, 0, 0) == (0.0, 0.0, 0.0)


def test_text_finder_on_demo_and_clean_scenes() -> None:
    d = demo_image()
    boxes = find_text(to_uint8(d.watermarked))
    ty, tx = d.text_pos
    th, tw = d.text_mark.alpha.shape
    big = [b for b in boxes if b.w > 300]
    assert big, boxes
    b = big[0]
    cx, cy = b.x + b.w / 2, b.y + b.h / 2
    assert tx < cx < tx + tw
    assert ty < cy < ty + th
    small = [b for b in boxes if b.y > 800]
    assert len(small) == 2
    assert len(find_text(to_uint8(scene("checker", 300, 300)))) == 0
    assert len(find_text(to_uint8(scene("stripes", 300, 300)))) == 0
    mask = boxes_to_mask((900, 1350), boxes)
    assert mask[cy.__int__(), cx.__int__()]
    assert b.as_tuple() == (b.x, b.y, b.w, b.h)


def test_text_finder_dark_text_and_downscaling() -> None:
    img = np.full((300, 600, 3), 230, np.uint8)
    cv2.putText(
        img,
        "Hello world 2026",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (30, 30, 30),
        3,
        cv2.LINE_AA,
    )
    boxes = find_text(img)
    assert len(boxes) == 1
    assert boxes[0].y < 150 < boxes[0].y + boxes[0].h
    big = cv2.resize(img, (2400, 1200), interpolation=cv2.INTER_CUBIC)
    assert len(find_text(big)) >= 1


def test_find_similar_tiled_logos() -> None:
    img = landscape(700, 1000, seed=3)
    small = fake_logo(80, opacity=0.6)
    positions = [(ty, tx) for ty in range(50, 600, 250) for tx in range(60, 900, 300)]
    for ty, tx in positions:
        img = composite(img, small, ty, tx)
    img8 = to_uint8(img)
    hits = find_similar(img8, (60, 50, 80, 80))
    assert len(hits) == len(positions)
    assert sum(h.is_source for h in hits) == 1
    found = {(h.y, h.x) for h in hits}
    for p in positions:
        assert any(abs(p[0] - y) <= 2 and abs(p[1] - x) <= 2 for y, x in found)
    m = hits_to_mask(img8.shape[:2], hits, (60, 50, 80, 80))
    assert m.sum() == len(positions) * 80 * 80
    src = np.zeros(img8.shape[:2], bool)
    src[60:120, 70:130] = True
    shaped = hits_to_mask(img8.shape[:2], hits, (60, 50, 80, 80), src)
    assert 0 < shaped.sum() < m.sum()


def test_find_similar_scaled_and_errors() -> None:
    base = landscape(500, 800, seed=4)
    logo = fake_logo(80, opacity=0.7)
    img = composite(base, logo, 40, 40)
    big = fake_logo(100, opacity=0.7)
    img = composite(img, big, 300, 500)
    hits = find_similar(to_uint8(img), (40, 40, 80, 80), threshold=0.5)
    assert any(abs(h.x - 500) < 8 and abs(h.y - 300) < 8 and h.scale == 1.25 for h in hits)
    with pytest.raises(ValueError, match="4x4"):
        find_similar(to_uint8(img), (0, 0, 2, 2))
    with pytest.raises(ValueError, match="no detail"):
        find_similar(np.zeros((50, 50), np.uint8), (5, 5, 10, 10))
    huge = cv2.resize(to_uint8(img), (3200, 2000))
    assert find_similar(huge, (160, 160, 320, 320), max_results=3)
    assert hits_to_mask((10, 10), [], (0, 0, 4, 4)).sum() == 0
