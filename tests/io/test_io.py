from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageCms, ImageOps

from silgi.core.color import to_display_rgb8
from silgi.core.io import (
    ExportOptions,
    UnsupportedImageError,
    _pil_to_array,
    apply_orientation,
    encode_image,
    format_from_path,
    read_image,
    save_image,
)
from tests.conftest import make_image

ICC = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def _exif(orientation: int) -> bytes:
    e = Image.Exif()
    e[0x0112] = orientation
    e[0x010F] = "silgi-test-camera"
    return e.tobytes()


@pytest.mark.parametrize("fmt", ["png", "tiff"])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_lossless_roundtrip_bit_exact(
    rng: np.random.Generator, fmt: str, dtype: type, channels: int
) -> None:
    img = make_image(rng, 37, 53, channels, dtype)
    enc = encode_image(img, ExportOptions(format=fmt), icc_profile=ICC)  # type: ignore[arg-type]
    assert enc.lossless
    back = read_image(enc.data)
    assert back.pixels.dtype == img.dtype
    assert back.pixels.shape == img.shape
    assert np.array_equal(back.pixels, img)
    assert back.icc_profile == ICC
    assert back.bit_depth == (16 if dtype == np.uint16 else 8)
    assert back.has_alpha == (channels == 4)
    assert (back.height, back.width, back.channels) == (37, 53, channels)


def test_webp_lossless_is_exact(rng: np.random.Generator) -> None:
    img = make_image(rng, 30, 40, 4)
    enc = encode_image(img, ExportOptions(format="webp"), icc_profile=ICC)
    back = read_image(enc.data)
    assert np.array_equal(back.pixels, img)
    assert back.icc_profile == ICC


def test_webp_gray_and_lossy(rng: np.random.Generator) -> None:
    gray = make_image(rng, 20, 20, 1)
    back = read_image(encode_image(gray, ExportOptions(format="webp")).data)
    assert back.pixels.shape == (20, 20, 3)
    lossy = encode_image(
        make_image(rng, 20, 20, 3), ExportOptions(format="webp", webp_lossless=False)
    )
    assert not lossy.lossless
    assert any("lossy" in w for w in lossy.warnings)


def test_jpeg_reduces_16bit_and_flattens_alpha(rng: np.random.Generator) -> None:
    img = make_image(rng, 16, 16, 4, np.uint16)
    enc = encode_image(img, ExportOptions(format="jpeg", quality=80))
    assert not enc.lossless
    joined = " ".join(enc.warnings)
    assert "8-bit" in joined
    assert "transparency" in joined
    back = read_image(enc.data)
    assert back.pixels.shape == (16, 16, 3)
    assert back.pixels.dtype == np.uint8


def test_jpeg_gray(rng: np.random.Generator) -> None:
    back = read_image(encode_image(make_image(rng, 16, 16, 1), ExportOptions(format="jpeg")).data)
    assert back.pixels.shape == (16, 16)


@pytest.mark.parametrize("orientation", range(1, 9))
@pytest.mark.parametrize("pil_format", ["PNG", "JPEG"])
def test_exif_orientation_matches_pillow(orientation: int, pil_format: str) -> None:
    base = np.random.default_rng(orientation).integers(0, 255, (12, 20, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(base).save(buf, pil_format, exif=_exif(orientation), quality=100)
    data = buf.getvalue()
    expected = np.asarray(ImageOps.exif_transpose(Image.open(io.BytesIO(data))))
    loaded = read_image(data)
    assert loaded.orientation == orientation
    assert np.array_equal(loaded.pixels, expected)


def test_metadata_off_by_default_and_orientation_reset() -> None:
    base = np.zeros((10, 20, 3), np.uint8)
    buf = io.BytesIO()
    Image.fromarray(base).save(buf, "JPEG", exif=_exif(6), icc_profile=ICC)
    src = read_image(buf.getvalue())
    assert src.pixels.shape == (20, 10, 3)

    stripped = read_image(
        encode_image(src.pixels, ExportOptions(), icc_profile=src.icc_profile, exif=src.exif).data
    )
    assert stripped.exif is None
    assert stripped.icc_profile == ICC

    no_icc = read_image(
        encode_image(src.pixels, ExportOptions(keep_icc=False), icc_profile=src.icc_profile).data
    )
    assert no_icc.icc_profile is None

    for fmt in ("png", "jpeg", "webp"):
        kept = encode_image(
            src.pixels,
            ExportOptions(format=fmt, keep_metadata=True),  # type: ignore[arg-type]
            icc_profile=src.icc_profile,
            exif=src.exif,
        )
        back = read_image(kept.data)
        assert back.orientation == 1, fmt
        assert back.pixels.shape == (20, 10, 3), fmt
        exif = Image.Exif()
        assert back.exif is not None
        exif.load(back.exif)
        assert exif[0x010F] == "silgi-test-camera"


def test_tiff_warns_about_exif() -> None:
    enc = encode_image(
        np.zeros((4, 4, 3), np.uint8),
        ExportOptions(format="tiff", keep_metadata=True),
        exif=_exif(1),
    )
    assert any("EXIF" in w for w in enc.warnings)


def test_save_image_uses_extension(tmp_path: Path, rng: np.random.Generator) -> None:
    img = make_image(rng, 10, 12, 3, np.uint16)
    src = read_image(encode_image(img, ExportOptions(), icc_profile=ICC).data)
    out = tmp_path / "x.tif"
    enc = save_image(out, src.pixels, source=src)
    assert enc.format == "tiff"
    back = read_image(out)
    assert np.array_equal(back.pixels, img)
    assert back.icc_profile == ICC
    with open(out, "rb") as fh:
        assert np.array_equal(read_image(fh).pixels, img)


def test_format_from_path() -> None:
    assert format_from_path("a.JPG") == "jpeg"
    assert format_from_path("a.tiff") == "tiff"
    with pytest.raises(ValueError, match="extension"):
        format_from_path("a.gif")
    with pytest.raises(ValueError, match="format"):
        encode_image(np.zeros((2, 2), np.uint8), ExportOptions(format="gif"))  # type: ignore[arg-type]


def test_unsupported_input() -> None:
    with pytest.raises(UnsupportedImageError):
        read_image(b"definitely not an image")


@pytest.mark.parametrize(
    ("mode", "shape"),
    [
        ("P", (7, 9, 3)),
        ("PA", (7, 9, 4)),
        ("LA", (7, 9, 4)),
        ("1", (7, 9)),
        ("CMYK", (7, 9, 3)),
        ("I;16", (7, 9)),
        ("I", (7, 9)),
        ("YCbCr", (7, 9, 3)),
    ],
)
def test_pillow_modes_are_normalised(mode: str, shape: tuple[int, ...]) -> None:
    img = Image.new("RGB", (9, 7), (10, 200, 30)).convert(mode)
    notes: list[str] = []
    arr = _pil_to_array(img, notes)
    assert arr.shape == shape
    assert arr.dtype == (np.uint16 if mode.startswith("I") else np.uint8)


def test_pillow_unsupported_mode() -> None:
    with pytest.raises(UnsupportedImageError):
        _pil_to_array(Image.new("F", (2, 2)), [])


def test_cmyk_jpeg_drops_profile_and_converts() -> None:
    buf = io.BytesIO()
    Image.new("CMYK", (5, 4), (0, 50, 100, 0)).save(buf, "JPEG")
    loaded = read_image(buf.getvalue())
    assert loaded.pixels.shape == (4, 5, 3)
    assert loaded.icc_profile is None
    assert loaded.notes


def test_gif_palette_with_transparency() -> None:
    img = Image.new("P", (6, 5), 1)
    img.putpalette([0, 0, 0, 255, 0, 0] + [0] * 762)
    img.putpixel((0, 0), 0)
    buf = io.BytesIO()
    img.save(buf, "GIF", transparency=0)
    loaded = read_image(buf.getvalue())
    assert loaded.pixels.shape[:2] == (5, 6)
    assert loaded.pixels.shape[2] in (3, 4)


def test_display_conversion(rng: np.random.Generator) -> None:
    for ch in (1, 3, 4):
        img = make_image(rng, 8, 8, ch, np.uint16)
        disp = to_display_rgb8(img, ICC)
        assert disp.dtype == np.uint8
        assert disp.shape == (8, 8, 4 if ch == 4 else 3)
    la = np.dstack([make_image(rng, 8, 8, 1), make_image(rng, 8, 8, 1)])
    assert to_display_rgb8(la).shape == (8, 8, 4)
    assert to_display_rgb8(make_image(rng, 8, 8, 3), b"broken icc").shape == (8, 8, 3)


def test_apply_orientation_identity() -> None:
    a = np.arange(6).reshape(2, 3)
    assert np.array_equal(apply_orientation(a, 1), a)
