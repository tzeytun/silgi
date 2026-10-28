"""Image loading and saving that preserves what the user gave us.

* Bit depth: 8- and 16-bit are kept. PNG and TIFF pixels are decoded with
  OpenCV because Pillow truncates 16-bit RGB to 8 bits.
* Alpha, ICC profile and EXIF are read with Pillow.
* EXIF orientation is applied on load so pixels are upright; when metadata is
  written back the orientation tag is reset to 1.
* Nothing is ever resized.
"""

from __future__ import annotations

import io
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Literal

import cv2
import numpy as np
import tifffile
from PIL import Image as PILImage

from ._types import Image
from .color import as_hwc, to_8bit

try:  # optional HEIC/HEIF support
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIF_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the optional extra
    HEIF_AVAILABLE = False

ExportFormat = Literal["png", "jpeg", "webp", "tiff"]
EXPORT_FORMATS: tuple[ExportFormat, ...] = ("png", "jpeg", "webp", "tiff")
LOSSLESS_FORMATS = frozenset({"png", "tiff"})

_EXTENSIONS: dict[str, ExportFormat] = {
    ".png": "png",
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".webp": "webp",
    ".tif": "tiff",
    ".tiff": "tiff",
}

_ORIENTATION_TAG = 0x0112
_EXIF_HEADER = b"Exif\x00\x00"

PathOrBytes = str | Path | bytes | IO[bytes]


class UnsupportedImageError(ValueError):
    """The file could not be decoded or uses an unsupported pixel format."""


@dataclass
class ImageFile:
    """Decoded pixels plus the metadata needed to write them back faithfully."""

    pixels: Image
    format: str
    icc_profile: bytes | None = None
    exif: bytes | None = None
    orientation: int = 1
    notes: list[str] = field(default_factory=list)

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def channels(self) -> int:
        return 1 if self.pixels.ndim == 2 else int(self.pixels.shape[2])

    @property
    def bit_depth(self) -> int:
        return 16 if self.pixels.dtype == np.uint16 else 8

    @property
    def has_alpha(self) -> bool:
        return self.channels == 4


@dataclass(frozen=True)
class ExportOptions:
    format: ExportFormat = "png"
    quality: int = 95
    keep_metadata: bool = False
    keep_icc: bool = True
    webp_lossless: bool = True


@dataclass
class Encoded:
    data: bytes
    format: ExportFormat
    lossless: bool
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------- reading


def _read_bytes(src: PathOrBytes) -> bytes:
    if isinstance(src, bytes):
        return src
    if isinstance(src, str | Path):
        return Path(src).read_bytes()
    return src.read()


def apply_orientation(pixels: Image, orientation: int) -> Image:
    """Apply an EXIF orientation (1..8) so the result is upright."""
    a = pixels
    if orientation == 2:
        a = a[:, ::-1]
    elif orientation == 3:
        a = a[::-1, ::-1]
    elif orientation == 4:
        a = a[::-1]
    elif orientation == 5:
        a = a.swapaxes(0, 1)
    elif orientation == 6:
        a = np.rot90(a, -1)
    elif orientation == 7:
        a = a[::-1, ::-1].swapaxes(0, 1)
    elif orientation == 8:
        a = np.rot90(a, 1)
    return np.ascontiguousarray(a)


def _pil_to_array(pil: PILImage.Image, notes: list[str]) -> Image:
    mode = pil.mode
    if mode in ("RGB", "RGBA", "L"):
        return np.asarray(pil)
    if mode == "1":
        return np.asarray(pil.convert("L"))
    if mode in ("LA", "PA", "La", "RGBa"):
        return np.asarray(pil.convert("RGBA"))
    if mode == "P":
        has_alpha = "transparency" in pil.info
        return np.asarray(pil.convert("RGBA" if has_alpha else "RGB"))
    if mode.startswith("I;16"):
        return np.asarray(pil).astype(np.uint16)
    if mode == "I":
        return np.asarray(np.clip(np.asarray(pil), 0, 65535), dtype=np.uint16)
    if mode in ("CMYK", "YCbCr", "LAB", "HSV"):
        notes.append(f"converted {mode} to RGB")
        return np.asarray(pil.convert("RGB"))
    raise UnsupportedImageError(f"unsupported pixel mode: {mode}")


def _cv2_decode(data: bytes) -> Image | None:
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        return None
    if arr.dtype not in (np.uint8, np.uint16):
        if arr.dtype.kind == "f":
            arr = np.clip(np.rint(np.clip(arr, 0.0, 1.0) * 65535), 0, 65535).astype(np.uint16)
        else:
            arr = np.clip(arr, 0, 65535).astype(np.uint16)
    if arr.ndim == 3:
        if arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        elif arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
        elif arr.shape[2] == 1:
            arr = arr[:, :, 0]
        elif arr.shape[2] == 2:
            gray, alpha = arr[:, :, 0], arr[:, :, 1]
            arr = np.dstack([gray, gray, gray, alpha])
    return np.ascontiguousarray(arr)


def _tiff_decode(data: bytes) -> Image | None:
    """Decode plain gray/RGB TIFFs with tifffile.

    OpenCV premultiplies 8-bit RGBA TIFFs, which would change pixel values;
    anything unusual (palette, YCbCr, codecs tifffile lacks) returns ``None``
    and falls back to OpenCV.
    """
    try:
        with tifffile.TiffFile(io.BytesIO(data)) as tif:
            page = tif.pages.first
            photometric = int(page.photometric)
            if photometric not in (1, 2):  # MINISBLACK, RGB
                return None
            arr = page.asarray()
            axes = page.axes
    except Exception:
        return None
    if arr.dtype not in (np.uint8, np.uint16):
        return None
    if axes.startswith("S") and arr.ndim == 3:
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3 and arr.shape[2] == 2:
        arr = np.dstack([arr[:, :, 0], arr[:, :, 0], arr[:, :, 0], arr[:, :, 1]])
    if arr.ndim == 3 and arr.shape[2] > 4:
        arr = arr[:, :, :4]
    if arr.ndim not in (2, 3):
        return None
    return np.ascontiguousarray(arr)


def read_image(src: PathOrBytes) -> ImageFile:
    """Decode an image file into upright pixels plus metadata."""
    data = _read_bytes(src)
    try:
        pil = PILImage.open(io.BytesIO(data))
    except (OSError, PILImage.DecompressionBombError) as exc:
        raise UnsupportedImageError(f"cannot identify image: {exc}") from exc
    fmt = (pil.format or "").upper()
    notes: list[str] = []

    icc = pil.info.get("icc_profile") or None
    exif_obj = pil.getexif()
    orientation = int(exif_obj.get(_ORIENTATION_TAG, 1) or 1)
    if orientation not in range(1, 9):
        orientation = 1
    exif: bytes | None = None
    if fmt != "TIFF" and len(exif_obj):
        exif = exif_obj.tobytes()

    pixels: Image | None = None
    if fmt == "TIFF":
        pixels = _tiff_decode(data)
    if pixels is None and fmt in ("PNG", "TIFF"):
        pixels = _cv2_decode(data)
    if pixels is None:
        try:
            pil.seek(0)
            pil.load()
        except OSError as exc:
            raise UnsupportedImageError(f"cannot decode image: {exc}") from exc
        pixels = _pil_to_array(pil, notes)
        if pil.mode == "CMYK":
            icc = None  # a CMYK profile no longer describes the RGB pixels

    pixels = apply_orientation(pixels, orientation)
    return ImageFile(
        pixels=pixels,
        format=fmt,
        icc_profile=icc,
        exif=exif,
        orientation=orientation,
        notes=notes,
    )


# --------------------------------------------------------------------- writing


def format_from_path(path: str | Path) -> ExportFormat:
    ext = Path(path).suffix.lower()
    if ext not in _EXTENSIONS:
        raise ValueError(f"unsupported output extension {ext!r}; use one of {sorted(_EXTENSIONS)}")
    return _EXTENSIONS[ext]


def _normalized_exif(exif: bytes) -> bytes:
    """Return EXIF bytes (with ``Exif`` header) whose orientation is 1."""
    obj = PILImage.Exif()
    obj.load(exif)
    if _ORIENTATION_TAG in obj:
        obj[_ORIENTATION_TAG] = 1
    return obj.tobytes()


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)


def _png_insert_chunks(png: bytes, chunks: list[bytes]) -> bytes:
    if not chunks:
        return png
    # signature (8) + IHDR chunk (4 len + 4 type + 13 data + 4 crc)
    split = 8 + 25
    if png[12:16] != b"IHDR":  # pragma: no cover - OpenCV always writes IHDR first
        raise RuntimeError("unexpected PNG layout")
    return png[:split] + b"".join(chunks) + png[split:]


def _to_cv2_order(pixels: Image) -> Image:
    if pixels.ndim == 2:
        return pixels
    if pixels.shape[2] == 3:
        return cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    if pixels.shape[2] == 4:
        return cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGRA)
    return pixels[:, :, 0]


def _encode_png(pixels: Image, icc: bytes | None, exif: bytes | None) -> bytes:
    ok, buf = cv2.imencode(".png", _to_cv2_order(pixels), [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:  # pragma: no cover
        raise RuntimeError("PNG encoding failed")
    chunks = []
    if icc:
        chunks.append(_png_chunk(b"iCCP", b"ICC Profile\x00\x00" + zlib.compress(icc)))
    if exif:
        raw = exif[len(_EXIF_HEADER) :] if exif.startswith(_EXIF_HEADER) else exif
        chunks.append(_png_chunk(b"eXIf", raw))
    return _png_insert_chunks(buf.tobytes(), chunks)


def _encode_tiff(pixels: Image, icc: bytes | None) -> bytes:
    out = io.BytesIO()
    kwargs: dict[str, Any] = {"compression": "zlib", "metadata": None}
    if pixels.ndim == 2:
        kwargs["photometric"] = "minisblack"
    else:
        kwargs["photometric"] = "rgb"
        if pixels.shape[2] == 4:
            kwargs["extrasamples"] = ["unassalpha"]
    if icc:
        kwargs["extratags"] = [(34675, "B", len(icc), icc, True)]
    tifffile.imwrite(out, pixels, **kwargs)
    return out.getvalue()


def _flatten_alpha(pixels: Image) -> Image:
    rgb = pixels[:, :, :3].astype(np.float32)
    a = pixels[:, :, 3:4].astype(np.float32) / 255.0
    flat = rgb * a + 255.0 * (1.0 - a)
    return np.asarray(np.clip(np.rint(flat), 0, 255), dtype=np.uint8)


def encode_image(
    pixels: Image,
    options: ExportOptions | None = None,
    *,
    icc_profile: bytes | None = None,
    exif: bytes | None = None,
) -> Encoded:
    """Encode pixels to bytes according to ``options``.

    Metadata is only written when ``options.keep_metadata`` is true (EXIF) or
    ``options.keep_icc`` is true (ICC). Lossy formats get warnings explaining
    what changed.
    """
    options = options or ExportOptions()
    fmt = options.format
    if fmt not in EXPORT_FORMATS:
        raise ValueError(f"unsupported export format {fmt!r}")
    warnings: list[str] = []
    icc = icc_profile if options.keep_icc else None
    ex = _normalized_exif(exif) if (options.keep_metadata and exif) else None
    pixels = np.ascontiguousarray(pixels)

    if fmt == "png":
        return Encoded(_encode_png(pixels, icc, ex), fmt, True, warnings)
    if fmt == "tiff":
        if ex:
            warnings.append("EXIF is not written to TIFF files; the ICC profile is kept")
        return Encoded(_encode_tiff(pixels, icc), fmt, True, warnings)

    # JPEG and WebP are 8-bit only
    if pixels.dtype != np.uint8:
        warnings.append(f"{fmt.upper()} is 8-bit only; reduced from 16 bits per channel")
        pixels = to_8bit(pixels)
    hwc = as_hwc(pixels)
    lossless = fmt == "webp" and options.webp_lossless
    save_kwargs: dict[str, Any] = {}
    if icc:
        save_kwargs["icc_profile"] = icc
    if ex:
        save_kwargs["exif"] = ex

    if fmt == "jpeg":
        if hwc.shape[2] == 4:
            warnings.append("JPEG has no transparency; flattened onto white")
            hwc = _flatten_alpha(hwc)
        pil = PILImage.fromarray(hwc[:, :, 0] if hwc.shape[2] == 1 else hwc)
        quality = int(min(max(options.quality, 1), 100))
        save_kwargs.update(quality=quality, optimize=True)
        if quality >= 90:
            save_kwargs["subsampling"] = 0
        warnings.append("JPEG is lossy: pixels outside the selection may change slightly")
        out = io.BytesIO()
        pil.save(out, "JPEG", **save_kwargs)
        return Encoded(out.getvalue(), fmt, False, warnings)

    # webp
    if hwc.shape[2] == 1:
        hwc = np.repeat(hwc, 3, axis=2)
    pil = PILImage.fromarray(hwc)
    if lossless:
        save_kwargs.update(lossless=True, quality=100, method=6, exact=True)
    else:
        save_kwargs["quality"] = int(min(max(options.quality, 1), 100))
        warnings.append("lossy WebP: pixels outside the selection may change slightly")
    out = io.BytesIO()
    pil.save(out, "WEBP", **save_kwargs)
    return Encoded(out.getvalue(), fmt, lossless, warnings)


def save_image(
    path: str | Path,
    pixels: Image,
    options: ExportOptions | None = None,
    *,
    source: ImageFile | None = None,
) -> Encoded:
    """Write ``pixels`` to ``path``; the format follows the extension.

    ``source`` supplies the ICC profile and EXIF to carry over.
    """
    base = options or ExportOptions()
    opts = ExportOptions(
        format=format_from_path(path),
        quality=base.quality,
        keep_metadata=base.keep_metadata,
        keep_icc=base.keep_icc,
        webp_lossless=base.webp_lossless,
    )
    enc = encode_image(
        pixels,
        opts,
        icc_profile=source.icc_profile if source else None,
        exif=source.exif if source else None,
    )
    Path(path).write_bytes(enc.data)
    return enc
