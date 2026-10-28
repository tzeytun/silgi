"""In-memory documents and their edit history.

A document is the decoded original plus a tree of versions. Each version
after the original stores only a *patch*: the box that changed and the pixels
inside it, together with the mask that produced it. That keeps a long history
cheap even for very large images, and every step can be undone exactly.
"""

from __future__ import annotations

import itertools
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from silgi.core._types import Box, Image
from silgi.core.color import to_display_rgb8
from silgi.core.io import ImageFile

TILE = 512


@dataclass
class Version:
    id: int
    parent: int | None
    box: Box | None = None
    patch: Image | None = None
    method: str = "original"
    reason: str = ""
    mask_png: bytes | None = None


@dataclass
class Document:
    id: str
    name: str
    source: ImageFile
    versions: dict[int, Version] = field(default_factory=dict)
    _counter: itertools.count[int] = field(default_factory=lambda: itertools.count(1))
    _pixels: OrderedDict[int, Image] = field(default_factory=OrderedDict)
    _display: OrderedDict[int, list[npt.NDArray[np.uint8]]] = field(default_factory=OrderedDict)
    _tiles: OrderedDict[tuple[int, int, int, int], bytes] = field(default_factory=OrderedDict)
    lock: threading.RLock = field(default_factory=threading.RLock)

    CACHE_VERSIONS = 2
    CACHE_TILES = 512

    def __post_init__(self) -> None:
        self.versions[0] = Version(0, None)

    # ---------------------------------------------------------------- info

    def info(self) -> dict[str, Any]:
        s = self.source
        h, w = s.height, s.width
        levels = max(1, int(np.ceil(np.log2(max(h, w) / TILE))) + 1)
        return {
            "id": self.id,
            "name": self.name,
            "width": w,
            "height": h,
            "channels": s.channels,
            "bit_depth": s.bit_depth,
            "has_alpha": s.has_alpha,
            "format": s.format,
            "has_icc": s.icc_profile is not None,
            "has_exif": s.exif is not None,
            "tile_size": TILE,
            "levels": levels,
            "notes": s.notes,
        }

    # ---------------------------------------------------------------- versions

    def chain(self, vid: int) -> list[Version]:
        if vid not in self.versions:
            raise KeyError(vid)
        out = []
        v: Version | None = self.versions[vid]
        while v is not None:
            out.append(v)
            v = self.versions[v.parent] if v.parent is not None else None
        return out[::-1]

    def pixels(self, vid: int) -> Image:
        """Full-resolution pixels of a version (do not modify the result)."""
        with self.lock:
            if vid in self._pixels:
                self._pixels.move_to_end(vid)
                return self._pixels[vid]
            chain = self.chain(vid)
            # start from the deepest cached ancestor
            start = 0
            base: Image = self.source.pixels
            for i in range(len(chain) - 1, -1, -1):
                if chain[i].id in self._pixels:
                    base, start = self._pixels[chain[i].id], i + 1
                    break
            else:
                start = 1
            out = base.copy() if start < len(chain) else base
            for v in chain[start:]:
                if v.box is not None and v.patch is not None:
                    y0, y1, x0, x1 = v.box
                    out[y0:y1, x0:x1] = v.patch
            self._remember(self._pixels, vid, out)
            return out

    def add_version(
        self,
        parent: int,
        result: Image,
        box: Box,
        *,
        method: str,
        reason: str,
        mask_png: bytes | None,
    ) -> Version:
        y0, y1, x0, x1 = box
        with self.lock:
            vid = next(self._counter)
            v = Version(
                vid,
                parent,
                box=box,
                patch=result[y0:y1, x0:x1].copy(),
                method=method,
                reason=reason,
                mask_png=mask_png,
            )
            self.versions[vid] = v
            self._remember(self._pixels, vid, result)
            return v

    def _remember(self, cache: OrderedDict[Any, Any], key: Any, value: Any) -> None:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > self.CACHE_VERSIONS:
            cache.popitem(last=False)

    # ---------------------------------------------------------------- display

    def display_level(self, vid: int, level: int) -> npt.NDArray[np.uint8]:
        """8-bit sRGB pixels at pyramid ``level`` (0 = full size)."""
        with self.lock:
            pyramid = self._display.get(vid)
            if pyramid is None:
                pyramid = [self._display_full(vid)]
                self._display[vid] = pyramid
                self._display.move_to_end(vid)
                while len(self._display) > self.CACHE_VERSIONS + 1:
                    self._display.popitem(last=False)
            while len(pyramid) <= level:
                prev = pyramid[-1]
                h, w = prev.shape[:2]
                nxt = cv2.resize(
                    prev, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA
                )
                pyramid.append(np.asarray(nxt, dtype=np.uint8))
            return pyramid[level]

    def _display_full(self, vid: int) -> npt.NDArray[np.uint8]:
        v = self.versions[vid]
        icc = self.source.icc_profile
        if (
            v.parent is not None
            and v.parent in self._display
            and v.box is not None
            and v.patch is not None
        ):
            # incremental: reuse the parent's display and convert only the patch
            out = self._display[v.parent][0].copy()
            y0, y1, x0, x1 = v.box
            out[y0:y1, x0:x1] = to_display_rgb8(v.patch, icc)
            return out
        return to_display_rgb8(self.pixels(vid), icc)

    def tile(self, vid: int, level: int, tx: int, ty: int) -> bytes:
        key = (vid, level, tx, ty)
        with self.lock:
            if key in self._tiles:
                self._tiles.move_to_end(key)
                return self._tiles[key]
        img = self.display_level(vid, level)
        h, w = img.shape[:2]
        x0, y0 = tx * TILE, ty * TILE
        if x0 >= w or y0 >= h or tx < 0 or ty < 0:
            raise KeyError(key)
        crop = img[y0 : min(h, y0 + TILE), x0 : min(w, x0 + TILE)]
        data = encode_webp(crop)
        with self.lock:
            self._tiles[key] = data
            while len(self._tiles) > self.CACHE_TILES:
                self._tiles.popitem(last=False)
        return data

    def preview(self, vid: int, max_side: int) -> bytes:
        h, w = self.source.height, self.source.width
        level = 0
        while max(h, w) / (2 ** (level + 1)) >= max_side:
            level += 1
        img = self.display_level(vid, level)
        ih, iw = img.shape[:2]
        if max(ih, iw) > max_side:
            f = max_side / max(ih, iw)
            img = np.asarray(
                cv2.resize(
                    img,
                    (max(1, round(iw * f)), max(1, round(ih * f))),
                    interpolation=cv2.INTER_AREA,
                ),
                dtype=np.uint8,
            )
        return encode_webp(img)


def encode_webp(rgb: npt.NDArray[np.uint8], quality: int = 92) -> bytes:
    """Fast display encoding (never used for exports)."""
    arr: npt.NDArray[np.uint8] = rgb
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = np.asarray(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), dtype=np.uint8)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = np.asarray(cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA), dtype=np.uint8)
    ok, buf = cv2.imencode(".webp", arr, [cv2.IMWRITE_WEBP_QUALITY, quality])
    if not ok:  # pragma: no cover
        raise RuntimeError("WebP encoding failed")
    return bytes(buf.tobytes())


class Store:
    """All open documents and uploaded watermarks of this server process."""

    MAX_DOCUMENTS = 64

    def __init__(self) -> None:
        self.documents: OrderedDict[str, Document] = OrderedDict()
        self.watermarks: dict[str, npt.NDArray[np.uint8]] = {}
        self._lock = threading.Lock()

    def add(self, name: str, source: ImageFile) -> Document:
        doc = Document(uuid.uuid4().hex[:12], name, source)
        with self._lock:
            self.documents[doc.id] = doc
            while len(self.documents) > self.MAX_DOCUMENTS:
                self.documents.popitem(last=False)
        return doc

    def get(self, doc_id: str) -> Document:
        return self.documents[doc_id]

    def remove(self, doc_id: str) -> None:
        with self._lock:
            self.documents.pop(doc_id, None)

    def add_watermark(self, rgba: npt.NDArray[np.uint8]) -> str:
        wid = uuid.uuid4().hex[:12]
        self.watermarks[wid] = rgba
        return wid
