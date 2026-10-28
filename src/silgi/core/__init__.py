"""silgi.core: the removal engine, usable as a library without any web dependencies.

>>> from silgi.core import read_image, remove
>>> img = read_image("photo.png")
>>> result = remove(img.pixels, mask, method="auto")
>>> result.method, result.reason
"""

from __future__ import annotations

from .engine import DESCRIPTIONS, METHODS, RemovalResult, remove
from .io import ExportOptions, ImageFile, encode_image, read_image, save_image
from .progress import Cancelled, Progress

__all__ = [
    "DESCRIPTIONS",
    "METHODS",
    "Cancelled",
    "ExportOptions",
    "ImageFile",
    "Progress",
    "RemovalResult",
    "encode_image",
    "read_image",
    "remove",
    "save_image",
]
