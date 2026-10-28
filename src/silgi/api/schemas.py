"""Request bodies for the REST API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Quality = Literal["fast", "balanced", "best"]
MethodName = Literal["auto", "alpha", "patchmatch", "telea", "ns", "clone", "heal"]


class RemoveOptions(BaseModel):
    version: int = 0
    method: MethodName = "auto"
    quality: Quality = "balanced"
    watermark_id: str | None = None
    watermark_position: tuple[float, float] | None = None
    offset: tuple[int, int] | None = None
    seed: int = 0
    feather: float = Field(0.0, ge=0, le=64)
    seam: Literal["none", "poisson"] = "none"
    mask_offset: tuple[int, int] = (0, 0)
    source_offset: tuple[int, int] = (0, 0)


class WandRequest(BaseModel):
    version: int = 0
    x: int
    y: int
    tolerance: float = Field(0.12, ge=0, le=1)
    contiguous: bool = True


class ColorRangeRequest(BaseModel):
    version: int = 0
    x: int | None = None
    y: int | None = None
    color: tuple[float, float, float] | None = None
    tolerance: float = Field(20.0, ge=0, le=200)


class TextRequest(BaseModel):
    version: int = 0


class SimilarRequest(BaseModel):
    version: int = 0
    box: tuple[int, int, int, int]
    threshold: float = Field(0.6, ge=0.2, le=0.99)


class LocateRequest(BaseModel):
    version: int = 0
    watermark_id: str
    box: tuple[int, int, int, int] | None = None


class ExportRequest(BaseModel):
    version: int = 0
    format: Literal["png", "jpeg", "webp", "tiff"] = "png"
    quality: int = Field(95, ge=1, le=100)
    keep_metadata: bool = False
    keep_icc: bool = True
    webp_lossless: bool = True


class ZipItem(BaseModel):
    image_id: str
    version: int


class ZipRequest(BaseModel):
    items: list[ZipItem]
    options: ExportRequest = Field(default_factory=ExportRequest)
