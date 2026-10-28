"""One entry point for every removal method."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from . import alpha, auto, blend, diffusion, healing, patchmatch
from ._types import BoolMask, Box, FloatImage, Image, MaskLike
from .progress import NULL_PROGRESS, Progress
from .roi import normalize_mask, run_in_roi

Method = Literal["auto", "alpha", "patchmatch", "telea", "ns", "clone", "heal"]
METHODS: tuple[str, ...] = ("auto", "alpha", "patchmatch", "telea", "ns", "clone", "heal")

#: One-line plain-English descriptions (also shown in the UI).
DESCRIPTIONS: dict[str, str] = {
    "auto": "Picks the best method for your selection and tells you why.",
    "alpha": "Undoes a known semi-transparent watermark and recovers the real pixels underneath.",
    "patchmatch": "Rebuilds the area with texture copied from its surroundings.",
    "telea": "Smoothly fills thin strokes from the edges inward. Fastest.",
    "ns": "Fluid-style smooth fill that follows edges into the hole.",
    "clone": "Copies pixels from another spot exactly.",
    "heal": "Copies texture from another spot and matches the local lighting.",
}


@dataclass
class RemovalResult:
    image: Image
    method: str
    reason: str
    notes: list[str] = field(default_factory=list)


def _finish(
    original: Image, result: Image, mask: MaskLike, feather: float, seam: blend.SeamMode
) -> Image:
    if feather <= 0 and seam == "none":
        return result
    hard, _ = normalize_mask(mask, original.shape)

    def _fill(roi: FloatImage, m: BoolMask, box: Box) -> FloatImage:
        from .color import to_float

        filled = to_float(result[box[0] : box[1], box[2] : box[3]])
        return blend.finish(roi, filled, m, feather_px=feather, seam_mode=seam)

    return run_in_roi(original, hard, _fill, margin=1)


def remove(
    image: Image,
    mask: MaskLike,
    method: Method = "auto",
    *,
    quality: patchmatch.Quality = "balanced",
    watermark: Any | None = None,
    watermark_position: tuple[float, float] | None = None,
    source_mask: MaskLike | None = None,
    offset: tuple[int, int] | None = None,
    seed: int = 0,
    feather: float = 0.0,
    seam: blend.SeamMode = "none",
    progress: Progress = NULL_PROGRESS,
) -> RemovalResult:
    """Remove the masked content with ``method``.

    Pixels outside ``mask`` are bit-identical in the result. ``feather`` and
    ``seam`` apply optional edge finishing, strictly inside the mask.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    reason = DESCRIPTIONS[method]
    notes: list[str] = []
    if method == "auto":
        decision = auto.choose(image, mask, has_watermark=watermark is not None)
        if decision.method == "none":
            return RemovalResult(image.copy(), "none", decision.reason)
        method = decision.method  # type: ignore[assignment]
        reason = decision.reason

    if method == "alpha":
        if watermark is None:
            raise ValueError("the alpha method needs a watermark image")
        out = alpha.remove(
            image,
            mask,
            watermark=watermark,
            position=watermark_position,
            quality=quality,
            progress=progress,
        )
    elif method == "patchmatch":
        out = patchmatch.remove(
            image, mask, quality=quality, seed=seed, source_mask=source_mask, progress=progress
        )
    elif method in ("telea", "ns"):
        out = diffusion.remove(image, mask, method=method)
    else:
        if offset is None:
            raise ValueError(f"the {method} method needs a source offset")
        out = healing.remove(image, mask, offset=offset, mode=method)  # type: ignore[arg-type]

    out = _finish(image, out, mask, feather, seam)
    progress.report(1.0, "done")
    return RemovalResult(out, str(method), reason, notes)
