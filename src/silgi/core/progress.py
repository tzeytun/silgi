"""Progress reporting and cooperative cancellation for long-running methods."""

from __future__ import annotations

from collections.abc import Callable

from ._types import Box, FloatImage

PreviewCallback = Callable[[FloatImage, Box], None]


class Cancelled(Exception):
    """Raised inside a method when the caller asked to stop."""


class Progress:
    """Passed to removal methods so callers can follow and cancel them.

    The default instance does nothing, so library users can ignore it.
    """

    def __init__(
        self,
        on_progress: Callable[[float, str], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        on_preview: PreviewCallback | None = None,
        *,
        box: Box | None = None,
    ) -> None:
        self._on_progress = on_progress
        self._should_cancel = should_cancel
        self._on_preview = on_preview
        self._box = box

    def report(self, fraction: float, message: str = "") -> None:
        if self._on_progress is not None:
            self._on_progress(min(max(fraction, 0.0), 1.0), message)

    def check(self) -> None:
        """Raise :class:`Cancelled` if the caller requested cancellation."""
        if self._should_cancel is not None and self._should_cancel():
            raise Cancelled()

    def preview(self, image: FloatImage) -> None:
        """Offer an intermediate result covering the current region of interest.

        ``image`` may be smaller than the region; the callback receives the
        region's box in full-image coordinates and scales as it needs.
        """
        if self._on_preview is not None and self._box is not None:
            self._on_preview(image, self._box)

    @property
    def wants_preview(self) -> bool:
        return self._on_preview is not None

    def for_box(self, box: Box) -> Progress:
        """Return a copy that tags previews with ``box``."""
        return Progress(self._on_progress, self._should_cancel, self._on_preview, box=box)


NULL_PROGRESS = Progress()
