"""Background jobs with progress, previews and cancellation.

Removal runs on a single worker thread: the heavy kernels release the GIL, so
the server stays responsive, and Numba's default threading layer must not be
entered from two threads at once. Jobs queue up in submission order.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal

import cv2
import numpy as np

from silgi.core._types import Box, FloatImage
from silgi.core.color import to_display_rgb8
from silgi.core.progress import Cancelled, Progress

from .sessions import encode_webp

Status = Literal["queued", "running", "done", "error", "cancelled"]
PREVIEW_MAX = 1024


@dataclass
class Job:
    id: str
    kind: str
    status: Status = "queued"
    fraction: float = 0.0
    message: str = ""
    result: dict[str, Any] | None = None
    error: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    previews: list[bytes] = field(default_factory=list)
    subscribers: list[asyncio.Queue[dict[str, Any]]] = field(default_factory=list)
    created: float = field(default_factory=time.monotonic)

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "fraction": self.fraction,
            "message": self.message,
            "result": self.result,
            "error": self.error,
        }


class JobManager:
    MAX_JOBS = 256

    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="silgi-job")
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def shutdown(self) -> None:
        for job in self.jobs.values():
            job.cancel_event.set()
        self._executor.shutdown(wait=False, cancel_futures=True)

    # ---------------------------------------------------------------- events

    def _publish(self, job: Job, event: dict[str, Any]) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        for q in list(job.subscribers):
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(q.put_nowait, event)

    def subscribe(self, job: Job) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        job.subscribers.append(q)
        q.put_nowait({"type": "status", **job.snapshot()})
        return q

    def unsubscribe(self, job: Job, q: asyncio.Queue[dict[str, Any]]) -> None:
        with contextlib.suppress(ValueError):
            job.subscribers.remove(q)

    # ---------------------------------------------------------------- running

    def submit(self, kind: str, fn: Callable[[Progress], dict[str, Any]]) -> Job:
        job = Job(uuid.uuid4().hex[:12], kind)
        with self._lock:
            self.jobs[job.id] = job
            if len(self.jobs) > self.MAX_JOBS:
                finished = [
                    j for j in self.jobs.values() if j.status in ("done", "error", "cancelled")
                ]
                for old in sorted(finished, key=lambda j: j.created)[
                    : len(self.jobs) - self.MAX_JOBS
                ]:
                    self.jobs.pop(old.id, None)
        self._executor.submit(self._run, job, fn)
        return job

    def _run(self, job: Job, fn: Callable[[Progress], dict[str, Any]]) -> None:
        if job.cancel_event.is_set():
            job.status = "cancelled"
            self._publish(job, {"type": "cancelled", **job.snapshot()})
            return
        job.status = "running"
        self._publish(job, {"type": "status", **job.snapshot()})
        last = [0.0]

        def on_progress(fraction: float, message: str) -> None:
            job.fraction, job.message = fraction, message
            now = time.monotonic()
            if now - last[0] >= 0.1 or fraction >= 1.0:
                last[0] = now
                self._publish(job, {"type": "progress", "fraction": fraction, "message": message})

        def on_preview(image: FloatImage, box: Box) -> None:
            data = _preview_webp(image)
            job.previews.append(data)
            idx = len(job.previews) - 1
            self._publish(job, {"type": "preview", "index": idx, "box": list(box)})

        progress = Progress(on_progress, job.cancel_event.is_set, on_preview)
        try:
            job.result = fn(progress)
            job.status = "done"
            job.fraction = 1.0
            self._publish(job, {"type": "done", **job.snapshot()})
        except Cancelled:
            job.status = "cancelled"
            self._publish(job, {"type": "cancelled", **job.snapshot()})
        except Exception as exc:
            job.status = "error"
            job.error = str(exc) or exc.__class__.__name__
            traceback.print_exc()
            self._publish(job, {"type": "error", **job.snapshot()})

    def cancel(self, job_id: str) -> Job:
        job = self.jobs[job_id]
        job.cancel_event.set()
        return job


def _preview_webp(image: FloatImage) -> bytes:
    rgb = to_display_rgb8(np.clip(np.rint(image * 255), 0, 255).astype(np.uint8))
    h, w = rgb.shape[:2]
    if max(h, w) > PREVIEW_MAX:
        f = PREVIEW_MAX / max(h, w)
        rgb = np.asarray(
            cv2.resize(rgb, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA), np.uint8
        )
    return encode_webp(rgb, quality=80)
