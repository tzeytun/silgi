"""The silgi web API and static frontend.

Everything runs locally; the server binds to 127.0.0.1 by default and never
contacts the network.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import zipfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image as PILImage
from starlette.concurrency import run_in_threadpool

from silgi import __version__
from silgi.core import alpha, alpha_estimate, engine
from silgi.core.io import ExportOptions, ImageFile, UnsupportedImageError, encode_image, read_image
from silgi.core.progress import Progress
from silgi.core.roi import mask_bbox
from silgi.core.select import color_range, find_similar, find_text, magic_wand
from silgi.core.select.colorrange import sample_color
from silgi.core.select.morph import mask_to_png

from .jobs import JobManager
from .schemas import (
    ColorRangeRequest,
    ExportRequest,
    LocateRequest,
    RemoveOptions,
    SimilarRequest,
    TextRequest,
    WandRequest,
    ZipRequest,
)
from .sessions import Document, Store

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
MAX_UPLOAD = 512 * 1024 * 1024
PILImage.MAX_IMAGE_PIXELS = 400_000_000

_EXT = {"png": "png", "jpeg": "jpg", "webp": "webp", "tiff": "tif"}
_MIME = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp", "tiff": "image/tiff"}


def _mask_response(mask: npt.NDArray[Any], headers: dict[str, str] | None = None) -> Response:
    """Send only the mask's bounding box; ``X-Silgi-Offset`` says where it goes."""
    hard = mask > 0
    box = mask_bbox(hard)
    if box is None:
        crop, x, y = np.zeros((1, 1), bool), 0, 0
    else:
        y0, y1, x0, x1 = box
        crop, x, y = mask[y0:y1, x0:x1], x0, y0
    h = {"X-Silgi-Offset": f"{x},{y}", "Access-Control-Expose-Headers": "X-Silgi-Offset"}
    h.update(headers or {})
    return Response(mask_to_png(crop), media_type="image/png", headers=h)


def _decode_mask(
    data: bytes, shape: tuple[int, int], offset: tuple[int, int] = (0, 0)
) -> npt.NDArray[np.uint8]:
    """Read an 8-bit grayscale (or RGBA alpha) mask PNG as soft weights 0..255.

    The PNG may cover only part of the image; ``offset`` = (x, y) places it.
    """
    try:
        pil = PILImage.open(io.BytesIO(data))
        if pil.mode in ("RGBA", "LA"):
            arr = np.asarray(pil.convert("RGBA"))[:, :, 3]
        else:
            arr = np.asarray(pil.convert("L"))
    except OSError as exc:
        raise HTTPException(400, f"mask is not a valid image: {exc}") from exc
    x, y = offset
    mh, mw = arr.shape
    if arr.shape == shape and offset == (0, 0):
        return np.ascontiguousarray(arr, dtype=np.uint8)
    if x < 0 or y < 0 or y + mh > shape[0] or x + mw > shape[1]:
        raise HTTPException(
            400, f"mask ({mw}x{mh} at {x},{y}) does not fit the {shape[1]}x{shape[0]} image"
        )
    full = np.zeros(shape, np.uint8)
    full[y : y + mh, x : x + mw] = arr
    return full


def create_app() -> FastAPI:
    store = Store()
    jobs = JobManager()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        jobs.bind_loop(asyncio.get_running_loop())
        yield
        jobs.shutdown()

    app = FastAPI(
        title="silgi", version=__version__, lifespan=lifespan, docs_url="/api/docs", redoc_url=None
    )
    app.state.store = store
    app.state.jobs = jobs

    def doc_or_404(doc_id: str) -> Document:
        try:
            return store.get(doc_id)
        except KeyError:
            raise HTTPException(404, "image not found (it may have been closed)") from None

    def version_or_404(doc: Document, version: int) -> int:
        if version not in doc.versions:
            raise HTTPException(404, f"version {version} not found")
        return version

    def open_document(name: str, src: ImageFile) -> dict[str, Any]:
        doc = store.add(name, src)
        return doc.info()

    # ------------------------------------------------------------------ meta

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "version": __version__}

    @app.get("/api/methods")
    def methods() -> dict[str, Any]:
        return {
            "methods": [{"id": m, "description": engine.DESCRIPTIONS[m]} for m in engine.METHODS]
        }

    # ------------------------------------------------------------------ images

    @app.post("/api/images")
    async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
        data = await file.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, "file is too large")
        try:
            src = await run_in_threadpool(read_image, data)
        except UnsupportedImageError as exc:
            raise HTTPException(415, str(exc)) from exc
        return open_document(file.filename or "image", src)

    @app.post("/api/sample")
    async def sample() -> dict[str, Any]:
        from silgi.samples import demo_image, to_uint8

        demo = await run_in_threadpool(demo_image)
        src = ImageFile(pixels=to_uint8(demo.watermarked), format="PNG")
        info = open_document("sample-landscape.png", src)
        info["watermarks"] = [
            {"id": store.add_watermark(demo.text_mark.rgba8()), "name": "SAMPLE text"},
            {"id": store.add_watermark(demo.logo.rgba8()), "name": "Corner logo"},
        ]
        return info

    @app.get("/api/images/{doc_id}")
    def info(doc_id: str) -> dict[str, Any]:
        doc = doc_or_404(doc_id)
        data = doc.info()
        data["versions"] = [
            {"id": v.id, "parent": v.parent, "method": v.method, "reason": v.reason, "box": v.box}
            for v in doc.versions.values()
        ]
        return data

    @app.delete("/api/images/{doc_id}")
    def close(doc_id: str) -> dict[str, bool]:
        store.remove(doc_id)
        return {"ok": True}

    @app.get("/api/images/{doc_id}/v/{version}/tiles/{level}/{tx}/{ty}.webp")
    async def tile(doc_id: str, version: int, level: int, tx: int, ty: int) -> Response:
        doc = doc_or_404(doc_id)
        version_or_404(doc, version)
        if not 0 <= level <= 16:
            raise HTTPException(404, "no such level")
        try:
            data = await run_in_threadpool(doc.tile, version, level, tx, ty)
        except KeyError:
            raise HTTPException(404, "tile out of range") from None
        return Response(
            data,
            media_type="image/webp",
            headers={"Cache-Control": "private, max-age=31536000, immutable"},
        )

    @app.get("/api/images/{doc_id}/v/{version}/preview.webp")
    async def preview(doc_id: str, version: int, size: int = 1024) -> Response:
        doc = doc_or_404(doc_id)
        version_or_404(doc, version)
        data = await run_in_threadpool(doc.preview, version, min(4096, max(64, size)))
        return Response(
            data,
            media_type="image/webp",
            headers={"Cache-Control": "private, max-age=31536000, immutable"},
        )

    @app.get("/api/images/{doc_id}/v/{version}/mask.png")
    def version_mask(doc_id: str, version: int) -> Response:
        doc = doc_or_404(doc_id)
        v = doc.versions[version_or_404(doc, version)]
        if v.mask_png is None:
            raise HTTPException(404, "this version has no mask")
        x, y = v.mask_offset
        return Response(
            v.mask_png,
            media_type="image/png",
            headers={
                "X-Silgi-Offset": f"{x},{y}",
                "Access-Control-Expose-Headers": "X-Silgi-Offset",
            },
        )

    # ------------------------------------------------------------------ selection

    @app.post("/api/images/{doc_id}/select/wand")
    async def wand(doc_id: str, req: WandRequest) -> Response:
        doc = doc_or_404(doc_id)
        px = doc.pixels(version_or_404(doc, req.version))
        try:
            mask = await run_in_threadpool(
                magic_wand, px, req.x, req.y, req.tolerance, contiguous=req.contiguous
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return _mask_response(mask)

    @app.post("/api/images/{doc_id}/select/color-range")
    async def colorrange(doc_id: str, req: ColorRangeRequest) -> Response:
        doc = doc_or_404(doc_id)
        px = doc.pixels(version_or_404(doc, req.version))
        if req.color is not None:
            color = req.color
        elif req.x is not None and req.y is not None:
            if not (0 <= req.x < px.shape[1] and 0 <= req.y < px.shape[0]):
                raise HTTPException(400, "point is outside the image")
            color = sample_color(px, req.x, req.y)
        else:
            raise HTTPException(400, "give a colour or a point")
        mask = await run_in_threadpool(color_range, px, color, req.tolerance)
        return _mask_response(mask, {"X-Silgi-Color": json.dumps(color)})

    @app.post("/api/images/{doc_id}/select/text")
    async def text(doc_id: str, req: TextRequest) -> dict[str, Any]:
        doc = doc_or_404(doc_id)
        px = doc.pixels(version_or_404(doc, req.version))
        boxes = await run_in_threadpool(find_text, px)
        return {
            "boxes": [{"x": b.x, "y": b.y, "w": b.w, "h": b.h, "score": b.score} for b in boxes]
        }

    @app.post("/api/images/{doc_id}/select/similar")
    async def similar(doc_id: str, req: SimilarRequest) -> dict[str, Any]:
        doc = doc_or_404(doc_id)
        px = doc.pixels(version_or_404(doc, req.version))
        try:
            hits = await run_in_threadpool(find_similar, px, req.box, threshold=req.threshold)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {
            "hits": [
                {
                    "x": h.x,
                    "y": h.y,
                    "w": h.w,
                    "h": h.h,
                    "score": h.score,
                    "scale": h.scale,
                    "is_source": h.is_source,
                }
                for h in hits
            ]
        }

    # ------------------------------------------------------------------ watermarks

    @app.post("/api/watermarks")
    async def upload_watermark(file: UploadFile = File(...)) -> dict[str, Any]:
        try:
            src = read_image(await file.read(MAX_UPLOAD))
        except UnsupportedImageError as exc:
            raise HTTPException(415, str(exc)) from exc
        px = src.pixels
        if px.ndim != 3 or px.shape[2] != 4:
            raise HTTPException(
                400, "the watermark must be a PNG with transparency (alpha channel)"
            )
        rgba = px if px.dtype == np.uint8 else np.clip(np.rint(px / 257.0), 0, 255).astype(np.uint8)
        wid = store.add_watermark(rgba)
        return {"id": wid, "width": int(px.shape[1]), "height": int(px.shape[0])}

    @app.get("/api/watermarks/{wid}.png")
    def get_watermark(wid: str) -> Response:
        if wid not in store.watermarks:
            raise HTTPException(404, "watermark not found")
        rgba = store.watermarks[wid]
        return Response(encode_image(rgba).data, media_type="image/png")

    @app.post("/api/watermarks/estimate")
    async def estimate(files: list[UploadFile] = File(...)) -> dict[str, Any]:
        datas = [await f.read(MAX_UPLOAD) for f in files]

        def run(progress: Progress) -> dict[str, Any]:
            imgs = [read_image(d).pixels for d in datas]
            est = alpha_estimate.estimate(imgs, progress=progress)
            wid = store.add_watermark(est.rgba8())
            return {
                "watermark_id": wid,
                "box": list(est.box),
                "offsets": [list(o) for o in est.offsets],
                "single_color": est.single_color,
            }

        job = jobs.submit("estimate", run)
        return {"job_id": job.id}

    @app.post("/api/images/{doc_id}/watermark/locate")
    async def locate(doc_id: str, req: LocateRequest) -> dict[str, Any]:
        doc = doc_or_404(doc_id)
        px = doc.pixels(version_or_404(doc, req.version))
        wm = store.watermarks.get(req.watermark_id)
        if wm is None:
            raise HTTPException(404, "watermark not found")
        region = None
        if req.box is not None:
            x, y, w, h = req.box
            region = (y, y + h, x, x + w)
        try:
            p = await run_in_threadpool(alpha.find_placement, px, wm, region=region)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {
            "top": p.top,
            "left": p.left,
            "score": p.score,
            "width": int(wm.shape[1]),
            "height": int(wm.shape[0]),
        }

    # ------------------------------------------------------------------ removal jobs

    @app.post("/api/images/{doc_id}/remove")
    async def remove(
        doc_id: str,
        options: str = Form("{}"),
        mask: UploadFile | None = File(None),
        source_mask: UploadFile | None = File(None),
    ) -> dict[str, Any]:
        doc = doc_or_404(doc_id)
        try:
            opts = RemoveOptions.model_validate_json(options)
        except ValueError as exc:
            raise HTTPException(422, f"invalid options: {exc}") from exc
        version_or_404(doc, opts.version)
        shape = (doc.source.height, doc.source.width)
        mask_bytes = await mask.read() if mask is not None else None
        m = _decode_mask(mask_bytes, shape, opts.mask_offset) if mask_bytes else None
        src_m = (
            _decode_mask(await source_mask.read(), shape, opts.source_offset)
            if source_mask is not None
            else None
        )
        watermark = None
        if opts.watermark_id is not None:
            watermark = store.watermarks.get(opts.watermark_id)
            if watermark is None:
                raise HTTPException(404, "watermark not found")
        if m is None and not (opts.method in ("alpha", "auto") and watermark is not None):
            raise HTTPException(400, "a mask is required for this method")
        if m is not None and not m.any():
            raise HTTPException(400, "the mask is empty; select something first")

        def run(progress: Progress) -> dict[str, Any]:
            img = doc.pixels(opts.version)
            if m is None:
                assert watermark is not None
                result_img = alpha.remove(
                    img,
                    None,
                    watermark=watermark,
                    position=opts.watermark_position,
                    quality=opts.quality,
                    progress=progress,
                )
                method, reason = "alpha", engine.DESCRIPTIONS["alpha"]
            else:
                res = engine.remove(
                    img,
                    m,
                    opts.method,
                    quality=opts.quality,
                    watermark=watermark,
                    watermark_position=opts.watermark_position,
                    source_mask=src_m,
                    offset=opts.offset,
                    seed=opts.seed,
                    feather=opts.feather,
                    seam=opts.seam,
                    progress=progress,
                )
                result_img, method, reason = res.image, res.method, res.reason
            changed = (
                np.any(result_img != img, axis=2) if result_img.ndim == 3 else result_img != img
            )
            box = mask_bbox(changed)
            if box is None:
                return {
                    "version": opts.version,
                    "method": method,
                    "reason": reason,
                    "box": None,
                    "changed": False,
                }
            v = doc.add_version(
                opts.version,
                result_img,
                box,
                method=method,
                reason=reason,
                mask_png=mask_bytes,
                mask_offset=opts.mask_offset,
            )
            return {
                "version": v.id,
                "method": method,
                "reason": reason,
                "box": list(box),
                "changed": True,
            }

        job = jobs.submit("remove", run)
        return {"job_id": job.id}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> dict[str, Any]:
        if job_id not in jobs.jobs:
            raise HTTPException(404, "job not found")
        return jobs.jobs[job_id].snapshot()

    @app.delete("/api/jobs/{job_id}")
    def job_cancel(job_id: str) -> dict[str, Any]:
        if job_id not in jobs.jobs:
            raise HTTPException(404, "job not found")
        return jobs.cancel(job_id).snapshot()

    @app.get("/api/jobs/{job_id}/previews/{index}.webp")
    def job_preview(job_id: str, index: int) -> Response:
        job = jobs.jobs.get(job_id)
        if job is None or not 0 <= index < len(job.previews):
            raise HTTPException(404, "preview not found")
        return Response(job.previews[index], media_type="image/webp")

    @app.websocket("/api/jobs/{job_id}/events")
    async def job_events(ws: WebSocket, job_id: str) -> None:
        await ws.accept()
        job = jobs.jobs.get(job_id)
        if job is None:
            await ws.send_json({"type": "error", "error": "job not found"})
            await ws.close()
            return
        q = jobs.subscribe(job)
        try:
            while True:
                event = await q.get()
                await ws.send_json(event)
                if event["type"] in ("done", "error", "cancelled"):
                    break
                if event["type"] == "status" and event.get("status") in (
                    "done",
                    "error",
                    "cancelled",
                ):
                    break
        except WebSocketDisconnect:
            pass
        finally:
            jobs.unsubscribe(job, q)
            with contextlib.suppress(RuntimeError):
                await ws.close()

    # ------------------------------------------------------------------ export

    def _export(doc: Document, req: ExportRequest) -> tuple[bytes, list[str], str]:
        px = doc.pixels(version_or_404(doc, req.version))
        enc = encode_image(
            px,
            ExportOptions(
                format=req.format,
                quality=req.quality,
                keep_metadata=req.keep_metadata,
                keep_icc=req.keep_icc,
                webp_lossless=req.webp_lossless,
            ),
            icc_profile=doc.source.icc_profile,
            exif=doc.source.exif,
        )
        stem = Path(doc.name).stem or "image"
        return enc.data, enc.warnings, f"{stem}-silgi.{_EXT[req.format]}"

    @app.post("/api/images/{doc_id}/export")
    async def export(doc_id: str, req: ExportRequest) -> Response:
        doc = doc_or_404(doc_id)
        data, warnings, filename = await run_in_threadpool(_export, doc, req)
        return Response(
            data,
            media_type=_MIME[req.format],
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-Silgi-Warnings": json.dumps(warnings),
                "Access-Control-Expose-Headers": "X-Silgi-Warnings, Content-Disposition",
            },
        )

    @app.post("/api/export/zip")
    async def export_zip(req: ZipRequest) -> Response:
        def build() -> bytes:
            buf = io.BytesIO()
            names: set[str] = set()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
                for item in req.items:
                    doc = doc_or_404(item.image_id)
                    data, _, name = _export(
                        doc, req.options.model_copy(update={"version": item.version})
                    )
                    base, n = name, 1
                    while name in names:
                        n += 1
                        stem, ext = base.rsplit(".", 1)
                        name = f"{stem}-{n}.{ext}"
                    names.add(name)
                    zf.writestr(name, data)
            return buf.getvalue()

        data = await run_in_threadpool(build)
        return Response(
            data,
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="silgi-batch.zip"'},
        )

    @app.exception_handler(ValueError)
    async def value_error(_: Any, exc: ValueError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    # ------------------------------------------------------------------ frontend

    if (WEB_DIR / "index.html").exists():
        app.mount(
            "/assets", StaticFiles(directory=WEB_DIR / "assets", check_dir=False), name="assets"
        )

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> Response:
            target = (WEB_DIR / path).resolve()
            if path and target.is_file() and WEB_DIR in target.parents:
                return FileResponse(target)
            return FileResponse(WEB_DIR / "index.html")

    else:

        @app.get("/", include_in_schema=False)
        def no_frontend() -> HTMLResponse:
            return HTMLResponse(
                "<!doctype html><title>silgi</title>"
                "<body style='font-family:sans-serif;padding:2rem'>"
                "<h1>silgi API is running</h1><p>The web interface has not been built. Run "
                "<code>npm run build</code> in <code>frontend/</code>, or use the API at "
                "<a href='/api/docs'>/api/docs</a>.</p></body>"
            )

    return app
