# Architecture

```
┌───────────────────────────┐   HTTP + WebSocket   ┌────────────────────────────────┐
│ frontend/ (React + TS)    │ ───────────────────▶ │ silgi.api (FastAPI)            │
│ canvas editor, Zustand    │ ◀─────────────────── │ sessions · jobs · tiles        │
└───────────────────────────┘   tiles, masks, PNG  └───────────────┬────────────────┘
                                                                    │ plain Python calls
                                                    ┌───────────────▼────────────────┐
                                                    │ silgi.core (NumPy, OpenCV,     │
                                                    │ SciPy, Numba) — no web imports │
                                                    └────────────────────────────────┘
```

## Packages

| Path | Role |
|---|---|
| `src/silgi/core/` | The removal engine. Usable as a library; a test asserts that importing it does not load `fastapi`. |
| `src/silgi/core/select/` | Selection tools: wand, colour range, text finder, find similar, shapes, mask operations. |
| `src/silgi/api/` | FastAPI app, in-memory documents with history, the job queue, request schemas. |
| `src/silgi/cli.py` | `silgi`, `silgi remove`, `silgi batch`. |
| `src/silgi/web/` | The built frontend (generated, not committed). |
| `frontend/` | React + TypeScript + Vite app. |

## The engine

Every removal method has the shape `remove(image, mask, **params) -> image`.
They all go through `roi.run_in_roi`, which

1. normalises the mask (bool, 8/16-bit or float, hard or soft),
2. crops the mask's bounding box plus a margin,
3. converts the crop to float32 in `[0, 1]`,
4. calls the method's `fill(roi, mask, box)`,
5. writes the result back **only inside the mask**, into a copy of the original.

This is what makes "pixels outside the mask are bit-identical" a property of one
function instead of a promise every method has to keep. `engine.remove` is the
single entry point used by the API and the CLI; it also applies optional edge
finishing (also confined to the mask).

Methods report progress and watch for cancellation through a `Progress` object,
and may offer low-resolution intermediate results (PatchMatch does, at every
pyramid level) so the UI can show a fast preview before the final result.

## The server

* **Documents.** An uploaded image is decoded once and kept in memory. Each edit
  creates a *version* that stores only a patch (the changed box and its pixels,
  plus the mask that produced it), so a long history stays cheap even for very
  large images. Any version can be reconstructed exactly.
* **Tiles.** The browser never receives the full image. The server converts a
  version to an 8-bit sRGB display pyramid (honouring the ICC profile) and serves
  512 px WebP tiles at `/api/images/{id}/v/{version}/tiles/{level}/{x}/{y}.webp`.
  Full-precision pixels stay on the server and are only used for processing and
  export.
* **Jobs.** Removal runs on a single worker thread. The heavy kernels release the
  GIL, so the server stays responsive, and Numba's threading layer is never
  entered from two threads at once. Progress and previews are pushed over a
  WebSocket (`/api/jobs/{id}/events`); `DELETE /api/jobs/{id}` cancels
  cooperatively.
* **Masks.** Selection endpoints return only the mask's bounding box with an
  `X-Silgi-Offset` header; removal requests accept a cropped mask with
  `mask_offset`. Large images therefore never need full-size mask uploads.
* **Offline.** The server binds to `127.0.0.1`, the frontend bundles its fonts, and
  the test suite runs with outbound sockets disabled (`pytest-socket`).

## The frontend

* **Viewport** (`frontend/src/canvas/Viewport.tsx`): a single canvas that draws
  server tiles for the visible area at the right pyramid level, over a
  coarsest-level base layer so panning never shows holes. Zoom is anchored at the
  pointer; pinch and space-drag pan are supported.
* **Mask layer** (`maskLayer.ts`): the selection lives in an alpha canvas
  (downscaled above 4096 px on the long side; exported masks are scaled back up).
  Shape tools draw into a scratch layer that is composited with the base under the
  selected mode (replace, add, subtract, intersect).
* **Undo.** Each undo entry stores the cropped mask snapshot plus the current
  version and path, under a memory budget. Removals are server-side versions, so
  undoing one is just switching versions.
* **State** is a single Zustand store; strings go through i18next (English and
  Turkish, with a test that keeps their keys and placeholders identical).

## Determinism and tests

Same input, mask and settings give the same bytes. PatchMatch is seeded and
thread-count independent; there is no other randomness. The test suite checks
this by encoding results twice and comparing bytes. See
[CONTRIBUTING.md](../CONTRIBUTING.md) for how to run everything.
