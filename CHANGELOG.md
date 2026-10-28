# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-28

First public release.

### Added

- Removal engine (`silgi.core`), usable as a library without web dependencies:
  - reverse alpha blending for known watermarks, with automatic alignment
    (gradient NCC and sub-pixel refinement) and estimation of an unknown
    watermark from several images;
  - deterministic multiscale PatchMatch completion with Wexler-style voting and
    an optional source-region restriction;
  - Telea and Navier–Stokes diffusion inpainting;
  - clone stamp and Poisson-based healing brush;
  - inward-only feathering and Poisson seam blending;
  - automatic method choice that reports what it picked and why.
- Guarantee, enforced in one place and tested: pixels outside the mask are
  bit-identical to the input.
- 8/16-bit PNG and TIFF, alpha, ICC profile and EXIF orientation preserved; PNG,
  JPEG, WebP and TIFF export with metadata off by default.
- Selection tools: brush, eraser, rectangle, ellipse, lasso, polygon lasso, magic
  wand, color range, classical text finder, find similar (multi-scale template
  matching), grow/shrink/smooth/feather, mask import/export.
- Web interface (React, TypeScript): tiled canvas with zoom, pan and pinch, mask
  overlay with marching ants, before/after slider, history with undo/redo, job
  progress with fast preview and cancel, batch queue, export dialog, light and
  dark themes, English and Turkish.
- FastAPI server with WebSocket progress; `silgi` starts it and opens the browser,
  `silgi --window` opens a desktop window.
- CLI: `silgi remove` and `silgi batch` (shared mask or logo template).
- Brand assets: the "lost dot" wordmark and app icon as outlined SVGs, PNG and
  ICO exports, and a social preview.
- Test suite: untouched-pixel checks over random masks, reverse-alpha accuracy,
  quality regression baselines, I/O round-trips, determinism, Hypothesis property
  tests for mask operations, API and CLI tests, Vitest and Playwright.

[Unreleased]: https://github.com/tzeytun/silgi/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/tzeytun/silgi/releases/tag/v0.1.0
