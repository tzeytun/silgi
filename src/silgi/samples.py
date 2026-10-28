"""Procedural sample images (CC0).

Everything here is drawn in code: synthetic test scenes with known ground
truth, a landscape-like demo picture, an invented "SAMPLE" text watermark and
a made-up logo. No third-party images are used anywhere in silgi.

The generated images are dedicated to the public domain (CC0 1.0).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

F32 = npt.NDArray[np.float32]

SCENES = ("flat", "gradient", "stripes", "checker", "noise", "edges")


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def fbm(h: int, w: int, seed: int = 0, octaves: int = 5, base: int = 4) -> F32:
    """Fractal value noise in [0, 1] (smooth, stationary, deterministic)."""
    rng = _rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        cells = base * (2**o)
        grid = rng.random((cells + 3, cells + 3), dtype=np.float32)
        up = cv2.resize(grid, (w + 2 * w // cells, h + 2 * h // cells), interpolation=cv2.INTER_CUBIC)
        oy, ox = h // cells, w // cells
        out += amp * up[oy : oy + h, ox : ox + w]
        total += amp
        amp *= 0.5
    out /= total
    lo, hi = float(out.min()), float(out.max())
    return ((out - lo) / max(hi - lo, 1e-6)).astype(np.float32)


def scene(name: str, h: int = 256, w: int = 256, seed: int = 0) -> F32:
    """A synthetic RGB scene in [0, 1] used for quality tests."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rng = _rng(seed)
    c1 = rng.uniform(0.15, 0.85, 3).astype(np.float32)
    c2 = rng.uniform(0.15, 0.85, 3).astype(np.float32)
    if name == "flat":
        img = np.broadcast_to(c1, (h, w, 3)).copy()
    elif name == "gradient":
        t = ((xx / max(w - 1, 1)) * 0.6 + (yy / max(h - 1, 1)) * 0.4)[..., None]
        img = c1 * (1 - t) + c2 * t
    elif name == "stripes":
        t = (np.sin(2 * np.pi * xx / 16.0) * 0.5 + 0.5)[..., None]
        img = c1 * (1 - t) + c2 * t
    elif name == "checker":
        t = (((xx // 16) + (yy // 16)) % 2)[..., None]
        img = c1 * (1 - t) + c2 * t
    elif name == "noise":
        t = fbm(h, w, seed=seed + 1, octaves=4, base=8)[..., None]
        img = c1 * (1 - t) + c2 * t
    elif name == "edges":
        t = ((yy - 0.35 * xx) > h * 0.3).astype(np.float32)[..., None]
        img = c1 * (1 - t) + c2 * t
        img[int(h * 0.7) : int(h * 0.7) + 3, :] = c2 * 0.4
    else:
        raise ValueError(f"unknown scene {name!r}; choose from {SCENES}")
    return np.clip(img, 0, 1).astype(np.float32)


def landscape(h: int = 900, w: int = 1350, seed: int = 7) -> F32:
    """A calm, photo-like landscape: sky, sun, mountains, lake, meadow."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    v = yy / h
    sky_top = np.array([0.36, 0.52, 0.74], np.float32)
    sky_low = np.array([0.95, 0.80, 0.66], np.float32)
    horizon = 0.55
    t = np.clip(v / horizon, 0, 1)[..., None] ** 1.3
    img = sky_top * (1 - t) + sky_low * t

    # soft clouds
    clouds = fbm(h, w, seed=seed, octaves=6, base=3)
    cmask = np.clip((clouds - 0.55) * 3.0, 0, 1) * np.clip(1.2 - v / horizon, 0, 1)
    img = img * (1 - cmask[..., None] * 0.55) + np.float32(0.98) * cmask[..., None] * 0.55

    # sun with glow
    sx, sy, sr = w * 0.68, h * 0.40, h * 0.055
    d = np.sqrt((xx - sx) ** 2 + (yy - sy) ** 2)
    glow = np.exp(-((d / (sr * 5)) ** 2))[..., None] * 0.35
    img = img + glow * np.array([1.0, 0.85, 0.6], np.float32)
    disc = np.clip((sr - d) / 2.0, 0, 1)[..., None]
    img = img * (1 - disc) + np.array([1.0, 0.95, 0.85], np.float32) * disc

    def ridge(seed_: int, base_y: float, amp: float) -> F32:
        n = fbm(1, w, seed=seed_, octaves=6, base=4)[0]
        return (base_y - amp * n) * h

    def layer(top: F32, color: npt.NDArray[np.float32], tex_seed: int, tex: float) -> None:
        nonlocal img
        cover = np.clip(yy - top[None, :] + 0.5, 0, 1)[..., None]
        shade = fbm(h, w, seed=tex_seed, octaves=6, base=6)[..., None]
        col = color * (1 - tex + tex * 2 * shade)
        img = img * (1 - cover) + col * cover

    layer(ridge(seed + 11, 0.52, 0.22), np.array([0.47, 0.50, 0.62], np.float32), seed + 12, 0.10)
    layer(ridge(seed + 21, 0.58, 0.12), np.array([0.30, 0.36, 0.40], np.float32), seed + 22, 0.18)

    # lake with reflection and ripples
    lake_top = int(h * 0.62)
    lake_bot = int(h * 0.80)
    mirror = img[2 * lake_top - lake_bot : lake_top][::-1].copy()
    rip = fbm(lake_bot - lake_top, w, seed=seed + 31, octaves=5, base=10)
    ripple = (np.sin(np.arange(lake_bot - lake_top, dtype=np.float32)[:, None] * 0.9 + rip * 8) * 0.04)
    water = mirror * 0.72 + np.array([0.10, 0.18, 0.26], np.float32) * 0.28 + ripple[..., None]
    img[lake_top:lake_bot] = water

    # meadow
    meadow_top = ridge(seed + 41, 0.83, 0.05)
    cover = np.clip(yy - meadow_top[None, :] + 0.5, 0, 1)[..., None]
    grass = fbm(h, w, seed=seed + 42, octaves=7, base=12)[..., None]
    blades = (np.sin(xx * 1.7 + fbm(h, w, seed=seed + 43, octaves=3, base=20) * 30) * 0.5 + 0.5)
    g = np.array([0.30, 0.42, 0.20], np.float32) * (0.75 + 0.5 * grass) * (
        0.9 + 0.1 * blades[..., None]
    )
    g = g * (0.85 + 0.25 * (v[..., None] - 0.8))
    img = img * (1 - cover) + g * cover

    # fine film grain so patches are not perfectly flat
    grain = _rng(seed + 99).normal(0, 0.008, (h, w, 1)).astype(np.float32)
    return np.clip(img + grain, 0, 1).astype(np.float32)


# ------------------------------------------------------------------ watermarks


@dataclass
class Watermark:
    """A watermark as colour ``rgb`` (HxWx3) plus coverage ``alpha`` (HxW), both float."""

    rgb: F32
    alpha: F32

    def rgba8(self) -> npt.NDArray[np.uint8]:
        a = np.dstack([self.rgb, self.alpha])
        return np.clip(np.rint(a * 255), 0, 255).astype(np.uint8)

    def rgba_float(self) -> F32:
        return np.dstack([self.rgb, self.alpha]).astype(np.float32)


def _text_alpha(text: str, height_px: int, thickness: int, font: int) -> F32:
    scale = cv2.getFontScaleFromHeight(font, height_px, thickness)
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    pad = thickness + 4
    canvas = np.zeros((th + base + 2 * pad, tw + 2 * pad), np.uint8)
    cv2.putText(canvas, text, (pad, pad + th), font, scale, 255, thickness, cv2.LINE_AA)
    return canvas.astype(np.float32) / 255.0


def text_watermark(
    text: str = "SAMPLE",
    height_px: int = 120,
    opacity: float = 0.45,
    color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    thickness: int | None = None,
) -> Watermark:
    """A semi-transparent text watermark drawn with OpenCV's Hershey font."""
    thick = thickness or max(2, height_px // 9)
    alpha = _text_alpha(text, height_px, thick, cv2.FONT_HERSHEY_DUPLEX) * opacity
    rgb = np.broadcast_to(np.array(color, np.float32), (*alpha.shape, 3)).copy()
    return Watermark(rgb, alpha.astype(np.float32))


def fake_logo(size: int = 160, opacity: float = 0.85) -> Watermark:
    """An invented logo: a ring with a diamond and a short wordmark below."""
    s = size * 4  # supersample for smooth edges
    ring = np.zeros((s, s), np.uint8)
    c = s // 2
    cv2.circle(ring, (c, int(s * 0.40)), int(s * 0.30), 255, int(s * 0.07), cv2.LINE_AA)
    pts = np.array(
        [[c, int(s * 0.22)], [int(s * 0.62), int(s * 0.40)], [c, int(s * 0.58)], [int(s * 0.38), int(s * 0.40)]],
        np.int32,
    )
    cv2.fillConvexPoly(ring, pts, 255, cv2.LINE_AA)
    word = _text_alpha("SAMPLE CO", int(s * 0.12), max(2, s // 60), cv2.FONT_HERSHEY_SIMPLEX)
    wy, wx = int(s * 0.80), (s - word.shape[1]) // 2
    if wx >= 0 and wy + word.shape[0] <= s:
        ring[wy : wy + word.shape[0], wx : wx + word.shape[1]] = np.maximum(
            ring[wy : wy + word.shape[0], wx : wx + word.shape[1]], (word * 255).astype(np.uint8)
        )
    alpha = cv2.resize(ring, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    rgb = np.broadcast_to(np.array([1.0, 1.0, 1.0], np.float32), (size, size, 3)).copy()
    return Watermark(rgb, (alpha * opacity).astype(np.float32))


def composite(image: F32, wm: Watermark, top: int, left: int) -> F32:
    """Blend ``wm`` over ``image`` at (top, left): I = α·W + (1 − α)·J."""
    out = image.copy()
    h, w = wm.alpha.shape
    a = wm.alpha[..., None]
    region = out[top : top + h, left : left + w, :3]
    out[top : top + h, left : left + w, :3] = a * wm.rgb + (1 - a) * region
    return out


def to_uint8(image: F32) -> npt.NDArray[np.uint8]:
    return np.clip(np.rint(image * 255), 0, 255).astype(np.uint8)


def to_uint16(image: F32) -> npt.NDArray[np.uint16]:
    return np.clip(np.rint(image * 65535), 0, 65535).astype(np.uint16)


@dataclass
class DemoImage:
    clean: F32
    watermarked: F32
    text_mark: Watermark
    text_pos: tuple[int, int]
    logo: Watermark
    logo_pos: tuple[int, int]


def demo_image(h: int = 900, w: int = 1350, seed: int = 7) -> DemoImage:
    """The landscape with a large "SAMPLE" text watermark, a corner logo and a caption."""
    clean = landscape(h, w, seed)
    text = text_watermark("SAMPLE", height_px=int(h * 0.16), opacity=0.42)
    th, tw = text.alpha.shape
    text_pos = ((h - th) // 2 - int(h * 0.05), (w - tw) // 2)
    logo = fake_logo(int(h * 0.15), opacity=0.9)
    lh, lw = logo.alpha.shape
    logo_pos = (h - lh - int(h * 0.04), w - lw - int(h * 0.04))
    caption = text_watermark("(c) sample photo", height_px=max(12, h // 45), opacity=0.95, thickness=2)
    out = composite(clean, text, *text_pos)
    out = composite(out, logo, *logo_pos)
    out = composite(out, caption, h - caption.alpha.shape[0] - int(h * 0.03), int(h * 0.04))
    return DemoImage(clean, out, text, text_pos, logo, logo_pos)


def write_samples(out_dir: str | Path) -> list[Path]:
    """Write the full CC0 sample set used by the docs, demo and tests."""
    from PIL import Image as PILImage

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def save(name: str, arr: npt.NDArray[np.uint8]) -> None:
        p = out / name
        PILImage.fromarray(arr).save(p)
        written.append(p)

    demo = demo_image()
    save("landscape-watermarked.png", to_uint8(demo.watermarked))
    save("landscape-clean.png", to_uint8(demo.clean))
    save("watermark-sample-text.png", demo.text_mark.rgba8())
    save("watermark-logo.png", demo.logo.rgba8())

    # a tiled watermark for "find similar"
    tiled = landscape(800, 1200, seed=3)
    small = fake_logo(90, opacity=0.6)
    for ty in range(60, 800 - 90, 250):
        for tx in range(80, 1200 - 90, 300):
            tiled = composite(tiled, small, ty, tx)
    save("tiled-logos.png", to_uint8(tiled))

    # multi-image set for watermark estimation (same watermark, same place)
    multi = out / "multi"
    multi.mkdir(exist_ok=True)
    mark = text_watermark("SAMPLE", height_px=70, opacity=0.5)
    for i in range(12):
        bg = landscape(360, 540, seed=100 + i)
        img = composite(bg, mark, 140, 120)
        p = multi / f"photo-{i:02d}.png"
        PILImage.fromarray(to_uint8(img)).save(p)
        written.append(p)

    for name in SCENES:
        save(f"scene-{name}.png", to_uint8(scene(name, 256, 256, seed=1)))

    (out / "LICENSE.txt").write_text(
        "The images in this folder were generated by silgi's scripts/make_samples.py.\n"
        "They are dedicated to the public domain under CC0 1.0 Universal:\n"
        "https://creativecommons.org/publicdomain/zero/1.0/\n",
        encoding="utf-8",
    )
    return written
