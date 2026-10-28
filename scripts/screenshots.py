"""Regenerate the README images and the demo GIF.

Builds the frontend, starts silgi on a free port, loads the generated CC0 samples
and drives the real UI with Playwright (1440x900 window, 2x device pixel ratio).

    pip install -e ".[dev]" && playwright install chromium
    python scripts/screenshots.py            # everything
    python scripts/screenshots.py --skip-build --only hero editor
    python scripts/screenshots.py --no-gif

Outputs go to docs/screenshots/ and docs/demo.gif. Re-run before each release.
"""

from __future__ import annotations

import argparse
import base64
import io
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "docs" / "screenshots"
GIF = ROOT / "docs" / "demo.gif"
SAMPLES = ROOT / "samples"
WIDTH, HEIGHT = 1440, 900
GIF_LIMIT = 8 * 1024 * 1024

ALL_SCENES = ["hero", "editor", "dark", "tools", "batch", "export"]


# ------------------------------------------------------------------ helpers


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def start_server(port: int) -> subprocess.Popen[bytes]:
    proc = subprocess.Popen(
        [sys.executable, "-m", "silgi", "--no-browser", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}/api/health"
    for _ in range(100):
        try:
            urllib.request.urlopen(url, timeout=1)
            return proc
        except OSError:
            time.sleep(0.2)
    proc.kill()
    raise SystemExit("server did not start")


def build_frontend() -> None:
    npm = shutil.which("npm")
    if npm is None:
        raise SystemExit("npm not found; install Node.js or use --skip-build")
    subprocess.run([npm, "ci"], cwd=ROOT / "frontend", check=True)
    subprocess.run([npm, "run", "build"], cwd=ROOT / "frontend", check=True)


def canvas_geometry(
    page: Page, image_w: int, image_h: int
) -> Callable[[float, float], tuple[float, float]]:
    """Map image pixels to page coordinates, mirroring the viewport's fit-to-window."""
    box = page.get_by_test_id("canvas").bounding_box()
    assert box is not None
    zoom = min((box["width"] - 64) / image_w, (box["height"] - 64) / image_h, 1.0)
    ox = box["x"] + (box["width"] - image_w * zoom) / 2
    oy = box["y"] + (box["height"] - image_h * zoom) / 2
    return lambda x, y: (ox + x * zoom, oy + y * zoom)


def drag(page: Page, a: tuple[float, float], b: tuple[float, float], steps: int = 14) -> None:
    page.mouse.move(*a)
    page.mouse.down()
    page.mouse.move(*b, steps=steps)
    page.mouse.up()


def open_sample(page: Page, base: str) -> None:
    page.goto(base)
    page.get_by_test_id("sample").click()
    page.get_by_test_id("filename").wait_for()
    page.wait_for_timeout(1200)


def wait_job(page: Page) -> None:
    page.get_by_test_id("job-bar").wait_for(state="visible", timeout=10_000)
    page.get_by_test_id("job-bar").wait_for(state="hidden", timeout=120_000)
    page.wait_for_timeout(700)


def new_page(browser, base: str, *, dark: bool = False, record: Path | None = None):  # type: ignore[no-untyped-def]
    kwargs: dict = dict(
        viewport={"width": WIDTH, "height": HEIGHT},
        device_scale_factor=2,
        locale="en-US",
        color_scheme="dark" if dark else "light",
    )
    if record is not None:
        kwargs["device_scale_factor"] = 1
        kwargs["record_video_dir"] = str(record)
        kwargs["record_video_size"] = {"width": 960, "height": 600}
    ctx = browser.new_context(**kwargs)
    return ctx, ctx.new_page()


# ------------------------------------------------------------------ scenes


def scene_hero(browser, base: str) -> None:  # type: ignore[no-untyped-def]
    """Before/after made with the real engine from the generated sample."""
    from silgi.core import engine
    from silgi.core.alpha import remove as alpha_remove
    from silgi.samples import demo_image, to_uint8

    d = demo_image()
    before = to_uint8(d.watermarked)
    h, w = before.shape[:2]
    # both known watermarks are undone by reverse alpha; only the near-opaque
    # parts of the logo are inpainted
    after = alpha_remove(before, None, watermark=d.text_mark)
    after = alpha_remove(after, None, watermark=d.logo, quality="best")
    cm = np.zeros((h, w), bool)
    cm[h - 70 : h - 20, 30:270] = True
    after = engine.remove(after, cm, "patchmatch", quality="best").image

    def uri(a: np.ndarray) -> str:
        buf = io.BytesIO()
        Image.fromarray(a).save(buf, "JPEG", quality=92)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    font = (
        ROOT / "frontend/node_modules/@fontsource/inter/files/inter-latin-600-normal.woff2"
    ).as_uri()
    html = f"""<html><head><style>
@font-face {{ font-family: Inter; src: url('{font}') format('woff2'); font-weight: 600; }}
body {{ margin:0; background:#fff; font-family:Inter,sans-serif; }}
.row {{ display:flex; gap:0; width:1600px; }}
.cell {{ position:relative; width:800px; height:533px; overflow:hidden; }}
img {{ width:800px; height:533px; display:block; }}
.tag {{ position:absolute; left:16px; top:16px; padding:6px 12px; border-radius:8px; font-size:15px;
  font-weight:600; background:#15171c; color:#fff; }}
.after .tag {{ background:#ef5b7b; color:#15171c; }}
.cut {{ position:absolute; right:0; top:0; bottom:0; width:3px; background:#fff; }}
</style></head><body><div class="row">
<div class="cell"><img src="{uri(before)}"><span class="tag">Before</span><span class="cut"></span></div>
<div class="cell after"><img src="{uri(after)}"><span class="tag">After</span></div>
</div></body></html>"""
    tmp = Path(tempfile.mkdtemp()) / "hero.html"
    tmp.write_text(html, encoding="utf-8")
    ctx = browser.new_context(viewport={"width": 1600, "height": 533}, device_scale_factor=1)
    page = ctx.new_page()
    page.goto(tmp.as_uri())
    page.wait_for_timeout(500)
    page.screenshot(path=str(SHOTS / "before-after.png"))
    ctx.close()


def scene_editor(browser, base: str, dark: bool = False) -> None:  # type: ignore[no-untyped-def]
    ctx, page = new_page(browser, base, dark=dark)
    open_sample(page, base)
    to_page = canvas_geometry(page, 1350, 900)
    if dark:
        # main editor in dark: text watermark removed, compare slider open
        page.get_by_label("SAMPLE text").check()
        page.get_by_test_id("remove").click()
        wait_job(page)
        page.keyboard.press("c")
        page.evaluate("document.querySelector('aside').scrollTop = 0")
        page.wait_for_timeout(500)
    else:
        # a selection around the corner logo shows the overlay and the ants
        page.get_by_test_id("tool-rect").click()
        drag(page, to_page(1150, 690), to_page(1325, 880))
        page.wait_for_timeout(400)
    page.screenshot(path=str(SHOTS / ("editor-dark.png" if dark else "editor-light.png")))
    ctx.close()


def scene_tools(browser, base: str) -> None:  # type: ignore[no-untyped-def]
    ctx, page = new_page(browser, base)
    page.goto(base)
    page.get_by_test_id("empty-file-input").set_input_files(str(SAMPLES / "tiled-logos.png"))
    page.get_by_test_id("filename").wait_for()
    page.wait_for_timeout(1200)
    to_page = canvas_geometry(page, 1200, 800)
    page.get_by_test_id("tool-similar").click()
    drag(page, to_page(70, 50), to_page(180, 160))
    page.wait_for_function("() => document.querySelectorAll('li').length > 3", timeout=15_000)
    page.wait_for_timeout(600)
    page.screenshot(path=str(SHOTS / "selection-tools.png"))
    ctx.close()


def scene_batch(browser, base: str) -> None:  # type: ignore[no-untyped-def]
    ctx, page = new_page(browser, base)
    files = sorted((SAMPLES / "multi").glob("photo-*.png"))
    page.goto(base)
    page.get_by_test_id("empty-file-input").set_input_files(str(files[0]))
    page.get_by_test_id("filename").wait_for()
    # estimate the shared watermark from the whole set
    with page.expect_file_chooser() as fc:
        page.get_by_role("button", name="Estimate from several images").click()
    fc.value.set_files([str(f) for f in files])
    page.get_by_text("Estimated watermark").first.wait_for(timeout=60_000)
    page.get_by_test_id("batch-toggle").click()
    page.get_by_test_id("batch-view").wait_for()
    with page.expect_file_chooser() as fc2:
        page.get_by_role("button", name="Add images").click()
    fc2.value.set_files([str(f) for f in files[:8]])
    page.wait_for_function(
        "() => document.querySelectorAll('[data-testid=batch-item] img').length >= 8",
        timeout=30_000,
    )
    page.get_by_role("button", name="Process all").click()
    page.wait_for_function(
        "() => [...document.querySelectorAll('[data-testid=batch-item]')].every(e => e.textContent.includes('Done'))",
        timeout=120_000,
    )
    page.wait_for_timeout(900)
    page.screenshot(path=str(SHOTS / "batch.png"))
    ctx.close()


def scene_export(browser, base: str) -> None:  # type: ignore[no-untyped-def]
    ctx, page = new_page(browser, base)
    open_sample(page, base)
    page.get_by_label("SAMPLE text").check()
    page.get_by_test_id("remove").click()
    wait_job(page)
    page.get_by_test_id("export").click()
    page.get_by_test_id("export-dialog").wait_for()
    page.wait_for_timeout(500)
    page.screenshot(path=str(SHOTS / "export-dialog.png"))
    ctx.close()


def make_gif(browser, base: str) -> None:  # type: ignore[no-untyped-def]
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("ffmpeg not found; skipping the demo GIF")
        return
    vid_dir = Path(tempfile.mkdtemp())
    ctx, page = new_page(browser, base, record=vid_dir)
    page.goto(base)
    page.wait_for_timeout(900)
    page.get_by_test_id("sample").click()
    page.get_by_test_id("filename").wait_for()
    page.wait_for_timeout(1400)
    to_page = canvas_geometry(page, 1350, 900)
    page.get_by_test_id("tool-rect").click()
    page.mouse.move(*to_page(320, 345))
    page.wait_for_timeout(300)
    drag(page, to_page(320, 345), to_page(1015, 505), steps=28)
    page.wait_for_timeout(900)
    page.get_by_test_id("remove").click()
    wait_job(page)
    page.wait_for_timeout(700)
    page.keyboard.press("c")
    page.wait_for_timeout(500)
    box = page.get_by_test_id("canvas").bounding_box()
    assert box is not None
    mid = box["y"] + box["height"] / 2
    sx = box["x"] + box["width"] * 0.5
    page.mouse.move(sx, mid)
    page.mouse.down()
    for frac in (0.5, 0.65, 0.8, 0.9, 0.7, 0.45, 0.25, 0.1, 0.3, 0.55):
        page.mouse.move(box["x"] + box["width"] * frac, mid, steps=14)
        page.wait_for_timeout(120)
    page.mouse.up()
    page.wait_for_timeout(1000)
    video = page.video
    ctx.close()
    assert video is not None
    src = Path(video.path())
    palette = vid_dir / "palette.png"
    base_filter = "fps=10,scale=800:-1:flags=lanczos"
    for colors in (128, 96, 64):
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-v",
                "error",
                "-i",
                str(src),
                "-vf",
                f"{base_filter},palettegen=max_colors={colors}",
                str(palette),
            ],
            check=True,
        )
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-v",
                "error",
                "-i",
                str(src),
                "-i",
                str(palette),
                "-lavfi",
                f"{base_filter} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
                str(GIF),
            ],
            check=True,
        )
        if GIF.stat().st_size <= GIF_LIMIT:
            break
    print(f"demo.gif: {GIF.stat().st_size / 1e6:.1f} MB")


# ------------------------------------------------------------------ main


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--skip-build", action="store_true", help="do not rebuild the frontend")
    ap.add_argument("--no-gif", action="store_true", help="skip the demo GIF")
    ap.add_argument("--only", nargs="*", choices=[*ALL_SCENES, "gif"], help="only these scenes")
    args = ap.parse_args()

    if not args.skip_build:
        build_frontend()
    from silgi.samples import write_samples

    write_samples(SAMPLES)
    SHOTS.mkdir(parents=True, exist_ok=True)
    wanted = (
        set(args.only) if args.only else set(ALL_SCENES) | ({"gif"} if not args.no_gif else set())
    )

    port = free_port()
    server = start_server(port)
    base = f"http://127.0.0.1:{port}/"
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            scenes: dict[str, Callable[[], None]] = {
                "hero": lambda: scene_hero(browser, base),
                "editor": lambda: scene_editor(browser, base),
                "dark": lambda: scene_editor(browser, base, dark=True),
                "tools": lambda: scene_tools(browser, base),
                "batch": lambda: scene_batch(browser, base),
                "export": lambda: scene_export(browser, base),
                "gif": lambda: make_gif(browser, base),
            }
            for name, fn in scenes.items():
                if name in wanted:
                    print(f"-> {name}")
                    fn()
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=10)
    print(f"done: images in {SHOTS}")


if __name__ == "__main__":
    main()
