"""Command-line interface.

silgi                       start the web interface and open the browser
silgi --window              ... in a desktop window (needs silgi[window])
silgi remove IN --mask M    remove the masked area from one image
silgi batch IN_DIR OUT_DIR  process a folder with a shared template or mask
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
import webbrowser
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from silgi import __version__

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".heic", ".heif"}
METHODS = ("auto", "alpha", "patchmatch", "telea", "ns")
FORMAT_EXT = {"png": ".png", "jpeg": ".jpg", "webp": ".webp", "tiff": ".tif"}


# ----------------------------------------------------------------------- serve


def _free_port(host: str, preferred: int) -> int:
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
            except OSError:
                continue
            return int(s.getsockname()[1])
    raise RuntimeError("no free port")  # pragma: no cover


def serve(host: str, port: int, *, open_browser: bool, window: bool) -> int:
    import uvicorn

    from silgi.api import create_app

    port = _free_port(host, port)
    url = f"http://{host}:{port}/"
    config = uvicorn.Config(create_app(), host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    if window:
        try:
            import webview
        except ImportError:
            print(
                "Desktop window support is not installed. Run: pip install 'silgi[window]'",
                file=sys.stderr,
            )
            return 2
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        while not server.started:
            time.sleep(0.05)
        webview.create_window("silgi", url, width=1440, height=900, min_size=(960, 640))
        webview.start()
        server.should_exit = True
        thread.join(timeout=5)
        return 0

    def announce() -> None:
        while not server.started:
            time.sleep(0.05)
        print(f"silgi {__version__} is running at {url}  (Ctrl+C to stop)")
        if open_browser:
            webbrowser.open(url)

    threading.Thread(target=announce, daemon=True).start()
    server.run()
    return 0


# ----------------------------------------------------------------------- remove


def _output_path(src: Path, out: str | None, fmt: str | None) -> Path:
    if out:
        return Path(out)
    ext = FORMAT_EXT[fmt] if fmt else ".png"
    return src.with_name(f"{src.stem}.silgi{ext}")


def _export_options(args: argparse.Namespace, fmt: str):  # type: ignore[no-untyped-def]
    from silgi.core.io import ExportOptions

    return ExportOptions(
        format=fmt,  # type: ignore[arg-type]
        quality=args.jpeg_quality,
        keep_metadata=args.keep_metadata,
        keep_icc=not args.no_icc,
    )


def _print_progress(fraction: float, message: str) -> None:
    if sys.stderr.isatty():
        print(f"\r  {fraction * 100:5.1f}%  {message:<24}", end="", file=sys.stderr, flush=True)


def cmd_remove(args: argparse.Namespace) -> int:
    from silgi.core import engine
    from silgi.core.alpha import remove as alpha_remove
    from silgi.core.io import encode_image, format_from_path, read_image
    from silgi.core.progress import Progress
    from silgi.core.select.morph import load_mask

    src_path = Path(args.input)
    image = read_image(src_path)
    watermark = read_image(args.watermark).pixels if args.watermark else None
    if watermark is not None and (watermark.ndim != 3 or watermark.shape[2] != 4):
        print("error: --watermark must be a PNG with transparency", file=sys.stderr)
        return 2
    if args.mask is None and watermark is None:
        print("error: give --mask (or --watermark for reverse alpha blending)", file=sys.stderr)
        return 2

    out = _output_path(src_path, args.output, args.format)
    fmt = args.format or format_from_path(out)
    progress = Progress(_print_progress)
    position = tuple(args.position) if args.position else None

    if args.mask is None:
        result = alpha_remove(
            image.pixels, None, watermark=watermark, position=position, progress=progress
        )
        method, reason = "alpha", engine.DESCRIPTIONS["alpha"]
    else:
        mask = load_mask(args.mask, (image.height, image.width))
        res = engine.remove(
            image.pixels,
            mask,
            args.method,
            quality=args.quality,
            watermark=watermark,
            watermark_position=position,
            seed=args.seed,
            feather=args.feather,
            seam=args.seam,
            progress=progress,
        )
        result, method, reason = res.image, res.method, res.reason
    if sys.stderr.isatty():
        print(file=sys.stderr)

    enc = encode_image(
        result, _export_options(args, fmt), icc_profile=image.icc_profile, exif=image.exif
    )
    out.write_bytes(enc.data)
    print(f"{method}: {reason}")
    for w in enc.warnings:
        print(f"note: {w}")
    print(f"saved {out}")
    return 0


# ----------------------------------------------------------------------- batch


def _batch_mask(
    pixels: np.ndarray, template: np.ndarray, threshold: float, grow_px: int
) -> np.ndarray | None:
    from silgi.core.select import grow
    from silgi.core.select.similar import match_template

    hits = match_template(pixels, template, threshold=threshold)
    if not hits:
        return None
    mask = np.zeros(pixels.shape[:2], bool)
    for h in hits:
        mask[h.y : h.y + h.h, h.x : h.x + h.w] = True
    return grow(mask, grow_px)


def cmd_batch(args: argparse.Namespace) -> int:
    from silgi.core import alpha, engine
    from silgi.core.io import encode_image, read_image
    from silgi.core.select.morph import load_mask

    in_dir, out_dir = Path(args.input_dir), Path(args.output_dir)
    if not in_dir.is_dir():
        print(f"error: {in_dir} is not a folder", file=sys.stderr)
        return 2
    if (args.template is None) == (args.mask is None):
        print("error: give exactly one of --template or --mask", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        print(f"no images found in {in_dir}", file=sys.stderr)
        return 1

    template = read_image(args.template).pixels if args.template else None
    transparent = (
        template is not None
        and template.ndim == 3
        and template.shape[2] == 4
        and template[..., 3].min()
        < (template.dtype.type(255) if template.dtype == np.uint8 else 65535)
    )
    shared_mask = load_mask(args.mask) if args.mask else None
    fmt = args.format
    ok = skipped = failed = 0

    for i, path in enumerate(files, 1):
        prefix = f"[{i}/{len(files)}] {path.name}"
        try:
            img = read_image(path)
            px = img.pixels
            if shared_mask is not None:
                if shared_mask.shape != px.shape[:2]:
                    print(f"{prefix}: skipped (mask size differs)")
                    skipped += 1
                    continue
                res = engine.remove(
                    px, shared_mask, args.method, quality=args.quality, seed=args.seed
                )
                result, method = res.image, res.method
            elif transparent and args.method in ("auto", "alpha"):
                assert template is not None
                try:
                    place = alpha.find_placement(px, template)
                except ValueError:
                    place = None
                if place is None or (place.score is not None and place.score < args.threshold):
                    score = "" if place is None else f", score {place.score:.2f}"
                    print(f"{prefix}: skipped (watermark not found{score})")
                    skipped += 1
                    continue
                result = alpha.remove(
                    px,
                    None,
                    watermark=template,
                    position=(place.top, place.left),
                    quality=args.quality,
                )
                method = "alpha"
            else:
                assert template is not None
                mask = _batch_mask(px, template, args.threshold, args.grow)
                if mask is None:
                    print(f"{prefix}: skipped (template not found)")
                    skipped += 1
                    continue
                res = engine.remove(px, mask, args.method, quality=args.quality, seed=args.seed)
                result, method = res.image, res.method
            enc = encode_image(
                result, _export_options(args, fmt), icc_profile=img.icc_profile, exif=img.exif
            )
            target = out_dir / f"{path.stem}{FORMAT_EXT[fmt]}"
            target.write_bytes(enc.data)
            print(f"{prefix}: {method} -> {target.name}")
            ok += 1
        except Exception as exc:  # keep going; report at the end
            print(f"{prefix}: failed ({exc})", file=sys.stderr)
            failed += 1
    print(f"done: {ok} processed, {skipped} skipped, {failed} failed")
    return 0 if failed == 0 else 1


# ----------------------------------------------------------------------- parser


def _add_serve_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--host", default="127.0.0.1", help="address to listen on (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=8765, help="port (default: 8765, or a free one)")
    p.add_argument("--no-browser", action="store_true", help="do not open a browser")
    p.add_argument(
        "--window", action="store_true", help="open in a desktop window (needs silgi[window])"
    )


def _add_output_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--quality", choices=("fast", "balanced", "best"), default="balanced", help="quality preset"
    )
    p.add_argument(
        "--seed", type=int, default=0, help="random seed for PatchMatch (results are reproducible)"
    )
    p.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG/WebP quality when exporting lossy formats",
    )
    p.add_argument(
        "--keep-metadata",
        action="store_true",
        help="copy EXIF metadata (off by default for privacy)",
    )
    p.add_argument("--no-icc", action="store_true", help="do not embed the colour profile")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="silgi",
        description="Remove watermarks, logos and text without touching the rest of your image.",
    )
    parser.add_argument("--version", action="version", version=f"silgi {__version__}")
    sub = parser.add_subparsers(dest="command")

    _add_serve_args(sub.add_parser("serve", help="start the web interface (default)"))

    rm = sub.add_parser("remove", help="remove the masked area from one image")
    rm.add_argument("input", help="image to clean")
    rm.add_argument("--mask", help="mask PNG: white (or opaque) = remove")
    rm.add_argument("--method", choices=METHODS, default="auto")
    rm.add_argument(
        "--watermark", help="watermark PNG with transparency, for reverse alpha blending"
    )
    rm.add_argument(
        "--position",
        type=float,
        nargs=2,
        metavar=("TOP", "LEFT"),
        help="watermark position (default: found automatically)",
    )
    rm.add_argument(
        "--feather", type=float, default=0.0, help="feather the result inside the mask edge (px)"
    )
    rm.add_argument(
        "--seam", choices=("none", "poisson"), default="none", help="gradient-domain seam blending"
    )
    rm.add_argument("-o", "--output", help="output file (default: INPUT.silgi.png)")
    rm.add_argument(
        "--format", choices=tuple(FORMAT_EXT), help="output format (default: from -o, else png)"
    )
    _add_output_args(rm)

    bt = sub.add_parser("batch", help="process every image in a folder")
    bt.add_argument("input_dir")
    bt.add_argument("output_dir")
    src = bt.add_mutually_exclusive_group()
    src.add_argument(
        "--template",
        help="logo/watermark image to find in every file (PNG with alpha enables reverse alpha)",
    )
    src.add_argument(
        "--mask", help="the same mask for every image (all images must share its size)"
    )
    bt.add_argument("--method", choices=METHODS, default="auto")
    bt.add_argument(
        "--threshold", type=float, default=0.6, help="minimum match score for --template (0..1)"
    )
    bt.add_argument("--grow", type=int, default=3, help="grow template matches by this many px")
    bt.add_argument("--format", choices=tuple(FORMAT_EXT), default="png")
    _add_output_args(bt)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not args_list or args_list[0] not in (
        "serve",
        "remove",
        "batch",
        "-h",
        "--help",
        "--version",
    ):
        args_list = ["serve", *args_list]
    args = parser.parse_args(args_list)
    if args.command == "remove":
        return cmd_remove(args)
    if args.command == "batch":
        return cmd_batch(args)
    return serve(args.host, args.port, open_browser=not args.no_browser, window=args.window)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
