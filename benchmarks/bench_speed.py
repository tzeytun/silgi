"""Speed benchmark: time per method on 2, 12 and 24 megapixel images.

Run time depends on the *mask*, not the image, because every method works on the
mask's bounding box plus a margin. We therefore time two realistic masks per
image size: a logo-sized blob (about 1 % of the image) and a large one (5 %).

    python benchmarks/bench_speed.py            # prints a Markdown table
    python benchmarks/bench_speed.py --quick    # 2 MP only
"""

from __future__ import annotations

import argparse
import platform
import time

import cv2
import numpy as np

from silgi.core import engine
from silgi.samples import fbm, text_watermark, composite

SIZES = {"2 MP": (1200, 1700), "12 MP": (3000, 4000), "24 MP": (4000, 6000)}
MASKS = {"1 %": 0.01, "5 %": 0.05}
METHODS: list[tuple[str, dict]] = [
    ("telea", {}),
    ("ns", {}),
    ("heal", {"offset_frac": True}),
    ("patchmatch fast", {"quality": "fast"}),
    ("patchmatch balanced", {"quality": "balanced"}),
    ("patchmatch best", {"quality": "best"}),
    ("alpha (reverse blend)", {}),
]


def make_image(h: int, w: int) -> np.ndarray:
    base = np.dstack([fbm(h, w, seed=s, octaves=6, base=6) for s in (1, 2, 3)])
    grain = np.random.default_rng(0).normal(0, 0.01, (h, w, 3)).astype(np.float32)
    return np.clip(np.rint((base * 0.9 + grain) * 255), 0, 255).astype(np.uint8)


def make_mask(h: int, w: int, frac: float) -> np.ndarray:
    area = frac * h * w
    ry = int(np.sqrt(area / np.pi / 1.6))
    rx = int(1.6 * ry)
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (w // 2, h // 2), (rx, ry), 0, 0, 360, 1, -1)
    return m.astype(bool)


def timed(fn) -> float:
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    # compile Numba kernels first so timings measure the algorithm
    warm = make_image(200, 200)
    wm = np.zeros((200, 200), bool)
    wm[80:120, 80:120] = True
    engine.remove(warm, wm, "patchmatch", quality="fast")

    mark = text_watermark("SAMPLE", height_px=120, opacity=0.4)
    rows: dict[str, dict[str, str]] = {name: {} for name, _ in METHODS}
    sizes = {"2 MP": SIZES["2 MP"]} if args.quick else SIZES
    for size_name, (h, w) in sizes.items():
        img = make_image(h, w)
        wm_img = (composite(img.astype(np.float32) / 255, mark, h // 2, w // 2) * 255).astype(np.uint8)
        for mask_name, frac in MASKS.items():
            mask = make_mask(h, w, frac)
            ys, xs = np.nonzero(mask)
            for name, opts in METHODS:
                kwargs = {k: v for k, v in opts.items() if k != "offset_frac"}
                if name == "alpha (reverse blend)":
                    if frac != MASKS["1 %"]:
                        rows[name][f"{size_name} / {mask_name}"] = "n/a"
                        continue
                    secs = timed(lambda: engine.remove(wm_img, mask, "alpha", watermark=mark, watermark_position=(h // 2, w // 2)))
                elif opts.get("offset_frac"):
                    off = (0, int(xs.max() - xs.min()) + 4)
                    if xs.max() + off[1] >= w:
                        off = (0, -off[1])
                    secs = timed(lambda: engine.remove(img, mask, "heal", offset=off))
                else:
                    method = name.split()[0]
                    secs = timed(lambda: engine.remove(img, mask, method, **kwargs))
                rows[name][f"{size_name} / {mask_name}"] = f"{secs:.2f} s"
                print(f"{name:24s} {size_name:6s} mask {mask_name}: {secs:6.2f} s", flush=True)

    cols = [f"{s} / {m}" for s in sizes for m in MASKS]
    print()
    print(f"CPU: {platform.processor()} | cores: {__import__('os').cpu_count()} | Python {platform.python_version()}")
    print()
    print("| Method | " + " | ".join(cols) + " |")
    print("|---|" + "---:|" * len(cols))
    for name, _ in METHODS:
        print(f"| {name} | " + " | ".join(rows[name].get(c, "") for c in cols) + " |")


if __name__ == "__main__":
    main()
