from __future__ import annotations

import sys

import numpy as np
import pytest

from silgi.core import alpha, auto, diffusion, encode_image, engine, healing, patchmatch
from silgi.core.align import locate, shift_watermark, subpixel_peak
from silgi.core.metrics import psnr, ssim
from silgi.core.progress import Cancelled, Progress
from silgi.samples import composite, demo_image, scene, text_watermark, to_uint8, to_uint16
from tests.conftest import make_image

# ---------------------------------------------------------------- determinism


@pytest.mark.parametrize("method", ["telea", "ns", "patchmatch", "auto"])
def test_same_input_same_bytes(method: str) -> None:
    rng = np.random.default_rng(3)
    img = make_image(rng, 80, 96, 3)
    mask = np.zeros((80, 96), bool)
    mask[30:50, 20:70] = True
    a = encode_image(engine.remove(img, mask, method).image).data  # type: ignore[arg-type]
    b = encode_image(engine.remove(img.copy(), mask.copy(), method).image).data  # type: ignore[arg-type]
    assert a == b


def test_patchmatch_seed_changes_result_but_stays_reproducible() -> None:
    img = to_uint8(scene("noise", 96, 96, seed=2))
    mask = np.zeros((96, 96), bool)
    mask[30:60, 30:60] = True
    a = patchmatch.remove(img, mask, seed=1)
    b = patchmatch.remove(img, mask, seed=1)
    c = patchmatch.remove(img, mask, seed=2)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


# ---------------------------------------------------------------- reverse alpha


@pytest.mark.parametrize(
    ("bits", "opacity"),
    [(8, 0.3), (8, 0.5), (16, 0.5), (16, 0.8)],
)
def test_reverse_alpha_recovers_original(bits: int, opacity: float) -> None:
    clean = scene("noise", 200, 380, seed=4)
    mark = text_watermark("SAMPLE", height_px=60, opacity=opacity)
    conv = to_uint8 if bits == 8 else to_uint16
    img = conv(composite(clean, mark, 60, 40))
    gt = conv(clean)
    out = alpha.remove(img, watermark=mark)  # position found automatically
    th, tw = mark.alpha.shape
    box = (slice(60, 60 + th), slice(40, 40 + tw))
    assert psnr(out[box], gt[box]) > 45.0
    assert ssim(out[box], gt[box]) > 0.99


def test_reverse_alpha_demo_image_and_fallback() -> None:
    d = demo_image(450, 675)
    img, gt = to_uint8(d.watermarked), to_uint8(d.clean)
    out = alpha.remove(img, watermark=d.logo)  # logo opacity 0.9 > threshold: inpainted
    ly, lx = d.logo_pos
    lh, lw = d.logo.alpha.shape
    box = (slice(ly, ly + lh), slice(lx, lx + lw))
    assert psnr(out[box], gt[box]) > psnr(img[box], gt[box]) + 8


def test_reverse_alpha_with_mask_and_position() -> None:
    clean = scene("gradient", 100, 160, seed=1)
    mark = text_watermark("AB", height_px=40, opacity=0.4)
    img = to_uint8(composite(clean, mark, 20, 30))
    mask = np.zeros(img.shape[:2], bool)
    mask[:, :60] = True  # only the left half of the watermark
    out = alpha.remove(img, mask, watermark=mark.rgba8(), position=(20, 30))
    assert np.array_equal(out[:, 60:], img[:, 60:])
    assert not np.array_equal(out[:, :60], img[:, :60])
    with pytest.raises(ValueError, match="outside"):
        alpha.remove(img, watermark=mark, position=(1000, 1000))


def test_reverse_alpha_gray_image_and_telea_fallback() -> None:
    clean = scene("noise", 120, 200, seed=5)[:, :, 0]
    mark = text_watermark("X", height_px=50, opacity=0.95)
    img = to_uint8(composite(clean[..., None].repeat(3, 2), mark, 30, 40))[:, :, 0]
    out = alpha.remove(img, watermark=mark, position=(30, 40), fallback="telea")
    assert out.shape == img.shape
    assert not np.array_equal(out, img)


def test_as_watermark_inputs() -> None:
    mark = text_watermark("A", height_px=20)
    rgb, a = alpha.as_watermark(mark)
    _, a2 = alpha.as_watermark(mark.rgba8())
    assert np.abs(a - a2).max() < 1 / 255
    la = np.dstack([mark.rgba8()[:, :, 0], mark.rgba8()[:, :, 3]])
    assert alpha.as_watermark(la)[0].shape == rgb.shape
    assert alpha.as_watermark((mark.rgba8()[:, :, :3], mark.rgba8()[:, :, 3]))[1].shape == a.shape
    with pytest.raises(ValueError, match="transparency"):
        alpha.as_watermark(np.zeros((4, 4, 3)))
    with pytest.raises(ValueError, match="watermark image"):
        engine.remove(np.zeros((10, 10, 3), np.uint8), np.ones((10, 10), bool), "alpha")


def test_subpixel_alignment() -> None:
    clean = scene("noise", 160, 360, seed=8)
    mark = text_watermark("SAMPLE", height_px=50, opacity=0.5)
    rgb, a = shift_watermark(mark.rgb, mark.alpha, 0.5, 0.25)
    img = clean.copy()
    h, w = a.shape
    img[39 : 39 + h, 59 : 59 + w] = (
        a[..., None] * rgb + (1 - a[..., None]) * img[39 : 39 + h, 59 : 59 + w]
    )
    img16 = to_uint16(img)
    found = alpha.find_placement(img16, mark)
    # shift_watermark pads by one pixel: the true corner is (40.5, 60.25)
    assert abs(found.top - 40.5) < 0.3
    assert abs(found.left - 60.25) < 0.3
    out = alpha.remove(img16, watermark=mark, position=(found.top, found.left))
    sl = (slice(40, 40 + mark.alpha.shape[0]), slice(60, 60 + mark.alpha.shape[1]))
    assert psnr(out[sl], to_uint16(clean)[sl]) > 38
    coarse = alpha.find_placement(img16, mark, refine=False)
    assert coarse.score is not None


def test_locate_errors_and_peak() -> None:
    with pytest.raises(ValueError, match="larger"):
        locate(np.zeros((5, 5, 3), np.uint8), np.ones((10, 10), np.float32))
    s = np.array([[0, 1, 0], [1, 3, 2], [0, 1, 0]], np.float32)
    y, x = subpixel_peak(s, 1, 1)
    assert y == 1.0
    assert x > 1.0


# ------------------------------------------------------- multi-image estimation


def test_estimate_watermark_from_many_images() -> None:
    from silgi.core import alpha_estimate
    from silgi.samples import landscape

    mark = text_watermark("SAMPLE", height_px=60, opacity=0.5)
    cleans = [landscape(300, 460, seed=200 + i, vary_layout=True) for i in range(10)]
    imgs = [to_uint8(composite(c, mark, 110, 90)) for c in cleans]
    est = alpha_estimate.estimate(imgs)
    assert est.single_color
    assert np.allclose(est.rgb[est.alpha > 0.2].mean(axis=0), 1.0, atol=0.05)
    assert est.rgba8().shape[2] == 4
    gains = []
    th, tw = mark.alpha.shape
    fp = np.zeros(imgs[0].shape[:2], bool)
    fp[110 : 110 + th, 90 : 90 + tw] = mark.alpha > 0
    for i in range(3):
        out = alpha.remove(imgs[i], watermark=(est.rgb, est.alpha), position=est.position(i))
        gt = to_uint8(cleans[i])
        gains.append(psnr(out, gt, fp) - psnr(imgs[i], gt, fp))
    assert min(gains) > 12


def test_estimate_errors() -> None:
    from silgi.core import alpha_estimate

    with pytest.raises(ValueError, match="at least"):
        alpha_estimate.estimate([np.zeros((10, 10, 3), np.uint8)] * 2)
    with pytest.raises(ValueError, match="same width"):
        alpha_estimate.estimate(
            [np.zeros((10, 10, 3), np.uint8)] * 2 + [np.zeros((12, 10, 3), np.uint8)]
        )
    flat = [np.full((40, 40), 100, np.uint8) for _ in range(4)]
    with pytest.raises(ValueError, match="no consistent"):
        alpha_estimate.estimate(flat, region=(0, 40, 0, 40))


# ---------------------------------------------------------------- healing


def test_clone_copies_exactly_and_heal_matches_lighting() -> None:
    img = to_uint8(scene("stripes", 64, 96, seed=1))
    mask = np.zeros((64, 96), bool)
    mask[20:30, 10:26] = True
    cloned = healing.remove(img, mask, offset=(0, 32), mode="clone")
    assert np.array_equal(cloned[20:30, 10:26], img[20:30, 42:58])
    darker = img.copy()
    darker[:, 40:] = (darker[:, 40:] * 0.6).astype(np.uint8)
    healed = healing.remove(darker, mask, offset=(0, 32), mode="heal")
    cl = healing.remove(darker, mask, offset=(0, 32), mode="clone")
    ref = img[20:30, 10:26].astype(float)
    assert np.abs(healed[20:30, 10:26] - ref).mean() < np.abs(cl[20:30, 10:26] - ref).mean()


def test_healing_errors() -> None:
    img = np.zeros((20, 20, 3), np.uint8)
    m = np.zeros((20, 20), bool)
    m[5:10, 5:10] = True
    with pytest.raises(ValueError, match="outside"):
        healing.remove(img, m, offset=(0, 100))
    with pytest.raises(ValueError, match="zero"):
        healing.remove(img, m, offset=(0, 0))
    with pytest.raises(ValueError, match="mode"):
        healing.remove(img, m, offset=(0, 6), mode="smudge")  # type: ignore[arg-type]
    assert np.array_equal(healing.remove(img, np.zeros((20, 20), bool), offset=(1, 1)), img)
    with pytest.raises(ValueError, match="offset"):
        engine.remove(img, m, "heal")


# ---------------------------------------------------------------- auto


def test_auto_choices() -> None:
    flat = to_uint8(scene("flat", 80, 120))
    thin = np.zeros((80, 120), bool)
    thin[40:43, 10:110] = True
    d = auto.choose(flat, thin)
    assert d.method == "telea"
    assert "thin" in d.reason

    noisy = make_image(np.random.default_rng(0), 80, 120, 3)
    assert auto.choose(noisy, thin).method == "patchmatch"
    wide = np.zeros((80, 120), bool)
    wide[20:60, 30:90] = True
    d = auto.choose(flat, wide)
    assert d.method == "patchmatch"
    assert "wide" in d.reason
    assert auto.choose(flat, wide, has_watermark=True).method == "alpha"
    assert auto.choose(flat, np.zeros((80, 120), bool)).method == "none"
    assert auto.surround_texture(flat, np.zeros((80, 120), bool)) == 0.0
    full = np.ones((80, 120), bool)
    assert auto.surround_texture(flat, full) == 0.0

    res = engine.remove(flat, np.zeros((80, 120), bool))
    assert res.method == "none"
    res = engine.remove(flat, thin)
    assert res.method == "telea"
    assert res.reason


def test_engine_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="unknown method"):
        engine.remove(np.zeros((4, 4), np.uint8), np.ones((4, 4), bool), "magic")  # type: ignore[arg-type]
    assert set(engine.DESCRIPTIONS) == set(engine.METHODS)


# ---------------------------------------------------------------- patchmatch


def test_patchmatch_source_region_is_respected() -> None:
    img = np.zeros((80, 160, 3), np.uint8)
    img[:, :80] = (200, 30, 30)
    img[:, 80:] = (30, 30, 200)
    mask = np.zeros((80, 160), bool)
    mask[30:50, 70:90] = True
    src = np.zeros((80, 160), bool)
    src[:, 100:] = True
    out = patchmatch.remove(img, mask, source_mask=src)
    assert (out[mask][:, 2] > 150).all()


def test_patchmatch_progress_preview_and_cancel() -> None:
    img = to_uint8(scene("noise", 160, 160, seed=3))
    mask = np.zeros((160, 160), bool)
    mask[40:120, 40:120] = True
    seen: list[float] = []
    previews: list[tuple[int, ...]] = []
    prog = Progress(
        on_progress=lambda f, m: seen.append(f), on_preview=lambda im, box: previews.append(box)
    )
    patchmatch.remove(img, mask, progress=prog)
    assert seen
    assert seen == sorted(seen)
    assert previews
    with pytest.raises(Cancelled):
        patchmatch.remove(img, mask, progress=Progress(should_cancel=lambda: True))


def test_patchmatch_falls_back_without_context() -> None:
    img = to_uint8(scene("noise", 12, 12))
    mask = np.ones((12, 12), bool)
    mask[0, 0] = False
    out = patchmatch.remove(img, mask)
    assert out.shape == img.shape


def test_patchmatch_parameter_errors() -> None:
    roi = scene("flat", 32, 32)
    hole = np.zeros((32, 32), bool)
    hole[10:20, 10:20] = True
    with pytest.raises(ValueError, match="odd"):
        patchmatch.fill(roi, hole, params=patchmatch.Params(patch_size=6))
    out = patchmatch.remove(to_uint8(roi), hole, patch_size=5, margin=8)
    assert out.shape == (32, 32, 3)
    assert np.array_equal(patchmatch.remove(to_uint8(roi), np.zeros((32, 32), bool)), to_uint8(roi))
    assert patchmatch.default_margin(np.zeros((4, 4), bool), 7) == 0


def test_diffusion_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="unknown"):
        diffusion.fill(scene("flat", 8, 8), np.ones((8, 8), bool), method="magic")  # type: ignore[arg-type]


def test_core_has_no_web_dependencies() -> None:
    import subprocess

    code = (
        "import sys, silgi.core; "
        "print([m for m in ('fastapi', 'uvicorn', 'starlette') if m in sys.modules])"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
