from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from silgi import cli
from silgi.core.io import ExportOptions, encode_image, read_image
from silgi.core.select.morph import save_mask
from silgi.samples import composite, fake_logo, landscape, text_watermark, to_uint8


def _write(path: Path, arr: np.ndarray) -> Path:
    path.write_bytes(encode_image(arr, ExportOptions()).data)
    return path


def test_remove_with_mask(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    img = to_uint8(landscape(120, 180, seed=1))
    src = _write(tmp_path / "in.png", img)
    mask = np.zeros((120, 180), bool)
    mask[40:60, 50:120] = True
    save_mask(tmp_path / "m.png", mask)
    assert (
        cli.main(["remove", str(src), "--mask", str(tmp_path / "m.png"), "--method", "telea"]) == 0
    )
    out = read_image(tmp_path / "in.silgi.png").pixels
    assert np.array_equal(out[~mask], img[~mask])
    text = capsys.readouterr().out
    assert "telea" in text
    assert "saved" in text

    jpg = tmp_path / "o.jpg"
    assert (
        cli.main(
            [
                "remove",
                str(src),
                "--mask",
                str(tmp_path / "m.png"),
                "-o",
                str(jpg),
                "--keep-metadata",
            ]
        )
        == 0
    )
    assert "lossy" in capsys.readouterr().out
    assert read_image(jpg).format == "JPEG"


def test_remove_with_watermark_only(tmp_path: Path) -> None:
    clean = landscape(200, 420, seed=3)
    mark = text_watermark("SAMPLE", height_px=50, opacity=0.45)
    img = to_uint8(composite(clean, mark, 60, 50))
    src = _write(tmp_path / "w.png", img)
    wm = _write(tmp_path / "mark.png", mark.rgba8())
    out = tmp_path / "clean.tif"
    assert cli.main(["remove", str(src), "--watermark", str(wm), "-o", str(out)]) == 0
    res = read_image(out).pixels
    th, tw = mark.alpha.shape
    err = np.abs(
        res[60 : 60 + th, 50 : 50 + tw].astype(int) - to_uint8(clean)[60 : 60 + th, 50 : 50 + tw]
    ).mean()
    assert err < 1.5
    assert (
        cli.main(
            [
                "remove",
                str(src),
                "--watermark",
                str(wm),
                "--position",
                "60",
                "50",
                "--format",
                "webp",
            ]
        )
        == 0
    )


def test_remove_errors(tmp_path: Path) -> None:
    src = _write(tmp_path / "a.png", np.zeros((10, 10, 3), np.uint8))
    assert cli.main(["remove", str(src)]) == 2
    opaque = _write(tmp_path / "o.png", np.zeros((4, 4, 3), np.uint8))
    assert cli.main(["remove", str(src), "--watermark", str(opaque)]) == 2


def test_batch_with_mask_template_and_alpha(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ind, outd = tmp_path / "in", tmp_path / "out"
    ind.mkdir()
    logo = fake_logo(60, opacity=1.0)
    mark = text_watermark("SAMPLE", height_px=40, opacity=0.5)
    for i in range(3):
        bg = landscape(160, 260, seed=10 + i)
        _write(ind / f"p{i}.png", to_uint8(composite(composite(bg, logo, 20, 20), mark, 100, 30)))
    _write(ind / "small.png", np.zeros((30, 30, 3), np.uint8))
    (ind / "notes.txt").write_text("ignored")

    mask = np.zeros((160, 260), bool)
    mask[20:80, 20:80] = True
    save_mask(tmp_path / "m.png", mask)
    assert (
        cli.main(
            ["batch", str(ind), str(outd), "--mask", str(tmp_path / "m.png"), "--method", "telea"]
        )
        == 0
    )
    assert sorted(p.name for p in outd.iterdir()) == ["p0.png", "p1.png", "p2.png"]
    assert "mask size differs" in capsys.readouterr().out

    # opaque template -> template matching + inpainting
    opaque = np.dstack([to_uint8(logo.rgb), np.full(logo.alpha.shape, 255, np.uint8)])
    flat_logo = to_uint8(composite(np.full((60, 60, 3), 0.5, np.float32), logo, 0, 0))
    _write(tmp_path / "logo.png", flat_logo)
    assert opaque.shape[2] == 4
    out2 = tmp_path / "out2"
    assert (
        cli.main(
            [
                "batch",
                str(ind),
                str(out2),
                "--template",
                str(tmp_path / "logo.png"),
                "--format",
                "jpeg",
            ]
        )
        == 0
    )
    assert any(p.suffix == ".jpg" for p in out2.iterdir())

    # transparent template -> reverse alpha
    _write(tmp_path / "mark.png", mark.rgba8())
    out3 = tmp_path / "out3"
    assert cli.main(["batch", str(ind), str(out3), "--template", str(tmp_path / "mark.png")]) == 0
    assert "alpha" in capsys.readouterr().out


def test_batch_errors(tmp_path: Path) -> None:
    assert cli.main(["batch", str(tmp_path / "missing"), str(tmp_path / "o")]) == 2
    (tmp_path / "empty").mkdir()
    assert cli.main(["batch", str(tmp_path / "empty"), str(tmp_path / "o")]) == 2
    assert cli.main(["batch", str(tmp_path / "empty"), str(tmp_path / "o"), "--mask", "x.png"]) == 1
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "broken.png").write_bytes(b"not a png")
    save_mask(tmp_path / "m.png", np.ones((4, 4), bool))
    assert (
        cli.main(["batch", str(bad), str(tmp_path / "o"), "--mask", str(tmp_path / "m.png")]) == 1
    )


def test_default_command_is_serve(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int, bool, bool]] = []
    monkeypatch.setattr(
        cli,
        "serve",
        lambda h, p, open_browser, window: calls.append((h, p, open_browser, window)) or 0,
    )
    assert cli.main([]) == 0
    assert cli.main(["--port", "9000", "--no-browser"]) == 0
    assert cli.main(["serve", "--window"]) == 0
    assert calls == [
        ("127.0.0.1", 8765, True, False),
        ("127.0.0.1", 9000, False, False),
        ("127.0.0.1", 8765, True, True),
    ]
    with pytest.raises(SystemExit):
        cli.main(["--version"])


def test_free_port() -> None:
    port = cli._free_port("127.0.0.1", 0)
    assert port > 0
