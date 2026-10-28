from __future__ import annotations

import io
import json
import time
import zipfile
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from silgi.api import create_app
from silgi.core.io import ExportOptions, encode_image, read_image
from silgi.core.select.morph import mask_to_png
from silgi.samples import composite, landscape, text_watermark, to_uint8, to_uint16
from tests.conftest import make_image


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app()) as c:
        yield c


def _png(arr: np.ndarray) -> bytes:
    return encode_image(arr, ExportOptions()).data


def _upload(client: TestClient, arr: np.ndarray, name: str = "x.png") -> dict[str, Any]:
    r = client.post("/api/images", files={"file": (name, _png(arr), "image/png")})
    assert r.status_code == 200, r.text
    return r.json()


def _wait(client: TestClient, job_id: str, timeout: float = 60) -> dict[str, Any]:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        snap = client.get(f"/api/jobs/{job_id}").json()
        if snap["status"] in ("done", "error", "cancelled"):
            return snap
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def _mask_from(r: Any, shape: tuple[int, int]) -> np.ndarray:
    """Rebuild a full-size mask from a cropped mask response."""
    x, y = (int(v) for v in r.headers["x-silgi-offset"].split(","))
    crop = read_image(r.content).pixels > 0
    full = np.zeros(shape, bool)
    full[y : y + crop.shape[0], x : x + crop.shape[1]] = crop
    return full


def _remove(
    client: TestClient, doc_id: str, mask: np.ndarray | None, **options: Any
) -> dict[str, Any]:
    files = {"mask": ("m.png", mask_to_png(mask), "image/png")} if mask is not None else {}
    r = client.post(
        f"/api/images/{doc_id}/remove", data={"options": json.dumps(options)}, files=files
    )
    assert r.status_code == 200, r.text
    return _wait(client, r.json()["job_id"])


def test_health_and_methods(client: TestClient) -> None:
    assert client.get("/api/health").json()["ok"] is True
    methods = {m["id"] for m in client.get("/api/methods").json()["methods"]}
    assert {"auto", "alpha", "patchmatch", "telea"} <= methods


def test_upload_tiles_and_preview(client: TestClient) -> None:
    img = make_image(np.random.default_rng(0), 700, 1100, 3, np.uint16)
    info = _upload(client, img)
    assert (info["width"], info["height"], info["bit_depth"]) == (1100, 700, 16)
    assert info["levels"] == 3
    doc = info["id"]
    r = client.get(f"/api/images/{doc}/v/0/tiles/0/1/1.webp")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/webp"
    tile = read_image(r.content).pixels
    assert tile.shape == (700 - 512, 512, 3)
    assert client.get(f"/api/images/{doc}/v/0/tiles/0/5/0.webp").status_code == 404
    assert client.get(f"/api/images/{doc}/v/0/tiles/99/0/0.webp").status_code == 404
    assert client.get(f"/api/images/{doc}/v/7/tiles/0/0/0.webp").status_code == 404
    small = read_image(client.get(f"/api/images/{doc}/v/0/preview.webp?size=256").content).pixels
    assert max(small.shape[:2]) == 256
    assert client.get("/api/images/nope").status_code == 404


def test_remove_job_untouched_and_export(client: TestClient) -> None:
    img = make_image(np.random.default_rng(1), 96, 128, 4, np.uint16)
    doc = _upload(client, img)["id"]
    mask = np.zeros((96, 128), bool)
    mask[30:50, 40:90] = True
    snap = _remove(client, doc, mask, method="telea")
    assert snap["status"] == "done", snap
    res = snap["result"]
    assert res["version"] == 1
    assert res["method"] == "telea"
    assert res["changed"] is True
    y0, y1, x0, x1 = res["box"]
    assert y0 >= 30
    assert y1 <= 50
    assert x0 >= 40
    assert x1 <= 90

    r = client.post(f"/api/images/{doc}/export", json={"version": 1, "format": "png"})
    assert r.status_code == 200
    assert "x-silgi.png" in r.headers["content-disposition"]
    out = read_image(r.content).pixels
    assert out.dtype == np.uint16
    assert out.shape == img.shape
    assert np.array_equal(out[~mask], img[~mask])
    assert not np.array_equal(out[mask], img[mask])

    info = client.get(f"/api/images/{doc}").json()
    assert [v["id"] for v in info["versions"]] == [0, 1]
    vm = client.get(f"/api/images/{doc}/v/1/mask.png")
    assert vm.status_code == 200
    assert vm.headers["x-silgi-offset"] == "0,0"
    assert client.get(f"/api/images/{doc}/v/0/mask.png").status_code == 404
    assert client.get(f"/api/images/{doc}/v/1/tiles/0/0/0.webp").status_code == 200

    # a second edit on top of the first keeps the first one
    mask2 = np.zeros((96, 128), bool)
    mask2[60:70, 10:30] = True
    snap2 = _remove(client, doc, mask2, method="ns", version=1)
    assert snap2["result"]["version"] == 2
    out2 = read_image(client.post(f"/api/images/{doc}/export", json={"version": 2}).content).pixels
    assert np.array_equal(out2[mask], out[mask])
    untouched = ~(mask | mask2)
    assert np.array_equal(out2[untouched], img[untouched])

    jpg = client.post(f"/api/images/{doc}/export", json={"version": 2, "format": "jpeg"})
    warnings = json.loads(jpg.headers["x-silgi-warnings"])
    assert any("8-bit" in w for w in warnings)


def test_websocket_events(client: TestClient) -> None:
    img = to_uint8(landscape(200, 300, seed=2))
    doc = _upload(client, img)["id"]
    mask = np.zeros((200, 300), bool)
    mask[60:140, 100:220] = True
    r = client.post(
        f"/api/images/{doc}/remove",
        data={"options": json.dumps({"method": "patchmatch"})},
        files={"mask": ("m.png", mask_to_png(mask), "image/png")},
    )
    job = r.json()["job_id"]
    types = []
    with client.websocket_connect(f"/api/jobs/{job}/events") as ws:
        while True:
            ev = ws.receive_json()
            types.append(ev["type"])
            if ev["type"] == "preview":
                assert client.get(f"/api/jobs/{job}/previews/{ev['index']}.webp").status_code == 200
            if ev["type"] in ("done", "error", "cancelled") or ev.get("status") in (
                "done",
                "error",
            ):
                break
    assert types[-1] == "done" or "done" in types or types[0] == "status"
    assert _wait(client, job)["status"] == "done"
    assert client.get(f"/api/jobs/{job}/previews/999.webp").status_code == 404
    with client.websocket_connect("/api/jobs/unknown/events") as ws:
        assert ws.receive_json()["type"] == "error"


def test_cancel_queued_job(client: TestClient) -> None:
    img = to_uint8(landscape(300, 400, seed=5))
    doc = _upload(client, img)["id"]
    mask = np.zeros((300, 400), bool)
    mask[50:250, 50:350] = True
    opts = json.dumps({"method": "patchmatch", "quality": "best"})
    files = {"mask": ("m.png", mask_to_png(mask), "image/png")}
    first = client.post(f"/api/images/{doc}/remove", data={"options": opts}, files=files).json()[
        "job_id"
    ]
    second = client.post(f"/api/images/{doc}/remove", data={"options": opts}, files=files).json()[
        "job_id"
    ]
    client.delete(f"/api/jobs/{first}")
    assert client.delete(f"/api/jobs/{second}").status_code == 200
    assert _wait(client, second)["status"] == "cancelled"
    assert _wait(client, first)["status"] in ("cancelled", "done")
    assert client.delete("/api/jobs/nope").status_code == 404
    assert client.get("/api/jobs/nope").status_code == 404


def test_remove_errors(client: TestClient) -> None:
    doc = _upload(client, np.zeros((20, 30, 3), np.uint8))["id"]
    r = client.post(f"/api/images/{doc}/remove", data={"options": "{}"})
    assert r.status_code == 400
    empty = {"mask": ("m.png", mask_to_png(np.zeros((20, 30), bool)), "image/png")}
    assert client.post(f"/api/images/{doc}/remove", files=empty).status_code == 400
    wrong = {"mask": ("m.png", mask_to_png(np.ones((5, 5), bool)), "image/png")}
    far = json.dumps({"method": "telea", "mask_offset": [28, 0]})
    r = client.post(f"/api/images/{doc}/remove", data={"options": far}, files=wrong)
    assert r.status_code == 400
    near = json.dumps({"method": "telea", "mask_offset": [3, 4]})
    placed = client.post(f"/api/images/{doc}/remove", data={"options": near}, files=wrong)
    assert _wait(client, placed.json()["job_id"])["status"] == "done"
    bad = {"mask": ("m.png", b"nope", "image/png")}
    assert client.post(f"/api/images/{doc}/remove", files=bad).status_code == 400
    full = {"mask": ("m.png", mask_to_png(np.ones((20, 30), bool)), "image/png")}
    assert (
        client.post(f"/api/images/{doc}/remove", data={"options": "{bad"}, files=full).status_code
        == 422
    )
    opts = json.dumps({"watermark_id": "missing"})
    assert (
        client.post(f"/api/images/{doc}/remove", data={"options": opts}, files=full).status_code
        == 404
    )
    assert (
        client.post(
            f"/api/images/{doc}/remove", data={"options": json.dumps({"version": 9})}, files=full
        ).status_code
        == 404
    )
    r = client.post("/api/images", files={"file": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 415
    # heal without an offset fails inside the job
    snap = _remove(client, doc, np.ones((20, 30), bool) & (np.arange(30) < 5), method="heal")
    assert snap["status"] == "error"
    assert "offset" in snap["error"]
    assert client.delete(f"/api/images/{doc}").json()["ok"]
    assert client.get(f"/api/images/{doc}").status_code == 404


def test_sample_and_watermark_removal(client: TestClient) -> None:
    info = client.post("/api/sample").json()
    assert info["name"] == "sample-landscape.png"
    wid = info["watermarks"][0]["id"]
    loc = client.post(
        f"/api/images/{info['id']}/watermark/locate", json={"watermark_id": wid}
    ).json()
    assert loc["score"] > 0.5
    snap = _remove(client, info["id"], None, method="alpha", watermark_id=wid)
    assert snap["status"] == "done", snap
    assert snap["result"]["method"] == "alpha"
    assert client.get(f"/api/watermarks/{wid}.png").status_code == 200
    assert client.get("/api/watermarks/none.png").status_code == 404
    bad = client.post(f"/api/images/{info['id']}/watermark/locate", json={"watermark_id": "x"})
    assert bad.status_code == 404
    box = client.post(
        f"/api/images/{info['id']}/watermark/locate",
        json={"watermark_id": wid, "box": [0, 0, 10, 10]},
    )
    assert box.status_code == 400


def test_watermark_upload_and_estimate(client: TestClient) -> None:
    mark = text_watermark("SAMPLE", height_px=50, opacity=0.5)
    r = client.post("/api/watermarks", files={"file": ("w.png", _png(mark.rgba8()), "image/png")})
    assert r.status_code == 200
    assert r.json()["width"] == mark.alpha.shape[1]
    w16 = client.post(
        "/api/watermarks",
        files={"file": ("w.png", _png(to_uint16(mark.rgba_float())), "image/png")},
    )
    assert w16.status_code == 200
    opaque = client.post(
        "/api/watermarks",
        files={"file": ("w.png", _png(np.zeros((4, 4, 3), np.uint8)), "image/png")},
    )
    assert opaque.status_code == 400
    assert (
        client.post("/api/watermarks", files={"file": ("w.png", b"zz", "image/png")}).status_code
        == 415
    )

    imgs = [
        to_uint8(composite(landscape(240, 360, seed=300 + i, vary_layout=True), mark, 90, 60))
        for i in range(8)
    ]
    files = [("files", (f"{i}.png", _png(im), "image/png")) for i, im in enumerate(imgs)]
    job = client.post("/api/watermarks/estimate", files=files).json()["job_id"]
    snap = _wait(client, job)
    assert snap["status"] == "done", snap
    assert len(snap["result"]["offsets"]) == 8
    assert client.get(f"/api/watermarks/{snap['result']['watermark_id']}.png").status_code == 200


def test_selection_endpoints(client: TestClient) -> None:
    img = np.zeros((60, 80, 3), np.uint8)
    img[:, 40:] = (250, 250, 250)
    doc = _upload(client, img)["id"]
    r = client.post(f"/api/images/{doc}/select/wand", json={"x": 50, "y": 10})
    m = _mask_from(r, (60, 80))
    assert m[:, 40:].all()
    assert not m[:, :40].any()
    assert (
        client.post(f"/api/images/{doc}/select/wand", json={"x": 500, "y": 10}).status_code == 400
    )
    r = client.post(f"/api/images/{doc}/select/color-range", json={"x": 60, "y": 5, "tolerance": 5})
    assert _mask_from(r, (60, 80))[:, 40:].all()
    assert json.loads(r.headers["x-silgi-color"])[0] > 0.9
    r = client.post(f"/api/images/{doc}/select/color-range", json={"color": [0, 0, 0]})
    assert _mask_from(r, (60, 80))[:, :40].all()
    r = client.post(
        f"/api/images/{doc}/select/color-range", json={"color": [1, 0, 0], "tolerance": 1}
    )
    assert r.headers["x-silgi-offset"] == "0,0"
    assert not _mask_from(r, (60, 80)).any()
    assert client.post(f"/api/images/{doc}/select/color-range", json={}).status_code == 400
    assert (
        client.post(f"/api/images/{doc}/select/color-range", json={"x": 999, "y": 0}).status_code
        == 400
    )
    assert client.post(f"/api/images/{doc}/select/text", json={}).json() == {"boxes": []}
    tiled = to_uint8(landscape(300, 500, seed=8))
    doc2 = _upload(client, tiled)["id"]
    hits = client.post(
        f"/api/images/{doc2}/select/similar", json={"box": [100, 100, 60, 60]}
    ).json()["hits"]
    assert any(h["is_source"] for h in hits)
    flat = _upload(client, np.zeros((50, 50, 3), np.uint8))["id"]
    assert (
        client.post(f"/api/images/{flat}/select/similar", json={"box": [5, 5, 20, 20]}).status_code
        == 400
    )


def test_export_zip(client: TestClient) -> None:
    a = _upload(client, np.zeros((10, 10, 3), np.uint8), "a.png")["id"]
    b = _upload(client, np.ones((12, 12, 3), np.uint8), "a.png")["id"]
    r = client.post(
        "/api/export/zip",
        json={
            "items": [{"image_id": a, "version": 0}, {"image_id": b, "version": 0}],
            "options": {"format": "tiff"},
        },
    )
    assert r.status_code == 200
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert names == ["a-silgi.tif", "a-silgi-2.tif"]


def test_frontend_fallback(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "silgi" in r.text


def test_mask_morph(client: TestClient) -> None:
    m = np.zeros((40, 50), bool)
    m[15:25, 20:30] = True
    files = {"mask": ("m.png", mask_to_png(m), "image/png")}
    for op, check in (
        ("grow", lambda a: a.sum() > m.sum()),
        ("shrink", lambda a: 0 < a.sum() < m.sum()),
    ):
        r = client.post("/api/mask/morph", data={"op": op, "px": "2"}, files=files)
        assert r.status_code == 200
        assert check(_mask_from(r, (40, 50))), op
    r = client.post("/api/mask/morph", data={"op": "feather", "px": "3"}, files=files)
    assert r.status_code == 200
    assert (
        client.post("/api/mask/morph", data={"op": "smooth", "px": "2"}, files=files).status_code
        == 200
    )
    assert (
        client.post("/api/mask/morph", data={"op": "x", "px": "2"}, files=files).status_code == 400
    )
    bad = {"mask": ("m.png", b"nope", "image/png")}
    assert (
        client.post("/api/mask/morph", data={"op": "grow", "px": "2"}, files=bad).status_code == 400
    )
