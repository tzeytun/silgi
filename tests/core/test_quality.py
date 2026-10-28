"""Inpainting quality must not regress.

Baselines live in ``tests/baselines/quality.json``. After an intentional
improvement, refresh them with::

    SILGI_UPDATE_BASELINES=1 pytest tests/core/test_quality.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.quality_cases import cases, evaluate

BASELINE = Path(__file__).parents[1] / "baselines" / "quality.json"
PSNR_TOLERANCE = 0.5  # dB
SSIM_TOLERANCE = 0.005
UPDATE = os.environ.get("SILGI_UPDATE_BASELINES") == "1"

_results: dict[str, dict[str, float]] = {}


def _load() -> dict[str, dict[str, float]]:
    if BASELINE.exists():
        data: dict[str, dict[str, float]] = json.loads(BASELINE.read_text(encoding="utf-8"))
        return data
    return {}


@pytest.mark.parametrize(("scene_name", "mask_name", "method"), list(cases()))
def test_quality_does_not_regress(scene_name: str, mask_name: str, method: str) -> None:
    key = f"{scene_name}/{mask_name}/{method}"
    score = evaluate(scene_name, mask_name, method)
    _results[key] = {"psnr": round(score.psnr, 2), "ssim": round(score.ssim, 4)}
    if UPDATE:
        data = _load()
        data[key] = _results[key]
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(
            json.dumps(dict(sorted(data.items())), indent=2) + "\n", encoding="utf-8"
        )
        return
    base = _load().get(key)
    assert base is not None, f"no baseline for {key}; run with SILGI_UPDATE_BASELINES=1"
    assert score.psnr >= base["psnr"] - PSNR_TOLERANCE, (
        f"{key}: PSNR {score.psnr:.2f} < {base['psnr']}"
    )
    assert score.ssim >= base["ssim"] - SSIM_TOLERANCE, (
        f"{key}: SSIM {score.ssim:.4f} < {base['ssim']}"
    )
