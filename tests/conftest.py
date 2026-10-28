from __future__ import annotations

import cv2
import numpy as np
import pytest

from silgi.samples import fbm


def make_image(
    rng: np.random.Generator, h: int, w: int, channels: int = 3, dtype: type = np.uint8
) -> np.ndarray:
    """A textured random test image of the requested layout."""
    base = np.stack(
        [fbm(h, w, seed=int(rng.integers(1 << 30)), octaves=4, base=4) for _ in range(channels)],
        axis=2,
    )
    base = np.clip(base * 0.8 + rng.random((h, w, channels), dtype=np.float32) * 0.2, 0, 1)
    scale = 255 if dtype == np.uint8 else 65535
    img = np.rint(base * scale).astype(dtype)
    return img[:, :, 0] if channels == 1 else img


def random_mask(rng: np.random.Generator, h: int, w: int, max_frac: float = 0.25) -> np.ndarray:
    """A random bool mask made of blobs, boxes and strokes (sometimes on the border)."""
    m = np.zeros((h, w), np.uint8)
    for _ in range(int(rng.integers(1, 5))):
        kind = rng.integers(3)
        cy, cx = int(rng.integers(0, h)), int(rng.integers(0, w))
        size = int(rng.integers(2, max(3, int(min(h, w) * max_frac))))
        if kind == 0:
            cv2.circle(m, (cx, cy), size // 2 + 1, 1, -1)
        elif kind == 1:
            cv2.rectangle(m, (cx, cy), (cx + size, cy + size // 2 + 1), 1, -1)
        else:
            ex, ey = int(rng.integers(0, w)), int(rng.integers(0, h))
            cv2.line(m, (cx, cy), (ex, ey), 1, int(rng.integers(1, 6)))
    return m.astype(bool)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)
