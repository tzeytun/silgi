"""Generate silgi's CC0 sample and test images.

Usage:
    python scripts/make_samples.py [OUT_DIR]   (default: ./samples)
"""

from __future__ import annotations

import sys
from pathlib import Path

from silgi.samples import write_samples


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("samples")
    paths = write_samples(out)
    print(f"wrote {len(paths)} images to {out.resolve()}")


if __name__ == "__main__":
    main()
