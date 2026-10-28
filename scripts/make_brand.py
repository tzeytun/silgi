"""Generate silgi's brand assets ("Kayıp nokta" — the lost dot).

The wordmark "sılgı" is set in Manrope ExtraBold (800) with -0.05em tracking and
converted to outlines with fontTools. The dot over the first ı is a solid ink
circle; the dot over the second ı is a dashed eraser-pink ring: the dot that
was erased.

Requires the frontend dependencies (for the Manrope and Inter font files):

    cd frontend && npm install && cd ..
    python scripts/make_brand.py

PNG exports are rendered with headless Chromium through Playwright.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path

from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "brand"
FONTS = ROOT / "frontend" / "node_modules" / "@fontsource"
MANROPE = FONTS / "manrope" / "files" / "manrope-latin-800-normal.woff2"
INTER = FONTS / "inter" / "files" / "inter-latin-500-normal.woff2"
PUBLIC = ROOT / "frontend" / "public"

INK = "#15171C"
WHITE = "#FFFFFF"
PINK = "#EF5B7B"

TRACKING = -0.05  # em
DOT_D = 0.15  # em, solid ink dot diameter
RING_D = 0.19  # em, dashed ring outer diameter
RING_STROKE = 0.03  # em
DASHES = 8
DASH_FILL = 0.58  # fraction of each dash slot that is drawn
TAGLINE = "Remove watermarks without touching the rest of your image."


# ------------------------------------------------------------------ font helpers


def _pair_kerning(font: TTFont, left: str, right: str) -> int:
    """Horizontal kerning between two glyphs from GPOS PairPos lookups."""
    if "GPOS" not in font:
        return 0
    total = 0
    gpos = font["GPOS"].table
    for lookup in gpos.LookupList.Lookup:
        for sub in lookup.SubTable:
            st = sub.ExtSubTable if lookup.LookupType == 9 else sub
            if getattr(st, "LookupType", lookup.LookupType) != 2:
                continue
            if left not in st.Coverage.glyphs:
                continue
            if st.Format == 1:
                idx = st.Coverage.glyphs.index(left)
                for rec in st.PairSet[idx].PairValueRecord:
                    if rec.SecondGlyph == right and rec.Value1 is not None:
                        total += getattr(rec.Value1, "XAdvance", 0) or 0
            elif st.Format == 2:
                c1 = st.ClassDef1.classDefs.get(left, 0)
                c2 = st.ClassDef2.classDefs.get(right, 0)
                rec = st.Class1Record[c1].Class2Record[c2]
                if rec.Value1 is not None:
                    total += getattr(rec.Value1, "XAdvance", 0) or 0
    return total


@dataclass
class Mark:
    """The outlined wordmark and its two dot positions, in font units (y up)."""

    path: str
    upm: int
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    dots: list[tuple[float, float]]


def _dot_center(font: TTFont) -> tuple[float, float]:
    """Centre of the dot of the regular "i" (same x as the stem)."""
    from fontTools.pens.recordingPen import DecomposingRecordingPen

    gs = font.getGlyphSet()
    rp = DecomposingRecordingPen(gs)
    gs["i"].draw(rp)
    contours: list[list[tuple[float, float]]] = []
    cur: list[tuple[float, float]] = []
    for op, args in rp.value:
        cur.extend(args)
        if op in ("closePath", "endPath"):
            contours.append(cur)
            cur = []
    top = max(contours, key=lambda c: min(p[1] for p in c))
    xs, ys = [p[0] for p in top], [p[1] for p in top]
    return (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0


def wordmark(glyphs: tuple[str, ...] = ("s", "dotlessi", "l", "g", "dotlessi")) -> Mark:
    font = TTFont(MANROPE)
    gs = font.getGlyphSet()
    upm = font["head"].unitsPerEm
    track = TRACKING * upm
    dot_cx, dot_cy = _dot_center(font)
    pen = SVGPathPen(gs)
    x = 0.0
    dots = []
    bounds = BoundsPen(gs)
    for i, name in enumerate(glyphs):
        if i:
            x += _pair_kerning(font, glyphs[i - 1], name)
        # flip y so the path reads top-down like SVG; baseline at y = 0
        gs[name].draw(TransformPen(pen, (1, 0, 0, -1, x, 0)))
        gs[name].draw(TransformPen(bounds, (1, 0, 0, 1, x, 0)))
        if name == "dotlessi":
            dots.append((x + dot_cx, dot_cy))
        x += gs[name].width + (track if i < len(glyphs) - 1 else 0)
    xmin, ymin, xmax, ymax = bounds.bounds
    return Mark(pen.getCommands(), upm, xmin, xmax, ymin, ymax, dots)


# ------------------------------------------------------------------ svg parts


def _dashed_ring(cx: float, cy: float, d_outer: float, stroke: float, color: str) -> str:
    """A dashed ring as filled annular sectors (renders identically everywhere)."""
    ro, ri = d_outer / 2.0, d_outer / 2.0 - stroke
    slot = 2 * math.pi / DASHES
    parts = []
    for k in range(DASHES):
        # dashes centred between the cardinal points, leaving a gap at 12 o'clock
        a0 = -math.pi / 2 + slot * (k + (1 - DASH_FILL) / 2)
        a1 = a0 + slot * DASH_FILL

        def p(r: float, a: float) -> str:
            return f"{cx + r * math.cos(a):.2f} {cy + r * math.sin(a):.2f}"

        parts.append(
            f"M{p(ro, a0)}A{ro:.2f} {ro:.2f} 0 0 1 {p(ro, a1)}"
            f"L{p(ri, a1)}A{ri:.2f} {ri:.2f} 0 0 0 {p(ri, a0)}Z"
        )
    return f'<path fill="{color}" d="{"".join(parts)}"/>'


def _circle(cx: float, cy: float, d: float, color: str) -> str:
    return f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{d / 2:.2f}" fill="{color}"/>'


def logo_svg(word: str, dot: str, ring: str) -> str:
    m = wordmark()
    em = m.upm
    pad = 0.06 * em
    top = max(m.ymax, max(cy + RING_D * em / 2 for _, cy in m.dots))
    x0, x1 = m.xmin - pad, m.xmax + pad
    y0, y1 = -top - pad, -m.ymin + pad
    (d1x, d1y), (d2x, d2y) = m.dots
    body = [
        f'<path fill="{word}" d="{m.path}"/>',
        _circle(d1x, -d1y, DOT_D * em, dot),
        _dashed_ring(d2x, -d2y, RING_D * em, RING_STROKE * em, ring),
    ]
    w, h = x1 - x0, y1 - y0
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0:.0f} {y0:.0f} {w:.0f} {h:.0f}" '
        f'width="{w / em * 100:.0f}" height="{h / em * 100:.0f}" role="img" aria-label="silgi">'
        f"<title>silgi</title>{''.join(body)}</svg>\n"
    )


def icon_svg(rounded: bool = True) -> str:
    """Ink square with a white ı whose dot is the pink dashed ring."""
    m = wordmark(("dotlessi",))
    em = m.upm
    size = 512.0
    ((dx, dy),) = m.dots
    glyph_top = dy + RING_D * em / 2
    glyph_h = glyph_top - m.ymin
    scale = size * 0.62 / glyph_h
    cx = (m.xmin + m.xmax) / 2
    tx = size / 2 - cx * scale
    ty = size / 2 + (glyph_top + m.ymin) / 2 * scale
    radius = size * 0.22 if rounded else 0
    inner = f'<path fill="{WHITE}" d="{m.path}"/>' + _dashed_ring(
        dx, -dy, RING_D * em * 1.15, RING_STROKE * em * 1.25, PINK
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512" '
        f'role="img" aria-label="silgi"><title>silgi</title>'
        f'<rect width="512" height="512" rx="{radius:.1f}" fill="{INK}"/>'
        f'<g transform="translate({tx:.2f} {ty:.2f}) scale({scale:.5f})">{inner}</g></svg>\n'
    )


# ------------------------------------------------------------------ rendering


def _render(page: object, html: str, width: int, height: int, transparent: bool) -> bytes:
    pg = page  # playwright Page
    pg.set_viewport_size({"width": width, "height": height})  # type: ignore[attr-defined]
    pg.set_content(html)  # type: ignore[attr-defined]
    pg.wait_for_timeout(50)  # type: ignore[attr-defined]
    data: bytes = pg.screenshot(omit_background=transparent, type="png")  # type: ignore[attr-defined]
    return data


def _svg_page(svg: str, size: int) -> str:
    return (
        "<html><body style='margin:0;background:transparent'>"
        f"<img src='data:image/svg+xml;utf8,{_url(svg)}' width='{size}' height='{size}' "
        "style='display:block'></body></html>"
    )


def _url(svg: str) -> str:
    from urllib.parse import quote

    return quote(svg)


def _social_page(logo: str) -> str:
    font = INTER.resolve().as_uri()
    return f"""<html><head><style>
@font-face {{ font-family: Inter; src: url('{font}') format('woff2'); font-weight: 500; }}
body {{ margin: 0; width: 1280px; height: 640px; background: {WHITE}; display: flex;
  flex-direction: column; align-items: center; justify-content: center; gap: 44px;
  font-family: Inter, sans-serif; }}
img {{ width: 520px; height: auto; }}
p {{ margin: 0; color: #4A4F5C; font-size: 34px; font-weight: 500; letter-spacing: -0.01em; }}
</style></head><body>
<img src='data:image/svg+xml;utf8,{_url(logo)}' alt='silgi'>
<p>{TAGLINE}</p></body></html>"""


def main() -> None:
    if not MANROPE.exists():
        raise SystemExit("Manrope not found. Run `npm install` in frontend/ first.")
    OUT.mkdir(parents=True, exist_ok=True)
    light = logo_svg(INK, INK, PINK)
    dark = logo_svg(WHITE, WHITE, PINK)
    mono = logo_svg(INK, INK, INK)
    icon = icon_svg()
    files = {
        "logo-light.svg": light,
        "logo-dark.svg": dark,
        "logo-mono.svg": mono,
        "icon.svg": icon,
    }
    for name, svg in files.items():
        (OUT / name).write_text(svg, encoding="utf-8", newline="\n")

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(device_scale_factor=1)
        for size in (512, 192):
            png = _render(page, _svg_page(icon, size), size, size, True)
            (OUT / f"icon-{size}.png").write_bytes(png)
        touch = _render(page, _svg_page(icon_svg(rounded=False), 180), 180, 180, False)
        (OUT / "apple-touch-icon.png").write_bytes(touch)
        big = Image.open(io.BytesIO(_render(page, _svg_page(icon, 256), 256, 256, True)))
        big.save(OUT / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48), (64, 64)])
        social = _render(page, _social_page(light), 1280, 640, False)
        (OUT / "social-preview.png").write_bytes(social)
        browser.close()

    # the web app serves the favicon and icons itself (no network)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    for name in ("icon.svg", "favicon.ico", "apple-touch-icon.png", "icon-192.png", "icon-512.png"):
        (PUBLIC / name).write_bytes((OUT / name).read_bytes())
    for name in ("logo-light.svg", "logo-dark.svg"):
        (PUBLIC / name).write_bytes((OUT / name).read_bytes())
    print(f"wrote brand assets to {OUT}")


if __name__ == "__main__":
    main()
