"""Selection tools: everything that produces or edits a mask.

Masks are boolean HxW arrays (``feather`` returns float weights in [0, 1]).
"""

from __future__ import annotations

from .colorrange import color_range
from .morph import combine, feather, grow, shrink, smooth
from .shapes import brush_stroke, ellipse, polygon, rectangle
from .similar import find_similar
from .textfind import find_text
from .wand import magic_wand

__all__ = [
    "brush_stroke",
    "color_range",
    "combine",
    "ellipse",
    "feather",
    "find_similar",
    "find_text",
    "grow",
    "magic_wand",
    "polygon",
    "rectangle",
    "shrink",
    "smooth",
]
