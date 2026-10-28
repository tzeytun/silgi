"""Shared array type aliases for the core engine."""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

#: Any image array: HxW or HxWxC, uint8 / uint16 / float32.
Image: TypeAlias = npt.NDArray[Any]
#: Float32 image normalised to [0, 1], always HxWxC.
FloatImage: TypeAlias = npt.NDArray[np.float32]
#: Boolean mask, HxW.
BoolMask: TypeAlias = npt.NDArray[np.bool_]
#: A mask as accepted by public functions: bool, uint8 (0/255) or float in [0, 1].
MaskLike: TypeAlias = npt.NDArray[Any]
#: Inclusive-exclusive box (y0, y1, x0, x1).
Box: TypeAlias = tuple[int, int, int, int]
