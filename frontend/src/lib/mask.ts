/** Helpers for the selection mask, which lives in an alpha canvas (downscaled for huge images). */
export type CombineMode = 'replace' | 'add' | 'subtract' | 'intersect'

export const MAX_MASK_SIDE = 4096

export function maskScale(w: number, h: number): number {
  return Math.min(1, MAX_MASK_SIDE / Math.max(w, h))
}

export interface Box {
  x: number
  y: number
  w: number
  h: number
}

/** Bounding box of pixels with alpha >= threshold in RGBA data, or null. */
export function alphaBounds(
  data: Uint8ClampedArray | Uint8Array,
  w: number,
  h: number,
  threshold = 1,
): Box | null {
  let x0 = w
  let y0 = h
  let x1 = -1
  let y1 = -1
  for (let y = 0; y < h; y++) {
    const row = y * w * 4
    for (let x = 0; x < w; x++) {
      if (data[row + x * 4 + 3] >= threshold) {
        if (x < x0) x0 = x
        if (x > x1) x1 = x
        if (y < y0) y0 = y
        if (y > y1) y1 = y
      }
    }
  }
  return x1 < 0 ? null : { x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1 }
}

export const expandBox = (b: Box, m: number, w: number, h: number): Box => {
  const x = Math.max(0, b.x - m)
  const y = Math.max(0, b.y - m)
  return { x, y, w: Math.min(w, b.x + b.w + m) - x, h: Math.min(h, b.y + b.h + m) - y }
}

export const compositeOp = (mode: CombineMode): GlobalCompositeOperation => {
  switch (mode) {
    case 'subtract':
      return 'destination-out'
    case 'intersect':
      return 'destination-in'
    default:
      return 'source-over'
  }
}

/** Turns a grayscale image decoded by the browser into white pixels with alpha = brightness. */
export function grayToAlpha(data: Uint8ClampedArray): void {
  for (let i = 0; i < data.length; i += 4) {
    data[i + 3] = data[i]
    data[i] = data[i + 1] = data[i + 2] = 255
  }
}

/** Memory budget for undo snapshots, in bytes. */
export const SNAPSHOT_BUDGET = 256 * 1024 * 1024

/** Drop the oldest snapshots until the total fits the budget (always keeps at least one). */
export function trimSnapshots<T extends { bytes: number }>(
  list: T[],
  budget = SNAPSHOT_BUDGET,
): T[] {
  let total = list.reduce((s, e) => s + e.bytes, 0)
  let start = 0
  while (total > budget && list.length - start > 1) total -= list[start++].bytes
  return start ? list.slice(start) : list
}
