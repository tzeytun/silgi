import { describe, expect, it } from 'vitest'
import {
  alphaBounds,
  compositeOp,
  expandBox,
  grayToAlpha,
  maskScale,
  MAX_MASK_SIDE,
  trimSnapshots,
} from './mask'

describe('mask helpers', () => {
  it('downscales only very large masks', () => {
    expect(maskScale(1000, 800)).toBe(1)
    expect(maskScale(MAX_MASK_SIDE * 2, 100)).toBe(0.5)
  })

  it('finds the alpha bounding box', () => {
    const w = 6
    const h = 5
    const data = new Uint8ClampedArray(w * h * 4)
    expect(alphaBounds(data, w, h)).toBeNull()
    for (const [x, y] of [
      [1, 1],
      [4, 3],
    ])
      data[(y * w + x) * 4 + 3] = 255
    expect(alphaBounds(data, w, h)).toEqual({ x: 1, y: 1, w: 4, h: 3 })
  })

  it('expands a box without leaving the image', () => {
    expect(expandBox({ x: 2, y: 2, w: 4, h: 4 }, 5, 10, 10)).toEqual({ x: 0, y: 0, w: 10, h: 10 })
  })

  it('maps selection modes to canvas operations', () => {
    expect(compositeOp('add')).toBe('source-over')
    expect(compositeOp('replace')).toBe('source-over')
    expect(compositeOp('subtract')).toBe('destination-out')
    expect(compositeOp('intersect')).toBe('destination-in')
  })

  it('turns grayscale into white with alpha', () => {
    const d = new Uint8ClampedArray([128, 128, 128, 255, 0, 0, 0, 255])
    grayToAlpha(d)
    expect([...d]).toEqual([255, 255, 255, 128, 255, 255, 255, 0])
  })

  it('trims the oldest undo snapshots to fit the memory budget, keeping at least one', () => {
    const list = [{ bytes: 40 }, { bytes: 40 }, { bytes: 40 }]
    expect(trimSnapshots(list, 100)).toHaveLength(2)
    expect(trimSnapshots(list, 1)).toHaveLength(1)
    expect(trimSnapshots(list, 1000)).toBe(list)
  })
})
