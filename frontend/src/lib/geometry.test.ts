import { describe, expect, it } from 'vitest'
import {
  clampPan,
  clampZoom,
  fitView,
  formatZoom,
  MAX_ZOOM,
  MIN_ZOOM,
  pickLevel,
  toImage,
  toScreen,
  visibleTiles,
  zoomAt,
} from './geometry'

describe('geometry', () => {
  it('fits an image into the viewport, centred, never above 1:1', () => {
    const v = fitView({ w: 1000, h: 800 }, { w: 2000, h: 1000 }, 0)
    expect(v.zoom).toBeCloseTo(0.5)
    expect(v.x).toBeCloseTo(0)
    expect(v.y).toBeCloseTo(150)
    expect(fitView({ w: 1000, h: 800 }, { w: 100, h: 100 }).zoom).toBe(1)
    expect(fitView({ w: 1000, h: 800 }, { w: 100, h: 100 }, 0, true).zoom).toBeGreaterThan(1)
  })

  it('zooms around the anchor so that point stays put', () => {
    const view = { zoom: 1, x: 40, y: 20 }
    const anchor = { x: 300, y: 200 }
    const before = toImage(view, anchor.x, anchor.y)
    const next = zoomAt(view, 2, anchor)
    const after = toImage(next, anchor.x, anchor.y)
    expect(next.zoom).toBe(2)
    expect(after.x).toBeCloseTo(before.x)
    expect(after.y).toBeCloseTo(before.y)
  })

  it('clamps zoom', () => {
    expect(clampZoom(1000)).toBe(MAX_ZOOM)
    expect(clampZoom(0)).toBe(MIN_ZOOM)
    expect(zoomAt({ zoom: MAX_ZOOM, x: 0, y: 0 }, 2, { x: 0, y: 0 }).zoom).toBe(MAX_ZOOM)
  })

  it('converts between screen and image coordinates', () => {
    const view = { zoom: 2, x: 10, y: -5 }
    const s = toScreen(view, 7, 9)
    expect(toImage(view, s.x, s.y)).toEqual({ x: 7, y: 9 })
  })

  it('picks pyramid levels from zoom', () => {
    expect(pickLevel(1, 5)).toBe(0)
    expect(pickLevel(4, 5)).toBe(0)
    expect(pickLevel(0.5, 5)).toBe(1)
    expect(pickLevel(0.26, 5)).toBe(1)
    expect(pickLevel(0.24, 5)).toBe(2)
    expect(pickLevel(0.001, 3)).toBe(2)
  })

  it('lists only tiles that touch the viewport', () => {
    const image = { w: 2000, h: 1000 }
    const all = visibleTiles({ zoom: 1, x: 0, y: 0 }, { w: 5000, h: 5000 }, image, 512, 0)
    expect(all).toHaveLength(4 * 2)
    const part = visibleTiles({ zoom: 1, x: 0, y: 0 }, { w: 600, h: 300 }, image, 512, 0)
    expect(part.map((t) => [t.tx, t.ty])).toEqual([
      [0, 0],
      [1, 0],
    ])
    const panned = visibleTiles({ zoom: 1, x: -1100, y: 0 }, { w: 400, h: 400 }, image, 512, 0)
    expect(panned.map((t) => t.tx)).toEqual([2])
    expect(visibleTiles({ zoom: 1, x: 5000, y: 0 }, { w: 400, h: 400 }, image, 512, 0)).toEqual([])
  })

  it('keeps part of the image reachable when panning', () => {
    const v = clampPan(
      { zoom: 1, x: -99999, y: 99999 },
      { w: 800, h: 600 },
      { w: 1000, h: 1000 },
      80,
    )
    expect(v.x).toBe(80 - 1000)
    expect(v.y).toBe(600 - 80)
  })

  it('formats zoom', () => {
    expect(formatZoom(0.333)).toBe('33%')
  })
})
