/** Viewport maths: screen = image * zoom + (x, y), all in CSS pixels. */
export interface View {
  zoom: number
  x: number
  y: number
}

export interface Size {
  w: number
  h: number
}

export const MIN_ZOOM = 0.02
export const MAX_ZOOM = 64

export const clampZoom = (z: number): number => Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, z))

/** Zoom that fits the whole image in the view with a margin (never above 1:1 unless asked). */
export function fitView(viewport: Size, image: Size, margin = 32, allowUpscale = false): View {
  const zw = (viewport.w - 2 * margin) / image.w
  const zh = (viewport.h - 2 * margin) / image.h
  let zoom = clampZoom(Math.min(zw, zh))
  if (!allowUpscale) zoom = Math.min(zoom, 1)
  return {
    zoom,
    x: (viewport.w - image.w * zoom) / 2,
    y: (viewport.h - image.h * zoom) / 2,
  }
}

/** Change zoom keeping the image point under `anchor` (screen coords) fixed. */
export function zoomAt(view: View, factor: number, anchor: { x: number; y: number }): View {
  const zoom = clampZoom(view.zoom * factor)
  const k = zoom / view.zoom
  return { zoom, x: anchor.x - (anchor.x - view.x) * k, y: anchor.y - (anchor.y - view.y) * k }
}

export const toImage = (view: View, sx: number, sy: number) => ({
  x: (sx - view.x) / view.zoom,
  y: (sy - view.y) / view.zoom,
})

export const toScreen = (view: View, ix: number, iy: number) => ({
  x: ix * view.zoom + view.x,
  y: iy * view.zoom + view.y,
})

/** Pyramid level to use: level L is the image downscaled by 2^L. */
export function pickLevel(zoom: number, levels: number): number {
  if (zoom >= 1) return 0
  return Math.min(levels - 1, Math.max(0, Math.floor(Math.log2(1 / zoom))))
}

export interface TileRef {
  level: number
  tx: number
  ty: number
}

/** Tiles of `level` that intersect the viewport. */
export function visibleTiles(
  view: View,
  viewport: Size,
  image: Size,
  tileSize: number,
  level: number,
): TileRef[] {
  const span = tileSize * 2 ** level // image pixels covered by one tile
  const x0 = Math.max(0, Math.floor(-view.x / view.zoom / span))
  const y0 = Math.max(0, Math.floor(-view.y / view.zoom / span))
  const x1 = Math.min(
    Math.ceil(image.w / span) - 1,
    Math.floor((viewport.w - view.x) / view.zoom / span),
  )
  const y1 = Math.min(
    Math.ceil(image.h / span) - 1,
    Math.floor((viewport.h - view.y) / view.zoom / span),
  )
  const out: TileRef[] = []
  for (let ty = y0; ty <= y1; ty++) for (let tx = x0; tx <= x1; tx++) out.push({ level, tx, ty })
  return out
}

/** Keep at least part of the image reachable. */
export function clampPan(view: View, viewport: Size, image: Size, keep = 80): View {
  const w = image.w * view.zoom
  const h = image.h * view.zoom
  return {
    zoom: view.zoom,
    x: Math.min(viewport.w - keep, Math.max(keep - w, view.x)),
    y: Math.min(viewport.h - keep, Math.max(keep - h, view.y)),
  }
}

export const formatZoom = (z: number): string => `${Math.round(z * 100)}%`
