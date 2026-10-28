import { api, type MaskPng } from '../lib/api'
import {
  alphaBounds,
  compositeOp,
  expandBox,
  grayToAlpha,
  maskScale,
  type Box,
  type CombineMode,
} from '../lib/mask'

export interface Snapshot {
  /** Region of the mask this snapshot covers, in mask pixels; null = empty mask. */
  box: Box | null
  alpha: Uint8Array
  bytes: number
}

const makeCanvas = (w: number, h: number): HTMLCanvasElement => {
  const c = document.createElement('canvas')
  c.width = Math.max(1, w)
  c.height = Math.max(1, h)
  return c
}

/**
 * A selection mask kept in an alpha canvas. Large images use a downscaled mask
 * (at most 4096 px on the long side); exported masks are scaled back up.
 */
export class MaskLayer {
  readonly imgW: number
  readonly imgH: number
  readonly scale: number
  readonly canvas: HTMLCanvasElement
  private readonly ctx: CanvasRenderingContext2D
  private shape: HTMLCanvasElement
  private shapeCtx: CanvasRenderingContext2D
  private base: HTMLCanvasElement | null = null
  private mode: CombineMode = 'add'
  private listeners = new Set<() => void>()
  /** Bumped on every change so views know to redraw. */
  rev = 0
  private boundsCache: { rev: number; box: Box | null } | null = null

  constructor(imgW: number, imgH: number) {
    this.imgW = imgW
    this.imgH = imgH
    this.scale = maskScale(imgW, imgH)
    const w = Math.max(1, Math.round(imgW * this.scale))
    const h = Math.max(1, Math.round(imgH * this.scale))
    this.canvas = makeCanvas(w, h)
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true })!
    this.shape = makeCanvas(w, h)
    this.shapeCtx = this.shape.getContext('2d')!
  }

  get width(): number {
    return this.canvas.width
  }
  get height(): number {
    return this.canvas.height
  }

  subscribe(fn: () => void): () => void {
    this.listeners.add(fn)
    return () => this.listeners.delete(fn)
  }

  private changed(): void {
    this.rev++
    this.boundsCache = null
    this.listeners.forEach((l) => l())
  }

  // ------------------------------------------------------------ reading state

  /** Bounding box of the selection in mask pixels (scanned in strips; cached). */
  bounds(): Box | null {
    if (this.boundsCache?.rev === this.rev) return this.boundsCache.box
    const { width: w, height: h } = this.canvas
    let box: Box | null = null
    const strip = 256
    for (let y = 0; y < h; y += strip) {
      const sh = Math.min(strip, h - y)
      const d = this.ctx.getImageData(0, y, w, sh)
      const b = alphaBounds(d.data, w, sh)
      if (b) {
        const top = y + b.y
        box = box
          ? {
              x: Math.min(box.x, b.x),
              y: Math.min(box.y, top),
              w: Math.max(box.x + box.w, b.x + b.w) - Math.min(box.x, b.x),
              h: Math.max(box.y + box.h, top + b.h) - Math.min(box.y, top),
            }
          : { x: b.x, y: top, w: b.w, h: b.h }
      }
    }
    this.boundsCache = { rev: this.rev, box }
    return box
  }

  isEmpty(): boolean {
    return this.bounds() === null
  }

  // ------------------------------------------------------------ undo support

  snapshot(): Snapshot {
    const box = this.bounds()
    if (!box) return { box: null, alpha: new Uint8Array(0), bytes: 0 }
    const d = this.ctx.getImageData(box.x, box.y, box.w, box.h).data
    const alpha = new Uint8Array(box.w * box.h)
    for (let i = 0; i < alpha.length; i++) alpha[i] = d[i * 4 + 3]
    return { box, alpha, bytes: alpha.length }
  }

  restore(s: Snapshot): void {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height)
    if (s.box) {
      const img = this.ctx.createImageData(s.box.w, s.box.h)
      for (let i = 0; i < s.alpha.length; i++) {
        img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = 255
        img.data[i * 4 + 3] = s.alpha[i]
      }
      this.ctx.putImageData(img, s.box.x, s.box.y)
    }
    this.changed()
  }

  // ------------------------------------------------------------ editing

  /** Start an interactive edit; draw into the shape layer, then commit. */
  begin(mode: CombineMode): void {
    this.mode = mode
    this.base = makeCanvas(this.canvas.width, this.canvas.height)
    this.base.getContext('2d')!.drawImage(this.canvas, 0, 0)
    this.shapeCtx.clearRect(0, 0, this.shape.width, this.shape.height)
  }

  get editing(): boolean {
    return this.base !== null
  }

  /** The 2D context of the in-progress shape (mask-pixel coordinates, white ink). */
  get draw(): CanvasRenderingContext2D {
    return this.shapeCtx
  }

  /** Re-composite base + shape under the current mode (call after drawing). */
  refresh(): void {
    if (!this.base) return
    const c = this.ctx
    c.globalCompositeOperation = 'source-over'
    c.clearRect(0, 0, this.canvas.width, this.canvas.height)
    if (this.mode === 'replace') {
      c.drawImage(this.shape, 0, 0)
    } else {
      c.drawImage(this.base, 0, 0)
      c.globalCompositeOperation = compositeOp(this.mode)
      c.drawImage(this.shape, 0, 0)
      c.globalCompositeOperation = 'source-over'
    }
    this.changed()
  }

  commit(): void {
    this.refresh()
    this.base = null
  }

  cancel(): void {
    if (!this.base) return
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height)
    this.ctx.drawImage(this.base, 0, 0)
    this.base = null
    this.changed()
  }

  /** One-shot: draw something and combine it. */
  apply(mode: CombineMode, drawFn: (ctx: CanvasRenderingContext2D) => void): void {
    this.begin(mode)
    drawFn(this.shapeCtx)
    this.commit()
  }

  clear(): void {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height)
    this.changed()
  }

  selectAll(): void {
    this.ctx.fillStyle = '#fff'
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height)
    this.changed()
  }

  invert(): void {
    const copy = makeCanvas(this.canvas.width, this.canvas.height)
    copy.getContext('2d')!.drawImage(this.canvas, 0, 0)
    this.ctx.globalCompositeOperation = 'source-over'
    this.ctx.fillStyle = '#fff'
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height)
    this.ctx.globalCompositeOperation = 'destination-out'
    this.ctx.drawImage(copy, 0, 0)
    this.ctx.globalCompositeOperation = 'source-over'
    this.changed()
  }

  /** Convert image coordinates to mask pixels. */
  toMask(v: number): number {
    return v * this.scale
  }

  addBox(b: Box, mode: CombineMode = 'add'): void {
    const s = this.scale
    this.apply(mode, (c) => {
      c.fillStyle = '#fff'
      c.fillRect(b.x * s, b.y * s, b.w * s, b.h * s)
    })
  }

  // ------------------------------------------------------------ server masks

  /** Combine a (cropped) mask PNG returned by the server into the selection. */
  async applyServerMask(
    png: MaskPng,
    mode: CombineMode,
    offsetBase: [number, number] = [0, 0],
  ): Promise<void> {
    const bmp = await createImageBitmap(png.blob)
    const tmp = makeCanvas(bmp.width, bmp.height)
    const tc = tmp.getContext('2d', { willReadFrequently: true })!
    tc.drawImage(bmp, 0, 0)
    const data = tc.getImageData(0, 0, bmp.width, bmp.height)
    grayToAlpha(data.data)
    tc.putImageData(data, 0, 0)
    const s = this.scale
    const ox = (png.offset[0] + offsetBase[0]) * s
    const oy = (png.offset[1] + offsetBase[1]) * s
    this.apply(mode, (c) => {
      c.imageSmoothingEnabled = false
      c.drawImage(tmp, ox, oy, bmp.width * s, bmp.height * s)
    })
    bmp.close()
  }

  /** Replace the selection with a mask PNG (offset is in mask pixels, for morph results). */
  async replaceWithMaskPixels(png: MaskPng, cropOrigin: [number, number]): Promise<void> {
    const bmp = await createImageBitmap(png.blob)
    const tmp = makeCanvas(bmp.width, bmp.height)
    const tc = tmp.getContext('2d', { willReadFrequently: true })!
    tc.drawImage(bmp, 0, 0)
    const data = tc.getImageData(0, 0, bmp.width, bmp.height)
    grayToAlpha(data.data)
    tc.putImageData(data, 0, 0)
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height)
    this.ctx.drawImage(tmp, cropOrigin[0] + png.offset[0], cropOrigin[1] + png.offset[1])
    bmp.close()
    this.changed()
  }

  /** Run a server-side mask operation (grow, shrink, smooth, feather) in mask pixels. */
  async morph(op: 'grow' | 'shrink' | 'smooth' | 'feather', imagePx: number): Promise<void> {
    const box = this.bounds()
    if (!box) return
    const px = imagePx * this.scale
    const pad = Math.ceil(px) + 3
    const crop = expandBox(box, pad, this.canvas.width, this.canvas.height)
    const src = makeCanvas(crop.w, crop.h)
    src.getContext('2d')!.drawImage(this.canvas, -crop.x, -crop.y)
    const blob = await canvasBlob(src)
    const res = await api.morph(
      op,
      op === 'feather' ? Math.max(px, 0.5) : Math.max(1, Math.round(px)),
      blob,
    )
    await this.replaceWithMaskPixels(res, [crop.x, crop.y])
  }

  // ------------------------------------------------------------ export

  /**
   * The selection as a PNG cropped to its bounding box at full image resolution.
   * Returns null for an empty selection. ``offset`` is [x, y] in image pixels.
   */
  async exportPng(): Promise<{ blob: Blob; offset: [number, number] } | null> {
    const b = this.bounds()
    if (!b) return null
    const s = this.scale
    const x0 = Math.max(0, Math.floor(b.x / s))
    const y0 = Math.max(0, Math.floor(b.y / s))
    const x1 = Math.min(this.imgW, Math.ceil((b.x + b.w) / s))
    const y1 = Math.min(this.imgH, Math.ceil((b.y + b.h) / s))
    const out = makeCanvas(x1 - x0, y1 - y0)
    const c = out.getContext('2d')!
    c.imageSmoothingEnabled = s < 1
    c.imageSmoothingQuality = 'high'
    c.drawImage(this.canvas, b.x, b.y, b.w, b.h, 0, 0, x1 - x0, y1 - y0)
    return { blob: await canvasBlob(out), offset: [x0, y0] }
  }
}

export const canvasBlob = (c: HTMLCanvasElement): Promise<Blob> =>
  new Promise((resolve, reject) =>
    c.toBlob((b) => (b ? resolve(b) : reject(new Error('could not encode mask'))), 'image/png'),
  )
