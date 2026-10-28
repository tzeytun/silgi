import type { Box } from '../lib/mask'
import type { MaskLayer } from './maskLayer'

const make = (w: number, h: number): HTMLCanvasElement => {
  const c = document.createElement('canvas')
  c.width = Math.max(1, w)
  c.height = Math.max(1, h)
  return c
}

/** A mask tinted with one colour; rebuilt only when the mask changes. */
export class TintCache {
  private canvas: HTMLCanvasElement | null = null
  private rev = -1
  private key = ''

  get(layer: MaskLayer, color: string): HTMLCanvasElement {
    const key = `${layer.width}x${layer.height}${color}`
    if (this.canvas && this.rev === layer.rev && this.key === key) return this.canvas
    if (!this.canvas || this.canvas.width !== layer.width || this.canvas.height !== layer.height) {
      this.canvas = make(layer.width, layer.height)
    }
    const c = this.canvas.getContext('2d')!
    c.globalCompositeOperation = 'source-over'
    c.clearRect(0, 0, this.canvas.width, this.canvas.height)
    c.drawImage(layer.canvas, 0, 0)
    c.globalCompositeOperation = 'source-in'
    c.fillStyle = color
    c.fillRect(0, 0, this.canvas.width, this.canvas.height)
    c.globalCompositeOperation = 'source-over'
    this.rev = layer.rev
    this.key = key
    return this.canvas
  }
}

const stripes = (): HTMLCanvasElement => {
  const p = make(8, 8)
  const c = p.getContext('2d')!
  c.fillStyle = '#fff'
  c.fillRect(0, 0, 8, 8)
  c.fillStyle = '#000'
  c.beginPath()
  c.moveTo(0, 8)
  c.lineTo(8, 0)
  c.lineTo(8, 4)
  c.lineTo(4, 8)
  c.closePath()
  c.moveTo(0, 4)
  c.lineTo(4, 0)
  c.lineTo(0, 0)
  c.closePath()
  c.fill()
  return p
}

/**
 * The selection outline as a thin border canvas covering only the mask's bounding box,
 * filled with an animated diagonal pattern ("marching ants").
 */
export class AntsCache {
  private border: HTMLCanvasElement | null = null
  private out: HTMLCanvasElement | null = null
  private rev = -1
  private thickness = 0
  private box: Box | null = null
  private pattern: HTMLCanvasElement = stripes()

  get(
    layer: MaskLayer,
    thickness: number,
    phase: number,
  ): { canvas: HTMLCanvasElement; box: Box } | null {
    if (this.rev !== layer.rev || this.thickness !== thickness) {
      this.rev = layer.rev
      this.thickness = thickness
      this.rebuild(layer, thickness)
    }
    if (!this.border || !this.out || !this.box) return null
    const c = this.out.getContext('2d')!
    c.globalCompositeOperation = 'source-over'
    c.clearRect(0, 0, this.out.width, this.out.height)
    c.drawImage(this.border, 0, 0)
    c.globalCompositeOperation = 'source-in'
    const pat = c.createPattern(this.pattern, 'repeat')
    if (pat) {
      pat.setTransform(new DOMMatrix().translate(phase % 8, phase % 8))
      c.fillStyle = pat
      c.fillRect(0, 0, this.out.width, this.out.height)
    }
    c.globalCompositeOperation = 'source-over'
    return { canvas: this.out, box: this.box }
  }

  private rebuild(layer: MaskLayer, k: number): void {
    const b = layer.bounds()
    if (!b) {
      this.border = this.out = null
      this.box = null
      return
    }
    const pad = k + 1
    this.box = { x: b.x - pad, y: b.y - pad, w: b.w + 2 * pad, h: b.h + 2 * pad }
    const edge = make(this.box.w, this.box.h)
    const c = edge.getContext('2d')!
    c.imageSmoothingEnabled = false
    c.drawImage(layer.canvas, b.x, b.y, b.w, b.h, pad, pad, b.w, b.h)
    // erode: remove the mask shifted in all directions, leaving a border ~k px wide
    const copy = make(edge.width, edge.height)
    copy.getContext('2d')!.drawImage(edge, 0, 0)
    c.globalCompositeOperation = 'destination-out'
    for (const [dx, dy] of [
      [k, 0],
      [-k, 0],
      [0, k],
      [0, -k],
      [k, k],
      [-k, -k],
      [k, -k],
      [-k, k],
    ])
      c.drawImage(copy, dx, dy)
    c.globalCompositeOperation = 'source-over'
    this.border = edge
    this.out = make(edge.width, edge.height)
  }
}
