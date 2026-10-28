import i18n from '../i18n'
import { api } from '../lib/api'
import type { Box, CombineMode } from '../lib/mask'
import { useStore } from '../store/useStore'
import type { MaskLayer } from './maskLayer'

export interface Pt {
  x: number
  y: number
}

type Mods = { shift: boolean; alt: boolean }

/** Paint a round brush stamp (hard or soft) in mask-pixel coordinates. */
export function stamp(
  c: CanvasRenderingContext2D,
  x: number,
  y: number,
  radius: number,
  hardness: number,
): void {
  const r = Math.max(radius, 0.5)
  if (hardness >= 0.98) {
    c.fillStyle = '#fff'
    c.beginPath()
    c.arc(x, y, r, 0, Math.PI * 2)
    c.fill()
    return
  }
  const g = c.createRadialGradient(x, y, r * hardness, x, y, r)
  g.addColorStop(0, 'rgba(255,255,255,1)')
  g.addColorStop(1, 'rgba(255,255,255,0)')
  c.fillStyle = g
  c.beginPath()
  c.arc(x, y, r, 0, Math.PI * 2)
  c.fill()
}

export function stampLine(
  c: CanvasRenderingContext2D,
  a: Pt,
  b: Pt,
  radius: number,
  hardness: number,
): void {
  const dist = Math.hypot(b.x - a.x, b.y - a.y)
  const step = Math.max(0.75, radius * 0.18)
  const n = Math.max(1, Math.ceil(dist / step))
  for (let i = 0; i <= n; i++) {
    const t = i / n
    stamp(c, a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, radius, hardness)
  }
}

const PAINT_TOOLS = new Set(['brush', 'eraser', 'source', 'clone', 'heal'])

/** Turns pointer input on the canvas into selection edits. Image coordinates in, mask edits out. */
export class ToolController {
  /** Rubber-band box (image coordinates) drawn by the viewport, e.g. for "find similar". */
  guide: Box | null = null
  /** Polygon lasso in progress (image coordinates). */
  polygon: Pt[] = []
  /** Cursor position in image coordinates (for the brush outline). */
  cursor: Pt | null = null

  private layer: MaskLayer | null = null
  private down: Pt | null = null
  private last: Pt | null = null
  private lastEnd: Pt | null = null
  private lasso: Pt[] = []
  private busy = false
  private cloneStart: Pt | null = null
  private onChange: () => void

  constructor(onChange: () => void) {
    this.onChange = onChange
  }

  get active(): boolean {
    return this.layer?.editing === true || this.polygon.length > 0 || this.guide !== null
  }

  private mode(mods: Mods, fallback: CombineMode): CombineMode {
    return mods.shift ? 'add' : mods.alt ? 'subtract' : fallback
  }

  pointerDown(p: Pt, mods: Mods): void {
    const s = useStore.getState()
    const tool = s.tool
    if (!s.mask || !s.source || s.job.id) return
    this.down = p
    this.last = p
    const o = s.options

    if (tool === 'clone' || tool === 'heal') {
      if (mods.alt) {
        s.setCloneSource(p)
        this.down = null
        return
      }
      if (!s.cloneSource) {
        s.notify(i18n.t('remove.needSource'))
        this.down = null
        return
      }
      this.cloneStart = p
    }

    if (PAINT_TOOLS.has(tool)) {
      const layer = tool === 'source' ? s.source : s.mask
      this.layer = layer
      const paintMode: CombineMode =
        tool === 'eraser'
          ? 'subtract'
          : mods.alt && !(tool === 'clone' || tool === 'heal')
            ? 'subtract'
            : 'add'
      if (tool !== 'clone' && tool !== 'heal') s.pushUndo()
      layer.begin(tool === 'clone' || tool === 'heal' ? 'replace' : paintMode)
      const a = mods.shift && this.lastEnd ? this.lastEnd : p
      this.paint(a, p)
      return
    }

    switch (tool) {
      case 'rect':
      case 'ellipse':
      case 'lasso':
        s.pushUndo()
        this.layer = s.mask
        s.mask.begin(this.mode(mods, o.mode))
        this.lasso = [p]
        break
      case 'polygon':
        if (this.polygon.length === 0) {
          s.pushUndo()
          this.layer = s.mask
          s.mask.begin(this.mode(mods, o.mode))
        }
        this.polygon.push(p)
        this.drawPolygon(p)
        break
      case 'similar':
        this.guide = { x: p.x, y: p.y, w: 0, h: 0 }
        break
      case 'wand':
      case 'color':
        void this.pick(tool, p, this.mode(mods, o.mode))
        this.down = null
        break
      default:
        this.down = null
    }
  }

  pointerMove(p: Pt, mods: Mods): void {
    this.cursor = p
    const s = useStore.getState()
    const tool = s.tool
    if (this.down && this.layer?.editing && PAINT_TOOLS.has(tool)) {
      this.paint(this.last ?? p, p)
      this.last = p
    } else if (this.down && this.layer?.editing && (tool === 'rect' || tool === 'ellipse')) {
      this.drawBox(this.down, p, tool === 'ellipse', mods.shift)
    } else if (this.down && this.layer?.editing && tool === 'lasso') {
      this.lasso.push(p)
      this.drawLasso()
    } else if (tool === 'polygon' && this.polygon.length > 0) {
      this.drawPolygon(p)
    } else if (this.down && tool === 'similar' && this.guide) {
      this.guide = {
        x: Math.min(this.down.x, p.x),
        y: Math.min(this.down.y, p.y),
        w: Math.abs(p.x - this.down.x),
        h: Math.abs(p.y - this.down.y),
      }
    }
    this.onChange()
  }

  pointerUp(p: Pt): void {
    const s = useStore.getState()
    const tool = s.tool
    const layer = this.layer
    if (!this.down) return
    if (PAINT_TOOLS.has(tool) && layer?.editing) {
      layer.commit()
      this.lastEnd = p
      if ((tool === 'clone' || tool === 'heal') && s.cloneSource && this.cloneStart) {
        const dx = Math.round(s.cloneSource.x - this.cloneStart.x)
        const dy = Math.round(s.cloneSource.y - this.cloneStart.y)
        this.cloneStart = null
        this.layer = null
        this.down = null
        void s.runRemove({ offset: [dy, dx], method: tool })
        return
      }
    } else if ((tool === 'rect' || tool === 'ellipse') && layer?.editing) {
      const tiny = Math.hypot(p.x - this.down.x, p.y - this.down.y) < 2
      if (tiny) layer.cancel()
      else layer.commit()
    } else if (tool === 'lasso' && layer?.editing) {
      if (this.lasso.length > 2) layer.commit()
      else layer.cancel()
      this.lasso = []
    } else if (tool === 'similar' && this.guide) {
      const g = this.guide
      this.guide = null
      if (g.w >= 4 && g.h >= 4) void this.findSimilar(g)
    }
    if (tool !== 'polygon') this.layer = null
    this.down = null
    this.last = null
    s.bumpMask()
    this.onChange()
  }

  /** Close the polygon lasso (double-click or Enter). */
  finishPolygon(): void {
    if (this.polygon.length >= 3 && this.layer?.editing) this.layer.commit()
    else this.layer?.cancel()
    this.polygon = []
    this.layer = null
    this.down = null
    this.onChange()
  }

  cancel(): void {
    if (this.layer?.editing) this.layer.cancel()
    this.polygon = []
    this.lasso = []
    this.guide = null
    this.layer = null
    this.down = null
    this.onChange()
  }

  // ------------------------------------------------------------ drawing helpers

  private paint(a: Pt, b: Pt): void {
    const s = useStore.getState()
    const layer = this.layer
    if (!layer) return
    const sc = layer.scale
    stampLine(
      layer.draw,
      { x: a.x * sc, y: a.y * sc },
      { x: b.x * sc, y: b.y * sc },
      (s.options.size / 2) * sc,
      s.options.hardness,
    )
    layer.refresh()
  }

  private drawBox(a: Pt, b: Pt, ellipse: boolean, square: boolean): void {
    const layer = this.layer!
    const sc = layer.scale
    let w = b.x - a.x
    let h = b.y - a.y
    if (square) {
      const m = Math.max(Math.abs(w), Math.abs(h))
      w = Math.sign(w || 1) * m
      h = Math.sign(h || 1) * m
    }
    const c = layer.draw
    c.clearRect(0, 0, layer.width, layer.height)
    c.fillStyle = '#fff'
    c.beginPath()
    if (ellipse)
      c.ellipse(
        (a.x + w / 2) * sc,
        (a.y + h / 2) * sc,
        Math.abs(w / 2) * sc,
        Math.abs(h / 2) * sc,
        0,
        0,
        Math.PI * 2,
      )
    else c.rect(a.x * sc, a.y * sc, w * sc, h * sc)
    c.fill()
    layer.refresh()
  }

  private fillPath(pts: Pt[]): void {
    const layer = this.layer!
    const sc = layer.scale
    const c = layer.draw
    c.clearRect(0, 0, layer.width, layer.height)
    if (pts.length < 2) {
      layer.refresh()
      return
    }
    c.fillStyle = '#fff'
    c.beginPath()
    c.moveTo(pts[0].x * sc, pts[0].y * sc)
    for (const q of pts.slice(1)) c.lineTo(q.x * sc, q.y * sc)
    c.closePath()
    c.fill('evenodd')
    layer.refresh()
  }

  private drawLasso(): void {
    this.fillPath(this.lasso)
  }

  private drawPolygon(cursor: Pt): void {
    this.fillPath([...this.polygon, cursor])
  }

  // ------------------------------------------------------------ server-backed picks

  private async pick(tool: 'wand' | 'color', p: Pt, mode: CombineMode): Promise<void> {
    const s = useStore.getState()
    if (!s.doc || !s.mask || this.busy) return
    const x = Math.floor(p.x)
    const y = Math.floor(p.y)
    if (x < 0 || y < 0 || x >= s.doc.width || y >= s.doc.height) return
    this.busy = true
    try {
      const png =
        tool === 'wand'
          ? await api.wand(s.doc.id, s.version, x, y, s.options.tolerance, s.options.contiguous)
          : await api.colorRange(s.doc.id, s.version, x, y, s.options.colorTolerance)
      s.pushUndo()
      await s.mask.applyServerMask(png, mode)
      s.bumpMask()
    } catch (e) {
      s.notify((e as Error).message, 'error')
    } finally {
      this.busy = false
    }
  }

  private async findSimilar(g: Box): Promise<void> {
    const s = useStore.getState()
    if (!s.doc) return
    const box: [number, number, number, number] = [
      Math.round(g.x),
      Math.round(g.y),
      Math.round(g.w),
      Math.round(g.h),
    ]
    try {
      const hits = await api.findSimilar(s.doc.id, s.version, box, s.options.threshold)
      s.set({ candidates: hits, candidateKind: 'similar', similarBox: box })
    } catch (e) {
      s.notify((e as Error).message, 'error')
    }
    this.onChange()
  }
}
