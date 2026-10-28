/* The viewport is an imperative canvas renderer: it reads refs and mutable controller state
   inside render callbacks by design, so the React Compiler ref/immutability rules do not fit. */
/* eslint-disable react-hooks/refs, react-hooks/immutability */
import { useCallback, useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { tileUrl, type ImageInfo } from '../lib/api'
import {
  clampPan,
  pickLevel,
  toImage,
  toScreen,
  visibleTiles,
  zoomAt,
  type View,
} from '../lib/geometry'
import { useStore } from '../store/useStore'
import { AntsCache, TintCache } from './overlay'
import { ToolController } from './tools'

interface TileEntry {
  img: HTMLImageElement
  ready: boolean
}

const MAX_TILES = 700
const tiles = new Map<string, TileEntry>()

const PINK = 'rgba(239, 91, 123, 0.5)'
const SOURCE = 'rgba(76, 123, 244, 0.45)'

function getTile(
  doc: ImageInfo,
  version: number,
  level: number,
  tx: number,
  ty: number,
  onLoad: () => void,
): HTMLImageElement | null {
  const key = `${doc.id}/${version}/${level}/${tx}/${ty}`
  let e = tiles.get(key)
  if (e) {
    tiles.delete(key)
    tiles.set(key, e)
  } else {
    const img = new Image()
    e = { img, ready: false }
    const entry = e
    img.onload = () => {
      entry.ready = true
      onLoad()
    }
    img.src = tileUrl(doc.id, version, level, tx, ty)
    tiles.set(key, e)
    if (tiles.size > MAX_TILES) tiles.delete(tiles.keys().next().value as string)
  }
  return e.ready ? e.img : null
}

let checkerKey = ''
let checkerPattern: CanvasPattern | null = null

/** Transparency checkerboard (colours follow the theme). */
function checker(ctx: CanvasRenderingContext2D): CanvasPattern | null {
  const css = getComputedStyle(document.documentElement)
  const a = css.getPropertyValue('--checker-a').trim() || '#fff'
  const b = css.getPropertyValue('--checker-b').trim() || '#e6e8ec'
  if (checkerPattern && checkerKey === a + b) return checkerPattern
  const c = document.createElement('canvas')
  c.width = c.height = 16
  const x = c.getContext('2d')!
  x.fillStyle = a
  x.fillRect(0, 0, 16, 16)
  x.fillStyle = b
  x.fillRect(0, 0, 8, 8)
  x.fillRect(8, 8, 8, 8)
  checkerKey = a + b
  checkerPattern = ctx.createPattern(c, 'repeat')
  return checkerPattern
}

function drawVersion(
  ctx: CanvasRenderingContext2D,
  doc: ImageInfo,
  version: number,
  view: View,
  vp: { w: number; h: number },
  schedule: () => void,
): void {
  const level = pickLevel(view.zoom, doc.levels)
  const top = doc.levels - 1
  ctx.imageSmoothingEnabled = view.zoom < 2
  ctx.imageSmoothingQuality = 'high'
  const drawTile = (img: HTMLImageElement, lv: number, tx: number, ty: number) => {
    const span = doc.tile_size * 2 ** lv
    const p = toScreen(view, tx * span, ty * span)
    const q = toScreen(
      view,
      tx * span + img.naturalWidth * 2 ** lv,
      ty * span + img.naturalHeight * 2 ** lv,
    )
    const x0 = Math.floor(p.x)
    const y0 = Math.floor(p.y)
    ctx.drawImage(img, x0, y0, Math.ceil(q.x) - x0, Math.ceil(q.y) - y0)
  }
  // the whole image at the coarsest level first, so panning never shows holes
  const base = getTile(doc, version, top, 0, 0, schedule)
  if (base) drawTile(base, top, 0, 0)
  if (level === top) return
  for (const t of visibleTiles(view, vp, { w: doc.width, h: doc.height }, doc.tile_size, level)) {
    const img = getTile(doc, version, t.level, t.tx, t.ty, schedule)
    if (img) drawTile(img, t.level, t.tx, t.ty)
  }
}

export function Viewport() {
  const { t } = useTranslation()
  const wrap = useRef<HTMLDivElement>(null)
  const canvas = useRef<HTMLCanvasElement>(null)
  const raf = useRef(0)
  const tint = useRef({ mask: new TintCache(), source: new TintCache() })
  const ants = useRef(new AntsCache())
  const phase = useRef(0)
  const touched = useRef(false)
  const spaceDown = useRef(false)
  const pointers = useRef(new Map<number, { x: number; y: number }>())
  const gesture = useRef<null | {
    kind: 'pan' | 'pinch' | 'divider'
    last: { x: number; y: number }
    dist?: number
  }>(null)
  const [cursor, setCursor] = useState('crosshair')
  const doc = useStore((s) => s.doc)

  const schedule = useCallback(() => {
    if (raf.current) return
    raf.current = requestAnimationFrame(() => {
      raf.current = 0
      render()
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const scheduleRef = useRef<() => void>(() => {})
  const [ctrl] = useState(() => new ToolController(() => scheduleRef.current()))

  function render(): void {
    const c = canvas.current
    const s = useStore.getState()
    if (!c) return
    const ctx = c.getContext('2d')!
    const dpr = window.devicePixelRatio || 1
    const { w, h } = s.viewport
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
      c.width = Math.round(w * dpr)
      c.height = Math.round(h * dpr)
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, w, h)
    const d = s.doc
    if (!d) return
    const view = s.view
    const vp = s.viewport

    if (d.has_alpha) {
      const pat = checker(ctx)
      if (pat) {
        ctx.fillStyle = pat
        ctx.fillRect(view.x, view.y, d.width * view.zoom, d.height * view.zoom)
      }
    }

    if (s.compare) {
      const dx = Math.round(vp.w * s.compareAt)
      ctx.save()
      ctx.beginPath()
      ctx.rect(0, 0, dx, vp.h)
      ctx.clip()
      drawVersion(ctx, d, 0, view, vp, schedule)
      ctx.restore()
      ctx.save()
      ctx.beginPath()
      ctx.rect(dx, 0, vp.w - dx, vp.h)
      ctx.clip()
      drawVersion(ctx, d, s.version, view, vp, schedule)
      ctx.restore()
      ctx.fillStyle = '#fff'
      ctx.fillRect(dx - 1, 0, 2, vp.h)
      ctx.strokeStyle = 'rgba(0,0,0,0.35)'
      ctx.strokeRect(dx - 1.5, 0, 3, vp.h)
      ctx.fillStyle = '#fff'
      ctx.beginPath()
      ctx.arc(dx, vp.h / 2, 14, 0, Math.PI * 2)
      ctx.fill()
      ctx.stroke()
      ctx.fillStyle = '#15171c'
      ctx.font = '600 11px Inter, sans-serif'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText('↔', dx, vp.h / 2)
    } else {
      drawVersion(ctx, d, s.version, view, vp, schedule)
    }

    // fast low-resolution preview of a running job
    const pv = s.job.preview
    if (pv) {
      const p = toScreen(view, pv.box.x, pv.box.y)
      ctx.imageSmoothingEnabled = true
      ctx.drawImage(pv.bitmap, p.x, p.y, pv.box.w * view.zoom, pv.box.h * view.zoom)
    }

    const imgRect = { x: view.x, y: view.y, w: d.width * view.zoom, h: d.height * view.zoom }
    const pixelated = view.zoom / (s.mask?.scale ?? 1) > 1
    if (s.mask && !s.mask.isEmpty()) {
      ctx.imageSmoothingEnabled = !pixelated
      ctx.drawImage(tint.current.mask.get(s.mask, PINK), imgRect.x, imgRect.y, imgRect.w, imgRect.h)
      const thick = Math.min(8, Math.max(1, Math.ceil(1.5 / (view.zoom / s.mask.scale))))
      const a = ants.current.get(s.mask, thick, phase.current)
      if (a) {
        const k = view.zoom / s.mask.scale
        ctx.imageSmoothingEnabled = false
        ctx.drawImage(
          a.canvas,
          imgRect.x + a.box.x * k,
          imgRect.y + a.box.y * k,
          a.box.w * k,
          a.box.h * k,
        )
      }
    }
    if (s.source && !s.source.isEmpty()) {
      ctx.imageSmoothingEnabled = !pixelated
      ctx.globalAlpha = 1
      ctx.drawImage(
        tint.current.source.get(s.source, SOURCE),
        imgRect.x,
        imgRect.y,
        imgRect.w,
        imgRect.h,
      )
    }

    // candidate boxes (text finder / find similar)
    for (const cd of s.candidates) {
      const p = toScreen(view, cd.x, cd.y)
      const wpx = cd.w * view.zoom
      const hpx = cd.h * view.zoom
      ctx.lineWidth = 3
      ctx.strokeStyle = 'rgba(255,255,255,0.95)'
      ctx.strokeRect(p.x, p.y, wpx, hpx)
      ctx.lineWidth = 1.5
      ctx.strokeStyle = '#ef5b7b'
      ctx.setLineDash(cd.is_source ? [] : [6, 4])
      ctx.strokeRect(p.x, p.y, wpx, hpx)
      ctx.setLineDash([])
    }

    const g = ctrl.guide
    if (g) {
      const p = toScreen(view, g.x, g.y)
      ctx.lineWidth = 1.5
      ctx.strokeStyle = '#fff'
      ctx.strokeRect(p.x, p.y, g.w * view.zoom, g.h * view.zoom)
      ctx.setLineDash([5, 4])
      ctx.strokeStyle = '#15171c'
      ctx.strokeRect(p.x, p.y, g.w * view.zoom, g.h * view.zoom)
      ctx.setLineDash([])
    }

    if (s.cloneSource && (s.tool === 'clone' || s.tool === 'heal')) {
      const p = toScreen(view, s.cloneSource.x, s.cloneSource.y)
      ctx.strokeStyle = '#fff'
      ctx.lineWidth = 3
      ctx.beginPath()
      ctx.arc(p.x, p.y, 9, 0, Math.PI * 2)
      ctx.moveTo(p.x - 14, p.y)
      ctx.lineTo(p.x + 14, p.y)
      ctx.moveTo(p.x, p.y - 14)
      ctx.lineTo(p.x, p.y + 14)
      ctx.stroke()
      ctx.strokeStyle = '#15171c'
      ctx.lineWidth = 1.2
      ctx.stroke()
    }

    // brush outline
    const cur = ctrl.cursor
    if (cur && ['brush', 'eraser', 'source', 'clone', 'heal'].includes(s.tool) && !s.job.id) {
      const p = toScreen(view, cur.x, cur.y)
      const r = (s.options.size / 2) * view.zoom
      ctx.lineWidth = 2.5
      ctx.strokeStyle = 'rgba(255,255,255,0.9)'
      ctx.beginPath()
      ctx.arc(p.x, p.y, Math.max(r, 1.5), 0, Math.PI * 2)
      ctx.stroke()
      ctx.lineWidth = 1
      ctx.strokeStyle = '#15171c'
      ctx.stroke()
    }
  }

  useEffect(() => {
    scheduleRef.current = schedule
  }, [schedule])

  // redraw on any relevant state change
  useEffect(() => {
    const unsub = useStore.subscribe(() => schedule())
    return unsub
  }, [schedule])

  // marching ants
  useEffect(() => {
    const id = window.setInterval(() => {
      const s = useStore.getState()
      if (s.mask && !s.mask.isEmpty() && !document.hidden) {
        phase.current = (phase.current + 1) % 8
        schedule()
      }
    }, 140)
    return () => clearInterval(id)
  }, [schedule])

  // size tracking
  useEffect(() => {
    const el = wrap.current
    if (!el) return
    const ro = new ResizeObserver(() => {
      const r = el.getBoundingClientRect()
      useStore.getState().setViewport({ w: r.width, h: r.height })
      if (!touched.current) useStore.getState().fit()
      schedule()
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [schedule])

  useEffect(() => {
    touched.current = false
    if (doc) useStore.getState().fit()
  }, [doc])

  // wheel zoom (needs a non-passive listener)
  useEffect(() => {
    const el = canvas.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const s = useStore.getState()
      if (!s.doc) return
      touched.current = true
      const r = el.getBoundingClientRect()
      const factor = Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0015))
      s.setView(zoomAt(s.view, factor, { x: e.clientX - r.left, y: e.clientY - r.top }))
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [])

  // space = temporary hand tool
  useEffect(() => {
    const typing = (e: KeyboardEvent) =>
      e.target instanceof HTMLElement && /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)
    const down = (e: KeyboardEvent) => {
      if (e.code === 'Space' && !typing(e) && !(e.target instanceof HTMLButtonElement)) {
        e.preventDefault()
        spaceDown.current = true
        setCursor('grab')
      }
      if (typing(e)) return
      if (e.key === 'Escape') ctrl.cancel()
      if (e.key === 'Enter') ctrl.finishPolygon()
    }
    const up = (e: KeyboardEvent) => {
      if (e.code === 'Space') {
        spaceDown.current = false
        setCursor('crosshair')
      }
    }
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    return () => {
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
    }
  }, [ctrl])

  const local = (e: React.PointerEvent | React.MouseEvent) => {
    const r = canvas.current!.getBoundingClientRect()
    return { x: e.clientX - r.left, y: e.clientY - r.top }
  }

  const onPointerDown = (e: React.PointerEvent) => {
    const s = useStore.getState()
    if (!s.doc) return
    canvas.current!.setPointerCapture(e.pointerId)
    const p = local(e)
    pointers.current.set(e.pointerId, p)
    touched.current = true
    if (pointers.current.size === 2) {
      ctrl.cancel()
      const [a, b] = [...pointers.current.values()]
      gesture.current = {
        kind: 'pinch',
        last: { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 },
        dist: Math.hypot(a.x - b.x, a.y - b.y),
      }
      return
    }
    if (s.compare && Math.abs(p.x - s.viewport.w * s.compareAt) < 18) {
      gesture.current = { kind: 'divider', last: p }
      return
    }
    if (spaceDown.current || s.tool === 'hand' || e.button === 1) {
      gesture.current = { kind: 'pan', last: p }
      setCursor('grabbing')
      return
    }
    if (e.button !== 0) return
    const ip = toImage(s.view, p.x, p.y)
    // accepting a candidate box by clicking it
    if ((s.tool === 'text' || s.tool === 'similar') && s.candidates.length && s.mask) {
      const hit = s.candidates.find(
        (c) => ip.x >= c.x && ip.x <= c.x + c.w && ip.y >= c.y && ip.y <= c.y + c.h,
      )
      if (hit) {
        s.pushUndo()
        s.mask.addBox(hit, s.options.mode === 'replace' ? 'add' : s.options.mode)
        s.set({ candidates: s.candidates.filter((c) => c !== hit) })
        return
      }
    }
    ctrl.pointerDown(ip, { shift: e.shiftKey, alt: e.altKey })
    schedule()
  }

  const onPointerMove = (e: React.PointerEvent) => {
    const s = useStore.getState()
    if (!s.doc) return
    const p = local(e)
    if (pointers.current.has(e.pointerId)) pointers.current.set(e.pointerId, p)
    const g = gesture.current
    if (g?.kind === 'pinch' && pointers.current.size >= 2) {
      const [a, b] = [...pointers.current.values()]
      const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }
      const dist = Math.hypot(a.x - b.x, a.y - b.y)
      let v = zoomAt(s.view, dist / (g.dist || dist), mid)
      v = { ...v, x: v.x + mid.x - g.last.x, y: v.y + mid.y - g.last.y }
      s.setView(v)
      gesture.current = { kind: 'pinch', last: mid, dist }
      return
    }
    if (g?.kind === 'pan') {
      const v: View = { ...s.view, x: s.view.x + p.x - g.last.x, y: s.view.y + p.y - g.last.y }
      s.setView(clampPan(v, s.viewport, { w: s.doc.width, h: s.doc.height }))
      gesture.current = { kind: 'pan', last: p }
      return
    }
    if (g?.kind === 'divider') {
      s.setCompareAt(Math.min(0.98, Math.max(0.02, p.x / s.viewport.w)))
      return
    }
    ctrl.pointerMove(toImage(s.view, p.x, p.y), { shift: e.shiftKey, alt: e.altKey })
  }

  const onPointerUp = (e: React.PointerEvent) => {
    const s = useStore.getState()
    pointers.current.delete(e.pointerId)
    if (gesture.current) {
      if (pointers.current.size < 2) gesture.current = null
      setCursor(spaceDown.current ? 'grab' : s.tool === 'hand' ? 'grab' : 'crosshair')
      return
    }
    if (!s.doc) return
    const p = toImage(s.view, local(e).x, local(e).y)
    ctrl.pointerUp(p)
    schedule()
  }

  const tool = useStore((s) => s.tool)
  const effectiveCursor = tool === 'hand' ? 'grab' : cursor

  return (
    <div ref={wrap} className="relative h-full w-full overflow-hidden bg-bg" data-testid="viewport">
      <canvas
        ref={canvas}
        aria-label={t('a11y.canvas')}
        role="img"
        data-testid="canvas"
        className="absolute inset-0 h-full w-full touch-none"
        style={{ cursor: effectiveCursor }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onPointerLeave={() => {
          ctrl.cursor = null
          schedule()
        }}
        onDoubleClick={() => ctrl.finishPolygon()}
        onContextMenu={(e) => e.preventDefault()}
      />
    </div>
  )
}
