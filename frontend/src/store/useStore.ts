import { create } from 'zustand'
import i18n from '../i18n'
import { MaskLayer, type Snapshot } from '../canvas/maskLayer'
import {
  api,
  watchJob,
  type Candidate,
  type ImageInfo,
  type Method,
  type Quality,
  type RemoveOptions,
} from '../lib/api'
import { fitView, type Size, type View } from '../lib/geometry'
import { trimSnapshots, type Box, type CombineMode } from '../lib/mask'

export type ToolId =
  | 'brush'
  | 'eraser'
  | 'rect'
  | 'ellipse'
  | 'lasso'
  | 'polygon'
  | 'wand'
  | 'color'
  | 'text'
  | 'similar'
  | 'source'
  | 'clone'
  | 'heal'
  | 'hand'

export interface ToolOptions {
  size: number
  hardness: number
  tolerance: number
  colorTolerance: number
  contiguous: boolean
  mode: CombineMode
  threshold: number
  amount: number
}

export interface WatermarkRef {
  id: string
  name: string
  width?: number
  height?: number
  estimated?: boolean
}

export interface JobState {
  id: string | null
  fraction: number
  message: string
  preview: { bitmap: ImageBitmap; box: Box } | null
}

interface UndoEntry {
  mask: Snapshot
  version: number
  path: PathEntry[]
  bytes: number
}

export interface PathEntry {
  version: number
  label: string
  reason?: string
}

export interface BatchItem {
  id: string
  file: File
  name: string
  status: 'waiting' | 'working' | 'done' | 'skipped' | 'error'
  method: Method | 'default'
  message?: string
  doc?: ImageInfo
  version: number
}

export type Theme = 'system' | 'light' | 'dark'

interface State {
  doc: ImageInfo | null
  mask: MaskLayer | null
  source: MaskLayer | null
  maskRev: number
  version: number
  path: PathEntry[]
  undoStack: UndoEntry[]
  redoStack: UndoEntry[]

  tool: ToolId
  options: ToolOptions
  view: View
  viewport: Size
  compare: boolean
  compareAt: number
  cloneSource: { x: number; y: number } | null

  method: Method
  quality: Quality
  feather: number
  seam: 'none' | 'poisson'
  seed: number
  watermarks: WatermarkRef[]
  watermarkId: string | null
  watermarkNote: string | null
  job: JobState
  lastDecision: string | null

  candidates: Candidate[]
  candidateKind: 'text' | 'similar' | null
  similarBox: [number, number, number, number] | null

  screen: 'editor' | 'batch'
  exportOpen: boolean
  batch: BatchItem[]
  theme: Theme
  toast: { text: string; kind: 'info' | 'error' } | null
  busy: boolean

  // actions
  openFile: (file: File) => Promise<void>
  openSample: () => Promise<void>
  closeDoc: () => void
  setTool: (t: ToolId) => void
  setOptions: (o: Partial<ToolOptions>) => void
  setView: (v: View) => void
  setViewport: (s: Size) => void
  fit: () => void
  setCompare: (on: boolean) => void
  setCompareAt: (v: number) => void
  setCloneSource: (p: { x: number; y: number } | null) => void
  set: (p: Partial<State>) => void
  notify: (text: string, kind?: 'info' | 'error') => void
  pushUndo: () => void
  undo: () => void
  redo: () => void
  jumpTo: (index: number) => void
  runRemove: (over?: {
    offset?: [number, number]
    mask?: MaskLayer
    method?: Method
  }) => Promise<void>
  cancelJob: () => Promise<void>
  setTheme: (t: Theme) => void
  bumpMask: () => void
}

const DEFAULT_OPTIONS: ToolOptions = {
  size: 40,
  hardness: 0.9,
  tolerance: 0.12,
  colorTolerance: 20,
  contiguous: true,
  mode: 'add',
  threshold: 0.6,
  amount: 3,
}

const EMPTY_JOB: JobState = { id: null, fraction: 0, message: '', preview: null }

let toastTimer: ReturnType<typeof setTimeout> | undefined

const readTheme = (): Theme => {
  try {
    const t = localStorage.getItem('silgi.theme')
    if (t === 'light' || t === 'dark') return t
  } catch {
    /* storage unavailable */
  }
  return 'system'
}

export const useStore = create<State>((set, get) => {
  const openInfo = (info: ImageInfo) => {
    const previous = get().doc
    if (previous) void api.close(previous.id)
    const mask = new MaskLayer(info.width, info.height)
    const source = new MaskLayer(info.width, info.height)
    mask.subscribe(() => set({ maskRev: mask.rev + source.rev }))
    source.subscribe(() => set({ maskRev: mask.rev + source.rev }))
    const wms = (info.watermarks ?? []).map((w) => ({ id: w.id, name: w.name }))
    set({
      doc: info,
      mask,
      source,
      version: 0,
      path: [{ version: 0, label: 'original' }],
      undoStack: [],
      redoStack: [],
      candidates: [],
      candidateKind: null,
      cloneSource: null,
      watermarks: wms,
      watermarkId: null,
      watermarkNote: null,
      lastDecision: null,
      compare: false,
      screen: 'editor',
      job: EMPTY_JOB,
      maskRev: 0,
    })
    get().fit()
  }

  const currentEntry = (): UndoEntry => {
    const { mask, version, path } = get()
    const snap = mask!.snapshot()
    return { mask: snap, version, path: path.slice(), bytes: snap.bytes }
  }

  const restore = (e: UndoEntry) => {
    get().mask!.restore(e.mask)
    set({ version: e.version, path: e.path, candidates: [], candidateKind: null })
  }

  return {
    doc: null,
    mask: null,
    source: null,
    maskRev: 0,
    version: 0,
    path: [],
    undoStack: [],
    redoStack: [],
    tool: 'brush',
    options: DEFAULT_OPTIONS,
    view: { zoom: 1, x: 0, y: 0 },
    viewport: { w: 800, h: 600 },
    compare: false,
    compareAt: 0.5,
    cloneSource: null,
    method: 'auto',
    quality: 'balanced',
    feather: 0,
    seam: 'none',
    seed: 0,
    watermarks: [],
    watermarkId: null,
    watermarkNote: null,
    job: EMPTY_JOB,
    lastDecision: null,
    candidates: [],
    candidateKind: null,
    similarBox: null,
    screen: 'editor',
    exportOpen: false,
    batch: [],
    theme: readTheme(),
    toast: null,
    busy: false,

    set: (p) => set(p),
    bumpMask: () => {
      const { mask, source } = get()
      if (mask && source) set({ maskRev: mask.rev + source.rev })
    },

    notify: (text, kind = 'info') => {
      clearTimeout(toastTimer)
      set({ toast: { text, kind } })
      toastTimer = setTimeout(() => set({ toast: null }), kind === 'error' ? 7000 : 3500)
    },

    openFile: async (file) => {
      set({ busy: true })
      try {
        openInfo(await api.upload(file))
      } catch (e) {
        get().notify((e as Error).message, 'error')
      } finally {
        set({ busy: false })
      }
    },

    openSample: async () => {
      set({ busy: true })
      try {
        openInfo(await api.sample())
      } catch (e) {
        get().notify((e as Error).message, 'error')
      } finally {
        set({ busy: false })
      }
    },

    closeDoc: () => {
      const d = get().doc
      if (d) void api.close(d.id)
      set({
        doc: null,
        mask: null,
        source: null,
        path: [],
        undoStack: [],
        redoStack: [],
        job: EMPTY_JOB,
      })
    },

    setTool: (tool) =>
      set({ tool, candidates: tool === 'text' || tool === 'similar' ? get().candidates : [] }),
    setOptions: (o) => set({ options: { ...get().options, ...o } }),
    setView: (view) => set({ view }),
    setViewport: (viewport) => set({ viewport }),
    fit: () => {
      const { doc, viewport } = get()
      if (doc) set({ view: fitView(viewport, { w: doc.width, h: doc.height }) })
    },
    setCompare: (compare) => set({ compare }),
    setCompareAt: (compareAt) => set({ compareAt }),
    setCloneSource: (cloneSource) => set({ cloneSource }),

    pushUndo: () => {
      if (!get().mask) return
      const entry = currentEntry()
      set({ undoStack: trimSnapshots([...get().undoStack, entry]), redoStack: [] })
    },

    undo: () => {
      const { undoStack, redoStack, mask } = get()
      if (!mask || undoStack.length === 0 || get().job.id) return
      const prev = undoStack[undoStack.length - 1]
      set({ undoStack: undoStack.slice(0, -1), redoStack: [...redoStack, currentEntry()] })
      restore(prev)
    },

    redo: () => {
      const { undoStack, redoStack, mask } = get()
      if (!mask || redoStack.length === 0 || get().job.id) return
      const next = redoStack[redoStack.length - 1]
      set({ redoStack: redoStack.slice(0, -1), undoStack: [...undoStack, currentEntry()] })
      restore(next)
    },

    jumpTo: (index) => {
      const { path, version, mask } = get()
      const target = path[index]
      if (!mask || !target || target.version === version || get().job.id) return
      get().pushUndo()
      set({ version: target.version })
    },

    setTheme: (theme) => {
      set({ theme })
      try {
        if (theme === 'system') localStorage.removeItem('silgi.theme')
        else localStorage.setItem('silgi.theme', theme)
      } catch {
        /* ignore */
      }
      if (theme === 'system') delete document.documentElement.dataset.theme
      else document.documentElement.dataset.theme = theme
    },

    cancelJob: async () => {
      const id = get().job.id
      if (id) await api.cancel(id)
    },

    runRemove: async (over) => {
      const s = get()
      const { doc } = s
      const layer = over?.mask ?? s.mask
      if (!doc || !layer || s.job.id) return
      const exported = await layer.exportPng()
      const wm = s.watermarks.find((w) => w.id === s.watermarkId)
      if (!exported && !(wm && (s.method === 'alpha' || s.method === 'auto'))) {
        s.notify(i18n.t('remove.noSelection'))
        return
      }
      const opts: RemoveOptions = {
        version: s.version,
        method: over?.method ?? s.method,
        quality: s.quality,
        watermark_id: wm?.id ?? null,
        offset: over?.offset ?? null,
        seed: s.seed,
        feather: s.feather,
        seam: s.seam,
        mask_offset: exported?.offset ?? [0, 0],
      }
      const src = s.source && !s.source.isEmpty() ? await s.source.exportPng() : null
      if (src) opts.source_offset = src.offset
      set({ job: { ...EMPTY_JOB, id: 'pending' } })
      try {
        const { job_id } = await api.remove(doc.id, opts, exported?.blob ?? null, src?.blob ?? null)
        set({ job: { ...EMPTY_JOB, id: job_id } })
        const snap = await watchJob(job_id, {
          progress: (fraction, message) => set({ job: { ...get().job, fraction, message } }),
          preview: async (index, box) => {
            try {
              const r = await fetch(`/api/jobs/${job_id}/previews/${index}.webp`)
              if (!r.ok) return
              const bitmap = await createImageBitmap(await r.blob())
              if (get().job.id === job_id) set({ job: { ...get().job, preview: { bitmap, box } } })
            } catch {
              /* previews are best-effort */
            }
          },
        })
        if (snap.status === 'done' && snap.result) {
          const r = snap.result as {
            version: number
            method: string
            reason: string
            changed: boolean
          }
          set({ lastDecision: `${r.method}|${r.reason}` })
          if (r.changed) {
            get().pushUndo()
            set({
              version: r.version,
              path: [...get().path, { version: r.version, label: r.method, reason: r.reason }],
            })
            layer.clear()
            get().bumpMask()
          } else {
            get().notify(i18n.t('remove.noChange'))
          }
        } else if (snap.status === 'error') {
          get().notify(snap.error ?? 'error', 'error')
        }
      } catch (e) {
        get().notify((e as Error).message, 'error')
      } finally {
        set({ job: EMPTY_JOB })
      }
    },
  }
})
