import { ChevronDown, ImageUp, Layers, Search } from 'lucide-react'
import { useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { api, watchJob, type Method } from '../lib/api'
import type { CombineMode } from '../lib/mask'
import { useStore } from '../store/useStore'
import { Button, Section, Segmented, Slider } from './ui'

const SIZE_TOOLS = ['brush', 'eraser', 'source', 'clone', 'heal']
const SHAPE_TOOLS = ['rect', 'ellipse', 'lasso', 'polygon', 'wand', 'color', 'text', 'similar']

function ToolOptions() {
  const { t } = useTranslation()
  const tool = useStore((s) => s.tool)
  const o = useStore((s) => s.options)
  const set = useStore((s) => s.setOptions)
  const doc = useStore((s) => s.doc)
  const version = useStore((s) => s.version)
  const candidates = useStore((s) => s.candidates)
  const kind = useStore((s) => s.candidateKind)
  const hint: Record<string, string> = {
    brush: 'options.hintBrush',
    eraser: 'options.hintBrush',
    polygon: 'options.hintPolygon',
    wand: 'options.hintWand',
    color: 'options.hintColor',
    text: 'options.hintText',
    similar: 'options.hintSimilar',
    source: 'options.hintSource',
    clone: 'options.hintClone',
    heal: 'options.hintClone',
  }
  const findText = async () => {
    if (!doc) return
    const s = useStore.getState()
    try {
      s.set({ candidates: await api.findText(doc.id, version), candidateKind: 'text' })
    } catch (e) {
      s.notify((e as Error).message, 'error')
    }
  }
  const accept = (indexes: number[]) => {
    const s = useStore.getState()
    if (!s.mask) return
    s.pushUndo()
    const mode: CombineMode = o.mode === 'replace' ? 'add' : o.mode
    indexes.forEach((i, n) => s.mask!.addBox(s.candidates[i], n === 0 ? mode : 'add'))
    s.set({ candidates: s.candidates.filter((_, i) => !indexes.includes(i)) })
  }

  return (
    <Section title={t('options.title')}>
      {hint[tool] && <p className="mb-3 mt-0 text-muted">{t(hint[tool])}</p>}
      {SIZE_TOOLS.includes(tool) && (
        <>
          <Slider
            label={t('options.size')}
            value={o.size}
            min={1}
            max={300}
            onChange={(size) => set({ size })}
            format={(v) => `${v} px`}
          />
          {tool !== 'clone' && tool !== 'heal' && (
            <Slider
              label={t('options.hardness')}
              value={o.hardness}
              min={0}
              max={1}
              step={0.05}
              onChange={(hardness) => set({ hardness })}
              format={(v) => `${Math.round(v * 100)}%`}
            />
          )}
        </>
      )}
      {tool === 'wand' && (
        <>
          <Slider
            label={t('options.tolerance')}
            value={o.tolerance}
            min={0}
            max={0.6}
            step={0.01}
            onChange={(tolerance) => set({ tolerance })}
            format={(v) => `${Math.round(v * 100)}%`}
          />
          <label className="mb-2 flex items-center gap-2">
            <input
              type="checkbox"
              checked={o.contiguous}
              onChange={(e) => set({ contiguous: e.target.checked })}
            />
            {t('options.contiguous')}
          </label>
        </>
      )}
      {tool === 'color' && (
        <Slider
          label={t('options.tolerance')}
          value={o.colorTolerance}
          min={1}
          max={80}
          onChange={(colorTolerance) => set({ colorTolerance })}
        />
      )}
      {tool === 'similar' && (
        <Slider
          label={t('options.threshold')}
          value={o.threshold}
          min={0.3}
          max={0.95}
          step={0.01}
          onChange={(threshold) => set({ threshold })}
          format={(v) => v.toFixed(2)}
        />
      )}
      {SHAPE_TOOLS.includes(tool) && (
        <div className="mb-2">
          <div className="mb-1">{t('options.mode')}</div>
          <Segmented
            label={t('options.mode')}
            value={o.mode}
            onChange={(mode) => set({ mode })}
            options={(['replace', 'add', 'subtract', 'intersect'] as const).map((m) => ({
              value: m,
              label: t(`options.${m}`),
            }))}
          />
        </div>
      )}
      {tool === 'text' && (
        <Button onClick={() => void findText()} testId="find-text" full>
          <Search size={14} className="mr-1 inline" /> {t('options.findNow')}
        </Button>
      )}
      {tool === 'source' && <SourceClear />}
      {(kind === 'text' || kind === 'similar') && (tool === 'text' || tool === 'similar') && (
        <div className="mt-3">
          <div className="mb-1 flex items-center justify-between">
            <span className="text-muted">{t('options.candidates')}</span>
            {candidates.length > 0 && (
              <button
                type="button"
                className="underline"
                onClick={() => accept(candidates.map((_, i) => i))}
              >
                {t('options.acceptAll')}
              </button>
            )}
          </div>
          {candidates.length === 0 ? (
            <p className="m-0 text-muted">{t('options.none')}</p>
          ) : (
            <ul className="m-0 max-h-40 list-none space-y-1 overflow-auto p-0 thin-scroll">
              {candidates.map((c, i) => (
                <li
                  key={`${c.x}-${c.y}-${i}`}
                  className="flex items-center justify-between rounded border border-line px-2 py-1"
                >
                  <span className="tabular-nums text-muted">
                    {c.w}×{c.h} · {Math.round(c.score * 100)}%
                  </span>
                  <button type="button" className="underline" onClick={() => accept([i])}>
                    {t('options.accept')}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </Section>
  )
}

function SourceClear() {
  const { t } = useTranslation()
  const source = useStore((s) => s.source)
  return (
    <Button onClick={() => source?.clear()} full>
      {t('options.clearSource')}
    </Button>
  )
}

function SelectionOps() {
  const { t } = useTranslation()
  const mask = useStore((s) => s.mask)
  const amount = useStore((s) => s.options.amount)
  const setOptions = useStore((s) => s.setOptions)
  const run = async (op: 'grow' | 'shrink' | 'smooth' | 'feather') => {
    const s = useStore.getState()
    if (!mask || mask.isEmpty()) return
    s.pushUndo()
    try {
      await mask.morph(op, amount)
    } catch (e) {
      s.notify((e as Error).message, 'error')
    }
  }
  const act = (fn: () => void) => () => {
    useStore.getState().pushUndo()
    fn()
  }
  return (
    <Section title={t('history.selection')}>
      <Slider
        label={t('options.amount')}
        value={amount}
        min={1}
        max={60}
        onChange={(v) => setOptions({ amount: v })}
      />
      <div className="mb-2 grid grid-cols-2 gap-1">
        {(['grow', 'shrink', 'smooth', 'feather'] as const).map((op) => (
          <Button key={op} onClick={() => void run(op)} testId={`op-${op}`}>
            {t(`options.${op}`)}
          </Button>
        ))}
      </div>
      <div className="grid grid-cols-3 gap-1">
        <Button onClick={act(() => mask?.clear())} testId="clear">
          {t('options.clear')}
        </Button>
        <Button onClick={act(() => mask?.invert())}>{t('options.invert')}</Button>
        <Button onClick={act(() => mask?.selectAll())}>{t('options.selectAll')}</Button>
      </div>
    </Section>
  )
}

function WatermarkSection() {
  const { t } = useTranslation()
  const s = useStore()
  const upload = useRef<HTMLInputElement>(null)
  const multi = useRef<HTMLInputElement>(null)
  const [estimating, setEstimating] = useState<number | null>(null)

  const onUpload = async (file: File) => {
    try {
      const w = await api.uploadWatermark(file)
      s.set({
        watermarks: [
          ...useStore.getState().watermarks,
          { id: w.id, name: file.name, width: w.width, height: w.height },
        ],
        watermarkId: w.id,
        method: 'alpha',
      })
    } catch (e) {
      s.notify((e as Error).message, 'error')
    }
  }
  const onEstimate = async (files: File[]) => {
    if (files.length < 3) {
      s.notify(t('watermark.estimating', { count: files.length }) + ' (≥ 3)', 'error')
      return
    }
    setEstimating(files.length)
    try {
      const { job_id } = await api.estimateWatermark(files)
      const snap = await watchJob(job_id, {})
      if (snap.status === 'done' && snap.result) {
        const id = snap.result.watermark_id as string
        s.set({
          watermarks: [
            ...useStore.getState().watermarks,
            { id, name: t('watermark.estimated'), estimated: true },
          ],
          watermarkId: id,
          method: 'alpha',
        })
      } else s.notify(snap.error ?? 'error', 'error')
    } catch (e) {
      s.notify((e as Error).message, 'error')
    } finally {
      setEstimating(null)
    }
  }
  const locate = async () => {
    if (!s.doc || !s.watermarkId) return
    try {
      const r = await api.locateWatermark(s.doc.id, s.version, s.watermarkId)
      s.set({
        watermarkNote: t('watermark.found', {
          left: Math.round(r.left),
          top: Math.round(r.top),
          score: r.score.toFixed(2),
        }),
      })
      s.mask?.addBox({ x: r.left, y: r.top, w: r.width, h: r.height }, 'replace')
    } catch (e) {
      s.notify((e as Error).message, 'error')
    }
  }

  return (
    <Section title={t('watermark.title')}>
      <p className="mb-2 mt-0 text-muted">{t('watermark.help')}</p>
      <input
        ref={upload}
        type="file"
        accept="image/png"
        hidden
        data-testid="wm-input"
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) void onUpload(f)
          e.target.value = ''
        }}
      />
      <input
        ref={multi}
        type="file"
        accept="image/*"
        multiple
        hidden
        onChange={(e) => {
          void onEstimate([...(e.target.files ?? [])])
          e.target.value = ''
        }}
      />
      <div className="mb-2 grid gap-1">
        <Button onClick={() => upload.current?.click()}>
          <ImageUp size={14} className="mr-1 inline" /> {t('watermark.upload')}
        </Button>
        <Button onClick={() => multi.current?.click()} disabled={estimating !== null}>
          <Layers size={14} className="mr-1 inline" />{' '}
          {estimating ? t('watermark.estimating', { count: estimating }) : t('watermark.estimate')}
        </Button>
      </div>
      {s.watermarks.length > 0 && (
        <div role="radiogroup" aria-label={t('watermark.title')} className="space-y-1">
          <label className="flex items-center gap-2">
            <input
              type="radio"
              name="wm"
              checked={s.watermarkId === null}
              onChange={() => s.set({ watermarkId: null, watermarkNote: null })}
            />
            {t('watermark.none')}
          </label>
          {s.watermarks.map((w) => (
            <label key={w.id} className="flex items-center gap-2">
              <input
                type="radio"
                name="wm"
                checked={s.watermarkId === w.id}
                onChange={() => s.set({ watermarkId: w.id, method: 'alpha', watermarkNote: null })}
              />
              <span className="checker inline-block h-7 w-10 shrink-0 overflow-hidden rounded border border-line">
                <img
                  src={`/api/watermarks/${w.id}.png`}
                  alt=""
                  className="h-full w-full object-contain"
                />
              </span>
              <span className="truncate">{w.name}</span>
            </label>
          ))}
        </div>
      )}
      {s.watermarkId && (
        <div className="mt-2">
          <Button onClick={() => void locate()} full>
            {t('watermark.locate')}
          </Button>
          {s.watermarkNote && <p className="mb-0 mt-2 text-muted">{s.watermarkNote}</p>}
        </div>
      )}
    </Section>
  )
}

const METHODS: Method[] = ['auto', 'alpha', 'patchmatch', 'telea', 'ns']

function RemoveSection() {
  const { t } = useTranslation()
  const s = useStore()
  const [open, setOpen] = useState(false)
  const running = s.job.id !== null
  const decision = s.lastDecision?.split('|')
  return (
    <Section title={t('remove.title')}>
      <div role="radiogroup" aria-label={t('remove.method')} className="mb-3 space-y-1">
        {METHODS.map((m) => (
          <label
            key={m}
            className={`block cursor-pointer rounded-md border px-2.5 py-1.5 ${s.method === m ? 'border-ink' : 'border-line hover:bg-panel2'}`}
          >
            <span className="flex items-center gap-2">
              <input
                type="radio"
                name="method"
                checked={s.method === m}
                onChange={() => s.set({ method: m })}
                data-testid={`method-${m}`}
              />
              <span className="font-medium">{t(`methods.${m}`)}</span>
            </span>
            <span className="mt-0.5 block pl-6 text-[12px] text-muted">{t(`methodHelp.${m}`)}</span>
          </label>
        ))}
      </div>
      <div className="mb-3">
        <div className="mb-1">{t('remove.quality')}</div>
        <Segmented
          label={t('remove.quality')}
          value={s.quality}
          onChange={(quality) => s.set({ quality })}
          options={(['fast', 'balanced', 'best'] as const).map((q) => ({
            value: q,
            label: t(`remove.${q}`),
          }))}
        />
      </div>
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
        className="mb-2 flex w-full items-center gap-1 text-muted"
      >
        <ChevronDown size={14} className={open ? '' : '-rotate-90'} /> {t('remove.advanced')}
      </button>
      {open && (
        <div className="mb-3">
          <Slider
            label={t('remove.feather')}
            value={s.feather}
            min={0}
            max={20}
            onChange={(feather) => s.set({ feather })}
          />
          <label className="mb-2 flex items-center gap-2">
            <input
              type="checkbox"
              checked={s.seam === 'poisson'}
              onChange={(e) => s.set({ seam: e.target.checked ? 'poisson' : 'none' })}
            />
            {t('remove.seam')}
          </label>
          <label className="block">
            {t('remove.seed')}
            <input
              type="number"
              value={s.seed}
              min={0}
              onChange={(e) => s.set({ seed: Math.max(0, Number(e.target.value) | 0) })}
              className="ml-2 w-20 rounded border border-line bg-panel px-1.5 py-0.5"
            />
          </label>
        </div>
      )}
      <Button
        variant="solid"
        onClick={() => void s.runRemove()}
        disabled={running}
        testId="remove"
        full
      >
        {running ? t('remove.working') : t('remove.run')} <kbd className="ml-1 opacity-60">R</kbd>
      </Button>
      {decision && (
        <p className="mb-0 mt-2 text-muted" data-testid="decision">
          {t('remove.chosen', {
            method: t(`methods.${decision[0]}`, decision[0]),
            reason: decision[1],
          })}
        </p>
      )}
    </Section>
  )
}

function History() {
  const { t } = useTranslation()
  const path = useStore((s) => s.path)
  const version = useStore((s) => s.version)
  const jumpTo = useStore((s) => s.jumpTo)
  return (
    <Section title={t('history.title')}>
      <ol className="m-0 list-none space-y-1 p-0" data-testid="history">
        {path.map((p, i) => (
          <li key={`${p.version}-${i}`}>
            <button
              type="button"
              onClick={() => jumpTo(i)}
              aria-current={p.version === version ? 'true' : undefined}
              className={`w-full rounded-md border px-2 py-1 text-left ${p.version === version ? 'border-ink' : 'border-line hover:bg-panel2'}`}
            >
              <span className="tabular-nums text-muted">{i}. </span>
              {p.label === 'original' ? t('history.original') : t(`methods.${p.label}`, p.label)}
            </button>
          </li>
        ))}
      </ol>
      {path.length === 1 && <p className="mb-0 mt-2 text-muted">{t('history.empty')}</p>}
    </Section>
  )
}

export function RightPanel() {
  const { t } = useTranslation()
  return (
    <aside
      aria-label={t('a11y.panel')}
      className="thin-scroll w-[300px] shrink-0 overflow-y-auto border-l border-line bg-panel"
    >
      <ToolOptions />
      <SelectionOps />
      <WatermarkSection />
      <RemoveSection />
      <History />
    </aside>
  )
}
