import { Download, Play, Plus, Trash2 } from 'lucide-react'
import { useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { api, previewUrl, type ExportRequest, type Method } from '../lib/api'
import { useStore, type BatchItem } from '../store/useStore'
import { download } from '../lib/download'
import { Button, IconButton, Segmented } from './ui'

const OVERRIDES: (Method | 'default')[] = ['default', 'auto', 'patchmatch', 'telea', 'ns']

export function BatchView() {
  const { t } = useTranslation()
  const s = useStore()
  const input = useRef<HTMLInputElement>(null)
  const [mode, setMode] = useState<'selection' | 'watermark'>('watermark')
  const [running, setRunning] = useState(false)

  const update = (id: string, patch: Partial<BatchItem>) =>
    useStore.setState((st) => ({
      batch: st.batch.map((b) => (b.id === id ? { ...b, ...patch } : b)),
    }))

  const add = async (files: File[]) => {
    const items: BatchItem[] = files.map((f) => ({
      id: crypto.randomUUID(),
      file: f,
      name: f.name,
      status: 'waiting',
      method: 'default',
      version: 0,
    }))
    useStore.setState((st) => ({ batch: [...st.batch, ...items] }))
    for (const it of items) {
      try {
        update(it.id, { doc: await api.upload(it.file) })
      } catch (e) {
        update(it.id, { status: 'error', message: (e as Error).message })
      }
    }
  }

  const run = async () => {
    const st = useStore.getState()
    if (mode === 'watermark' && !st.watermarkId) return st.notify(t('batch.needWatermark'), 'error')
    const sel = mode === 'selection' ? await st.mask?.exportPng() : null
    if (mode === 'selection' && !sel) return st.notify(t('batch.needSelection'), 'error')
    setRunning(true)
    for (const it of useStore.getState().batch) {
      if (it.status === 'done' || !it.doc) continue
      update(it.id, { status: 'working', message: undefined })
      try {
        const sameSize = st.doc && it.doc.width === st.doc.width && it.doc.height === st.doc.height
        if (mode === 'selection' && !sameSize) {
          update(it.id, { status: 'skipped', message: t('batch.needSelection') })
          continue
        }
        const method: Method =
          it.method === 'default' ? (mode === 'watermark' ? 'alpha' : st.method) : it.method
        const { job_id } = await api.remove(
          it.doc.id,
          {
            version: 0,
            method: mode === 'watermark' && it.method === 'default' ? 'alpha' : method,
            quality: st.quality,
            watermark_id: mode === 'watermark' ? st.watermarkId : null,
            mask_offset: sel?.offset ?? [0, 0],
          },
          sel?.blob ?? null,
        )
        let snap = await api.job(job_id)
        while (snap.status === 'queued' || snap.status === 'running') {
          await new Promise((r) => setTimeout(r, 200))
          snap = await api.job(job_id)
        }
        if (snap.status === 'done' && snap.result) {
          update(it.id, { status: 'done', version: snap.result.version as number })
        } else update(it.id, { status: 'error', message: snap.error ?? snap.status })
      } catch (e) {
        update(it.id, { status: 'error', message: (e as Error).message })
      }
    }
    setRunning(false)
  }

  const zip = async () => {
    const done = s.batch.filter((b) => b.status === 'done' && b.doc)
    if (!done.length) return
    const opts: ExportRequest = {
      version: 0,
      format: 'png',
      quality: 95,
      keep_metadata: false,
      keep_icc: true,
      webp_lossless: true,
    }
    try {
      download(
        await api.exportZip(
          done.map((b) => ({ image_id: b.doc!.id, version: b.version })),
          opts,
        ),
        'silgi-batch.zip',
      )
    } catch (e) {
      s.notify((e as Error).message, 'error')
    }
  }

  return (
    <div className="flex h-full flex-col" data-testid="batch-view">
      <div className="flex flex-wrap items-center gap-3 border-b border-line bg-panel px-4 py-3">
        <h2 className="m-0 mr-2 text-lg">{t('batch.title')}</h2>
        <input
          ref={input}
          type="file"
          accept="image/*"
          multiple
          hidden
          onChange={(e) => {
            void add([...(e.target.files ?? [])])
            e.target.value = ''
          }}
        />
        <Button onClick={() => input.current?.click()}>
          <Plus size={14} className="mr-1 inline" /> {t('batch.add')}
        </Button>
        <span className="text-muted">{t('batch.mode')}</span>
        <div className="w-[360px]">
          <Segmented
            label={t('batch.mode')}
            value={mode}
            onChange={setMode}
            options={[
              { value: 'watermark', label: t('batch.modeWatermark') },
              { value: 'selection', label: t('batch.modeSelection') },
            ]}
          />
        </div>
        <div className="ml-auto flex gap-2">
          <Button
            variant="solid"
            onClick={() => void run()}
            disabled={running || s.batch.length === 0}
          >
            <Play size={14} className="mr-1 inline" /> {t('batch.run')}
          </Button>
          <Button onClick={() => void zip()} disabled={!s.batch.some((b) => b.status === 'done')}>
            <Download size={14} className="mr-1 inline" /> {t('batch.export')}
          </Button>
        </div>
      </div>
      <div className="thin-scroll flex-1 overflow-auto p-4">
        {s.batch.length === 0 ? (
          <p className="mt-16 text-center text-muted">{t('batch.empty')}</p>
        ) : (
          <ul className="m-0 grid list-none grid-cols-[repeat(auto-fill,minmax(240px,1fr))] gap-3 p-0">
            {s.batch.map((b) => (
              <li
                key={b.id}
                className="overflow-hidden rounded-lg border border-line bg-panel"
                data-testid="batch-item"
              >
                <div className="checker grid h-36 place-items-center">
                  {b.doc && (
                    <img
                      src={previewUrl(b.doc.id, b.status === 'done' ? b.version : 0, 320)}
                      alt={b.name}
                      className="max-h-full max-w-full object-contain"
                    />
                  )}
                </div>
                <div className="space-y-2 p-3">
                  <div className="flex items-center justify-between gap-2">
                    <span className="truncate font-medium" title={b.name}>
                      {b.name}
                    </span>
                    <IconButton
                      label={t('batch.remove')}
                      onClick={() =>
                        useStore.setState((st) => ({
                          batch: st.batch.filter((x) => x.id !== b.id),
                        }))
                      }
                    >
                      <Trash2 size={15} />
                    </IconButton>
                  </div>
                  <div
                    className={
                      b.status === 'error' ? 'text-ink underline decoration-accent' : 'text-muted'
                    }
                    title={b.message}
                  >
                    {t(`batch.status.${b.status}`)}
                    {b.message ? ` — ${b.message}` : ''}
                  </div>
                  <label className="block text-[12px] text-muted">
                    {t('batch.override')}
                    <select
                      value={b.method}
                      onChange={(e) =>
                        update(b.id, { method: e.target.value as Method | 'default' })
                      }
                      className="mt-1 block w-full rounded border border-line bg-panel px-2 py-1 text-ink"
                    >
                      {OVERRIDES.map((m) => (
                        <option key={m} value={m}>
                          {m === 'default' ? t('batch.default') : t(`methods.${m}`)}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}
