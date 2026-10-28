import { useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { api, type ExportRequest } from '../lib/api'
import { useStore } from '../store/useStore'
import { download } from '../lib/download'
import { Button, Segmented, Slider } from './ui'

export function ExportDialog() {
  const { t } = useTranslation()
  const ref = useRef<HTMLDialogElement>(null)
  const open = useStore((s) => s.exportOpen)
  const doc = useStore((s) => s.doc)
  const version = useStore((s) => s.version)
  const [fmt, setFmt] = useState<ExportRequest['format']>('png')
  const [quality, setQuality] = useState(95)
  const [meta, setMeta] = useState(false)
  const [icc, setIcc] = useState(true)
  const [webpLossless, setWebpLossless] = useState(true)
  const [saving, setSaving] = useState(false)
  const close = () => useStore.getState().set({ exportOpen: false })

  useEffect(() => {
    const d = ref.current
    if (!d) return
    if (open && !d.open) d.showModal()
    if (!open && d.open) d.close()
  }, [open])

  const lossless = fmt === 'png' || fmt === 'tiff' || (fmt === 'webp' && webpLossless)
  const save = async () => {
    if (!doc) return
    setSaving(true)
    const s = useStore.getState()
    try {
      const r = await api.export(doc.id, {
        version,
        format: fmt,
        quality,
        keep_metadata: meta,
        keep_icc: icc,
        webp_lossless: webpLossless,
      })
      download(r.blob, r.filename)
      s.notify(
        t('export.saved', { name: r.filename }) +
          (r.warnings.length ? ` — ${r.warnings.join('; ')}` : ''),
      )
      close()
    } catch (e) {
      s.notify((e as Error).message, 'error')
    } finally {
      setSaving(false)
    }
  }

  return (
    <dialog
      ref={ref}
      aria-labelledby="export-title"
      onClose={close}
      onClick={(e) => e.target === ref.current && close()}
      className="m-auto w-[420px] max-w-[92vw] rounded-xl border border-line bg-panel p-0 text-ink backdrop:bg-black/40"
      data-testid="export-dialog"
    >
      <form
        method="dialog"
        className="p-5"
        onSubmit={(e) => {
          e.preventDefault()
          void save()
        }}
      >
        <h2 id="export-title" className="m-0 mb-4 text-lg">
          {t('export.title')}
        </h2>
        <div className="mb-3">
          <div className="mb-1">{t('export.format')}</div>
          <Segmented
            label={t('export.format')}
            value={fmt}
            onChange={setFmt}
            options={(['png', 'jpeg', 'webp', 'tiff'] as const).map((f) => ({
              value: f,
              label: f.toUpperCase(),
            }))}
          />
        </div>
        {(fmt === 'jpeg' || (fmt === 'webp' && !webpLossless)) && (
          <Slider
            label={t('export.quality')}
            value={quality}
            min={40}
            max={100}
            onChange={setQuality}
          />
        )}
        {fmt === 'webp' && (
          <label className="mb-2 flex items-center gap-2">
            <input
              type="checkbox"
              checked={webpLossless}
              onChange={(e) => setWebpLossless(e.target.checked)}
            />
            {t('export.webpLossless')}
          </label>
        )}
        <label className="mb-1 flex items-center gap-2">
          <input
            type="checkbox"
            checked={meta}
            onChange={(e) => setMeta(e.target.checked)}
            data-testid="keep-meta"
          />
          {t('export.metadata')}
        </label>
        <p className="m-0 mb-2 pl-6 text-[12px] text-muted">{t('export.metadataHelp')}</p>
        <label className="mb-3 flex items-center gap-2">
          <input type="checkbox" checked={icc} onChange={(e) => setIcc(e.target.checked)} />
          {t('export.icc')}
        </label>
        <p className="mb-4 mt-0 rounded-md bg-panel2 px-3 py-2 text-[12px] text-muted">
          {lossless ? t('export.lossless') : t('export.lossy')}
        </p>
        <div className="flex justify-end gap-2">
          <Button onClick={close}>{t('export.cancel')}</Button>
          <Button type="submit" variant="accent" disabled={saving} testId="save-export">
            {saving ? t('export.saving') : t('export.save')}
          </Button>
        </div>
      </form>
    </dialog>
  )
}
