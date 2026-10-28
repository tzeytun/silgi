import { ImagePlus } from 'lucide-react'
import { useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useStore } from '../store/useStore'
import { Button, Logo } from './ui'

export function EmptyState() {
  const { t } = useTranslation()
  const openFile = useStore((s) => s.openFile)
  const openSample = useStore((s) => s.openSample)
  const busy = useStore((s) => s.busy)
  const input = useRef<HTMLInputElement>(null)
  const [over, setOver] = useState(false)

  return (
    <div className="grid h-full place-items-center p-8">
      <div className="flex w-full max-w-xl flex-col items-center gap-6">
        <Logo height={44} />
        <div
          role="button"
          tabIndex={0}
          data-testid="dropzone"
          aria-label={t('empty.title')}
          onClick={() => input.current?.click()}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault()
              input.current?.click()
            }
          }}
          onDragOver={(e) => {
            e.preventDefault()
            setOver(true)
          }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => {
            e.preventDefault()
            setOver(false)
            const f = e.dataTransfer.files[0]
            if (f) void openFile(f)
          }}
          className={`flex w-full cursor-pointer flex-col items-center gap-3 rounded-2xl border-2 border-dashed px-8 py-14 text-center transition-colors ${
            over ? 'border-accent bg-panel' : 'border-line bg-panel hover:border-muted'
          }`}
        >
          <ImagePlus size={34} strokeWidth={1.5} className="text-muted" />
          <h1 className="m-0 text-xl">{t('empty.title')}</h1>
          <p className="m-0 max-w-md text-muted">{t('empty.formats')}</p>
          <input
            ref={input}
            hidden
            type="file"
            accept="image/*,.heic,.heif,.tif,.tiff"
            data-testid="empty-file-input"
            onChange={(e) => {
              const f = e.target.files?.[0]
              if (f) void openFile(f)
              e.target.value = ''
            }}
          />
        </div>
        <Button onClick={() => void openSample()} disabled={busy} testId="sample">
          {t('empty.sample')}
        </Button>
        <p className="m-0 text-center text-[12px] text-muted">{t('responsible')}</p>
      </div>
    </div>
  )
}
