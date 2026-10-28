import { X } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { useStore } from '../store/useStore'
import { Button } from './ui'

/** Non-blocking progress for the running job, with cancel. */
export function JobBar() {
  const { t } = useTranslation()
  const job = useStore((s) => s.job)
  const cancel = useStore((s) => s.cancelJob)
  if (!job.id) return null
  const pct = Math.round(job.fraction * 100)
  return (
    <div
      role="status"
      aria-live="polite"
      data-testid="job-bar"
      className="absolute bottom-4 left-1/2 z-10 flex w-[360px] -translate-x-1/2 items-center gap-3 rounded-xl border border-line bg-panel px-4 py-3 shadow-lg"
    >
      <div className="min-w-0 flex-1">
        <div className="mb-1.5 flex justify-between">
          <span>
            {job.fraction === 0 && job.message === '' ? t('job.queued') : t('job.running')}
          </span>
          <span className="tabular-nums text-muted">{pct}%</span>
        </div>
        <div
          className="h-1.5 overflow-hidden rounded-full bg-panel2"
          role="progressbar"
          aria-valuenow={pct}
          aria-valuemin={0}
          aria-valuemax={100}
        >
          <div
            className="h-full bg-ink transition-[width] duration-150"
            style={{ width: `${pct}%` }}
          />
        </div>
        {job.preview && <div className="mt-1.5 text-[12px] text-muted">{t('remove.preview')}</div>}
      </div>
      <Button onClick={() => void cancel()} testId="cancel-job">
        <X size={14} className="mr-1 inline" />
        {t('job.cancel')}
      </Button>
    </div>
  )
}

export function Toast() {
  const toast = useStore((s) => s.toast)
  if (!toast) return null
  return (
    <div
      role={toast.kind === 'error' ? 'alert' : 'status'}
      data-testid="toast"
      className={`absolute right-4 top-4 z-20 max-w-sm rounded-lg border bg-panel px-4 py-2.5 shadow-lg ${
        toast.kind === 'error' ? 'border-accent' : 'border-line'
      }`}
    >
      {toast.text}
    </div>
  )
}
