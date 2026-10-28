import {
  Columns2,
  FolderOpen,
  Images,
  Languages,
  Maximize,
  Minus,
  Moon,
  PenLine,
  Plus,
  Redo2,
  ScanLine,
  Sun,
  Undo2,
  X,
} from 'lucide-react'
import { useRef } from 'react'
import { useTranslation } from 'react-i18next'
import { formatZoom, zoomAt } from '../lib/geometry'
import { setLang } from '../i18n'
import { useStore } from '../store/useStore'
import { Button, IconButton, Logo } from './ui'
import { useEffectiveTheme } from './useTheme'

export function TopBar() {
  const { t, i18n } = useTranslation()
  const s = useStore()
  const theme = useEffectiveTheme()
  const input = useRef<HTMLInputElement>(null)
  const hasDoc = s.doc !== null
  const center = { x: s.viewport.w / 2, y: s.viewport.h / 2 }

  return (
    <header className="flex h-12 shrink-0 items-center gap-2 border-b border-line bg-panel px-3">
      <div className="mr-2 flex items-center gap-3">
        <Logo height={22} />
      </div>
      {hasDoc && (
        <span
          className="max-w-[18ch] truncate text-muted"
          title={s.doc!.name}
          data-testid="filename"
        >
          {s.doc!.name}
        </span>
      )}

      <div className="mx-auto flex items-center gap-1">
        {hasDoc && s.screen === 'editor' && (
          <>
            <IconButton
              label={t('top.undo')}
              shortcut="Ctrl+Z"
              onClick={s.undo}
              disabled={s.undoStack.length === 0 || !!s.job.id}
              testId="undo"
            >
              <Undo2 size={17} />
            </IconButton>
            <IconButton
              label={t('top.redo')}
              shortcut="Ctrl+Shift+Z"
              onClick={s.redo}
              disabled={s.redoStack.length === 0 || !!s.job.id}
              testId="redo"
            >
              <Redo2 size={17} />
            </IconButton>
            <span className="mx-2 h-5 w-px bg-line" />
            <IconButton
              label={t('top.zoomOut')}
              shortcut="-"
              onClick={() => s.setView(zoomAt(s.view, 1 / 1.25, center))}
            >
              <Minus size={17} />
            </IconButton>
            <button
              type="button"
              onClick={() => s.fit()}
              aria-label={t('top.fit')}
              className="tip min-w-[52px] rounded-md px-2 py-1 text-center tabular-nums hover:bg-panel2"
              data-tip={t('top.fit')}
              data-tip-side="bottom"
              data-testid="zoom"
            >
              {formatZoom(s.view.zoom)}
            </button>
            <IconButton
              label={t('top.zoomIn')}
              shortcut="+"
              onClick={() => s.setView(zoomAt(s.view, 1.25, center))}
            >
              <Plus size={17} />
            </IconButton>
            <IconButton
              label={t('top.actual')}
              shortcut="Ctrl+1"
              onClick={() => s.setView(zoomAt(s.view, 1 / s.view.zoom, center))}
            >
              <ScanLine size={17} />
            </IconButton>
            <IconButton label={t('top.fit')} shortcut="Ctrl+0" onClick={() => s.fit()}>
              <Maximize size={17} />
            </IconButton>
            <span className="mx-2 h-5 w-px bg-line" />
            <IconButton
              label={t('top.compare')}
              shortcut="C"
              active={s.compare}
              onClick={() => s.setCompare(!s.compare)}
              testId="compare"
            >
              <Columns2 size={17} />
            </IconButton>
          </>
        )}
      </div>

      <div className="flex items-center gap-1">
        <input
          ref={input}
          type="file"
          accept="image/*,.heic,.heif,.tif,.tiff"
          hidden
          data-testid="file-input"
          onChange={(e) => {
            const f = e.target.files?.[0]
            if (f) void s.openFile(f)
            e.target.value = ''
          }}
        />
        <IconButton label={t('top.open')} shortcut="Ctrl+O" onClick={() => input.current?.click()}>
          <FolderOpen size={17} />
        </IconButton>
        <IconButton
          label={s.screen === 'batch' ? t('top.editor') : t('top.batch')}
          active={s.screen === 'batch'}
          onClick={() => s.set({ screen: s.screen === 'batch' ? 'editor' : 'batch' })}
          testId="batch-toggle"
        >
          {s.screen === 'batch' ? <PenLine size={17} /> : <Images size={17} />}
        </IconButton>
        {hasDoc && (
          <IconButton label={t('top.close')} onClick={s.closeDoc}>
            <X size={17} />
          </IconButton>
        )}
        <IconButton
          label={t('top.language')}
          onClick={() => setLang(i18n.language === 'tr' ? 'en' : 'tr')}
        >
          <Languages size={17} />
        </IconButton>
        <IconButton
          label={t('top.theme')}
          onClick={() => s.setTheme(theme === 'dark' ? 'light' : 'dark')}
          testId="theme"
        >
          {theme === 'dark' ? <Sun size={17} /> : <Moon size={17} />}
        </IconButton>
        <span className="mx-1" />
        <Button
          variant="accent"
          disabled={!hasDoc}
          onClick={() => s.set({ exportOpen: true })}
          testId="export"
        >
          {t('top.export')}
        </Button>
      </div>
    </header>
  )
}
