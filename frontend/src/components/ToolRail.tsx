import { useTranslation } from 'react-i18next'
import { useStore } from '../store/useStore'
import { TOOLS } from './tools'
import { IconButton } from './ui'

export function ToolRail() {
  const { t } = useTranslation()
  const tool = useStore((s) => s.tool)
  const setTool = useStore((s) => s.setTool)
  const group = (g: 'selection' | 'manual') =>
    TOOLS.filter((x) => x.group === g).map((x) => (
      <IconButton
        key={x.id}
        label={t(`tools.${x.id}`)}
        shortcut={x.key}
        active={tool === x.id}
        onClick={() => setTool(x.id)}
        tip="right"
        testId={`tool-${x.id}`}
      >
        <x.icon size={18} />
      </IconButton>
    ))
  return (
    <nav
      aria-label={t('a11y.toolRail')}
      className="flex w-12 shrink-0 flex-col items-center gap-1 border-r border-line bg-panel py-2"
    >
      {group('selection')}
      <span className="my-1 h-px w-6 bg-line" />
      {group('manual')}
    </nav>
  )
}
