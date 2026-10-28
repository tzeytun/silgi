import { useEffect } from 'react'
import { Viewport } from './canvas/Viewport'
import { BatchView } from './components/BatchView'
import { EmptyState } from './components/EmptyState'
import { ExportDialog } from './components/ExportDialog'
import { JobBar, Toast } from './components/JobBar'
import { RightPanel } from './components/RightPanel'
import { ToolRail } from './components/ToolRail'
import { TopBar } from './components/TopBar'
import { installHotkeys } from './hotkeys'
import { useStore } from './store/useStore'

export default function App() {
  const doc = useStore((s) => s.doc)
  const screen = useStore((s) => s.screen)
  const openFile = useStore((s) => s.openFile)

  useEffect(() => installHotkeys(), [])

  // dropping an image anywhere opens it
  useEffect(() => {
    const over = (e: DragEvent) => e.preventDefault()
    const drop = (e: DragEvent) => {
      e.preventDefault()
      const f = e.dataTransfer?.files[0]
      if (f && useStore.getState().screen === 'editor') void openFile(f)
    }
    window.addEventListener('dragover', over)
    window.addEventListener('drop', drop)
    return () => {
      window.removeEventListener('dragover', over)
      window.removeEventListener('drop', drop)
    }
  }, [openFile])

  return (
    <div className="flex h-full flex-col">
      <TopBar />
      <div className="relative flex min-h-0 flex-1">
        {screen === 'batch' ? (
          <main className="min-w-0 flex-1">
            <BatchView />
          </main>
        ) : doc ? (
          <>
            <ToolRail />
            <main className="relative min-w-0 flex-1">
              <Viewport />
              <JobBar />
            </main>
            <RightPanel />
          </>
        ) : (
          <main className="min-w-0 flex-1">
            <EmptyState />
          </main>
        )}
        <Toast />
      </div>
      <ExportDialog />
    </div>
  )
}
