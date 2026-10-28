import { zoomAt } from './lib/geometry'
import { TOOLS } from './components/tools'
import { useStore } from './store/useStore'

const typing = (e: KeyboardEvent): boolean =>
  e.target instanceof HTMLElement &&
  (/^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName) || e.target.isContentEditable)

/** Global keyboard shortcuts. Returns an unsubscribe function. */
export function installHotkeys(): () => void {
  const onKey = (e: KeyboardEvent) => {
    if (typing(e) || document.querySelector('dialog[open]')) return
    const s = useStore.getState()
    const mod = e.ctrlKey || e.metaKey
    const k = e.key.toLowerCase()

    if (mod && k === 'z') {
      e.preventDefault()
      if (e.shiftKey) s.redo()
      else s.undo()
      return
    }
    if (mod && k === 'y') {
      e.preventDefault()
      s.redo()
      return
    }
    if (!s.doc) return
    if (mod && k === 'a') {
      e.preventDefault()
      s.pushUndo()
      s.mask?.selectAll()
      return
    }
    if (mod && k === 'i') {
      e.preventDefault()
      s.pushUndo()
      s.mask?.invert()
      return
    }
    if (mod && k === 'e') {
      e.preventDefault()
      s.set({ exportOpen: true })
      return
    }
    if (mod && k === '0') {
      e.preventDefault()
      s.fit()
      return
    }
    if (mod && k === '1') {
      e.preventDefault()
      s.setView(zoomAt(s.view, 1 / s.view.zoom, { x: s.viewport.w / 2, y: s.viewport.h / 2 }))
      return
    }
    if (mod || e.altKey) return

    const center = { x: s.viewport.w / 2, y: s.viewport.h / 2 }
    if (k === '+' || k === '=') return s.setView(zoomAt(s.view, 1.25, center))
    if (k === '-') return s.setView(zoomAt(s.view, 1 / 1.25, center))
    if (k === '[') return s.setOptions({ size: Math.max(1, Math.round(s.options.size * 0.85)) })
    if (k === ']')
      return s.setOptions({ size: Math.min(300, Math.round(s.options.size / 0.85) + 1) })
    if (k === 'r') {
      e.preventDefault()
      void s.runRemove()
      return
    }
    if (k === 'c') return s.setCompare(!s.compare)
    if (k === 'delete' || k === 'backspace') {
      s.pushUndo()
      s.mask?.clear()
      return
    }
    const tool = TOOLS.find((t) => t.key.toLowerCase() === k)
    if (tool) s.setTool(tool.id)
  }
  window.addEventListener('keydown', onKey)
  return () => window.removeEventListener('keydown', onKey)
}
