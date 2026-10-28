import { useEffect, useState } from 'react'
import { useStore } from '../store/useStore'

export function useEffectiveTheme(): 'light' | 'dark' {
  const theme = useStore((s) => s.theme)
  const [dark, setDark] = useState(() => window.matchMedia('(prefers-color-scheme: dark)').matches)
  useEffect(() => {
    const mq = window.matchMedia('(prefers-color-scheme: dark)')
    const on = () => setDark(mq.matches)
    mq.addEventListener('change', on)
    return () => mq.removeEventListener('change', on)
  }, [])
  return theme === 'system' ? (dark ? 'dark' : 'light') : theme
}
