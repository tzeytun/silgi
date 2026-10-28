import type { ReactNode } from 'react'
import { useEffectiveTheme } from './useTheme'

export function Logo({ height = 22 }: { height?: number }) {
  const theme = useEffectiveTheme()
  return (
    <img
      src={theme === 'dark' ? '/logo-dark.svg' : '/logo-light.svg'}
      alt="silgi"
      style={{ height }}
      draggable={false}
    />
  )
}

interface IconButtonProps {
  label: string
  shortcut?: string
  onClick?: () => void
  active?: boolean
  disabled?: boolean
  children: ReactNode
  tip?: 'right' | 'bottom'
  testId?: string
}

/** Square icon button with an accessible name and a styled tooltip. */
export function IconButton({
  label,
  shortcut,
  onClick,
  active,
  disabled,
  children,
  tip = 'bottom',
  testId,
}: IconButtonProps) {
  return (
    <button
      type="button"
      aria-label={label}
      aria-pressed={active === undefined ? undefined : active}
      data-tip={shortcut ? `${label} (${shortcut})` : label}
      data-tip-side={tip}
      data-testid={testId}
      disabled={disabled}
      onClick={onClick}
      className={`tip grid h-8 w-8 place-items-center rounded-md border transition-colors ${
        active
          ? 'border-ink bg-ink text-bg'
          : 'border-transparent text-ink hover:border-line hover:bg-panel2'
      }`}
    >
      {children}
    </button>
  )
}

export function Section({
  title,
  children,
  right,
}: {
  title: string
  children: ReactNode
  right?: ReactNode
}) {
  return (
    <section className="border-b border-line px-4 py-3">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="m-0 text-[12px] uppercase tracking-wide text-muted">{title}</h3>
        {right}
      </div>
      {children}
    </section>
  )
}

export function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  format,
}: {
  label: string
  value: number
  min: number
  max: number
  step?: number
  onChange: (v: number) => void
  format?: (v: number) => string
}) {
  return (
    <label className="mb-2 block">
      <span className="mb-1 flex justify-between">
        <span>{label}</span>
        <span className="tabular-nums text-muted">{format ? format(value) : value}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  )
}

export function Segmented<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string
  value: T
  options: { value: T; label: string }[]
  onChange: (v: T) => void
}) {
  return (
    <div role="radiogroup" aria-label={label} className="flex rounded-md border border-line p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={value === o.value}
          onClick={() => onChange(o.value)}
          className={`flex-1 rounded px-2 py-1 text-[12px] ${
            value === o.value ? 'bg-ink text-bg' : 'text-ink hover:bg-panel2'
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

export function Button({
  children,
  onClick,
  variant = 'default',
  disabled,
  testId,
  type = 'button',
  full,
}: {
  children: ReactNode
  onClick?: () => void
  variant?: 'default' | 'solid' | 'accent' | 'ghost'
  disabled?: boolean
  testId?: string
  type?: 'button' | 'submit'
  full?: boolean
}) {
  const styles = {
    default: 'border-line bg-panel text-ink hover:bg-panel2',
    ghost: 'border-transparent text-ink hover:bg-panel2',
    solid: 'border-ink bg-ink text-bg hover:opacity-90',
    accent: 'border-accent bg-accent text-onaccent hover:bg-accent-hover',
  }[variant]
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      data-testid={testId}
      className={`rounded-md border px-3 py-1.5 text-[13px] font-medium transition-colors ${styles} ${full ? 'w-full' : ''}`}
    >
      {children}
    </button>
  )
}
