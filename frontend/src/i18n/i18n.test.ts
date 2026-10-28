import { describe, expect, it } from 'vitest'
import en from './en'
import tr from './tr'

const keys = (o: object, prefix = ''): string[] =>
  Object.entries(o).flatMap(([k, v]) =>
    typeof v === 'object' && v !== null ? keys(v, `${prefix}${k}.`) : [`${prefix}${k}`],
  )

const placeholders = (s: string) => [...s.matchAll(/{{\s*(\w+)\s*}}/g)].map((m) => m[1]).sort()

const flat = (o: object, prefix = ''): Record<string, string> =>
  Object.fromEntries(
    Object.entries(o).flatMap(([k, v]) =>
      typeof v === 'object' && v !== null
        ? Object.entries(flat(v, `${prefix}${k}.`))
        : [[`${prefix}${k}`, String(v)]],
    ),
  )

describe('translations', () => {
  it('Turkish has exactly the English keys', () => {
    expect(keys(tr).sort()).toEqual(keys(en).sort())
  })

  it('uses the same placeholders in both languages', () => {
    const a = flat(en)
    const b = flat(tr)
    for (const k of Object.keys(a)) expect(placeholders(b[k]), k).toEqual(placeholders(a[k]))
  })

  it('has no empty strings', () => {
    for (const [k, v] of Object.entries({ ...flat(en), ...flat(tr) }))
      expect(v.trim(), k).not.toBe('')
  })
})
