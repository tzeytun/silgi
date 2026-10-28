import type { Box } from './mask'

export interface ImageInfo {
  id: string
  name: string
  width: number
  height: number
  channels: number
  bit_depth: number
  has_alpha: boolean
  format: string
  has_icc: boolean
  has_exif: boolean
  tile_size: number
  levels: number
  notes: string[]
  watermarks?: { id: string; name: string }[]
}

export interface JobSnapshot {
  id: string
  status: 'queued' | 'running' | 'done' | 'error' | 'cancelled'
  fraction: number
  message: string
  result: Record<string, unknown> | null
  error: string | null
}

export type Method = 'auto' | 'alpha' | 'patchmatch' | 'telea' | 'ns' | 'clone' | 'heal'
export type Quality = 'fast' | 'balanced' | 'best'

export interface RemoveOptions {
  version: number
  method: Method
  quality: Quality
  watermark_id?: string | null
  watermark_position?: [number, number] | null
  offset?: [number, number] | null
  seed?: number
  feather?: number
  seam?: 'none' | 'poisson'
  mask_offset?: [number, number]
  source_offset?: [number, number]
}

export interface ExportRequest {
  version: number
  format: 'png' | 'jpeg' | 'webp' | 'tiff'
  quality: number
  keep_metadata: boolean
  keep_icc: boolean
  webp_lossless: boolean
}

export interface Candidate {
  x: number
  y: number
  w: number
  h: number
  score: number
  scale?: number
  is_source?: boolean
}

export interface MaskPng {
  blob: Blob
  offset: [number, number]
  color?: [number, number, number]
}

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function fail(r: Response): Promise<never> {
  let msg = r.statusText
  try {
    const j = await r.json()
    msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail)
  } catch {
    /* keep the status text */
  }
  throw new ApiError(r.status, msg)
}

async function json<T>(r: Response): Promise<T> {
  if (!r.ok) await fail(r)
  return (await r.json()) as T
}

const post = (url: string, body: unknown) =>
  fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })

export const tileUrl = (id: string, version: number, level: number, tx: number, ty: number) =>
  `/api/images/${id}/v/${version}/tiles/${level}/${tx}/${ty}.webp`

export const previewUrl = (id: string, version: number, size = 256) =>
  `/api/images/${id}/v/${version}/preview.webp?size=${size}`

async function maskResponse(r: Response): Promise<MaskPng> {
  if (!r.ok) await fail(r)
  const [x, y] = (r.headers.get('x-silgi-offset') ?? '0,0').split(',').map(Number)
  const c = r.headers.get('x-silgi-color')
  return { blob: await r.blob(), offset: [x, y], color: c ? JSON.parse(c) : undefined }
}

export const api = {
  async upload(file: File | Blob, name?: string): Promise<ImageInfo> {
    const fd = new FormData()
    fd.append('file', file, name ?? (file instanceof File ? file.name : 'image'))
    return json(await fetch('/api/images', { method: 'POST', body: fd }))
  },
  async sample(): Promise<ImageInfo> {
    return json(await fetch('/api/sample', { method: 'POST' }))
  },
  async close(id: string): Promise<void> {
    await fetch(`/api/images/${id}`, { method: 'DELETE' })
  },
  async methods(): Promise<{ id: string; description: string }[]> {
    const r = await json<{ methods: { id: string; description: string }[] }>(
      await fetch('/api/methods'),
    )
    return r.methods
  },
  async remove(
    id: string,
    opts: RemoveOptions,
    mask: Blob | null,
    source?: Blob | null,
  ): Promise<{ job_id: string }> {
    const fd = new FormData()
    fd.append('options', JSON.stringify(opts))
    if (mask) fd.append('mask', mask, 'mask.png')
    if (source) fd.append('source_mask', source, 'source.png')
    return json(await fetch(`/api/images/${id}/remove`, { method: 'POST', body: fd }))
  },
  async job(id: string): Promise<JobSnapshot> {
    return json(await fetch(`/api/jobs/${id}`))
  },
  async cancel(id: string): Promise<void> {
    await fetch(`/api/jobs/${id}`, { method: 'DELETE' })
  },
  async wand(
    id: string,
    version: number,
    x: number,
    y: number,
    tolerance: number,
    contiguous: boolean,
  ) {
    return maskResponse(
      await post(`/api/images/${id}/select/wand`, { version, x, y, tolerance, contiguous }),
    )
  },
  async colorRange(id: string, version: number, x: number, y: number, tolerance: number) {
    return maskResponse(
      await post(`/api/images/${id}/select/color-range`, { version, x, y, tolerance }),
    )
  },
  async findText(id: string, version: number): Promise<Candidate[]> {
    const r = await json<{ boxes: Candidate[] }>(
      await post(`/api/images/${id}/select/text`, { version }),
    )
    return r.boxes
  },
  async findSimilar(
    id: string,
    version: number,
    box: [number, number, number, number],
    threshold: number,
  ): Promise<Candidate[]> {
    const r = await json<{ hits: Candidate[] }>(
      await post(`/api/images/${id}/select/similar`, { version, box, threshold }),
    )
    return r.hits
  },
  async morph(op: 'grow' | 'shrink' | 'smooth' | 'feather', px: number, mask: Blob) {
    const fd = new FormData()
    fd.append('op', op)
    fd.append('px', String(px))
    fd.append('mask', mask, 'mask.png')
    return maskResponse(await fetch('/api/mask/morph', { method: 'POST', body: fd }))
  },
  async uploadWatermark(file: File): Promise<{ id: string; width: number; height: number }> {
    const fd = new FormData()
    fd.append('file', file, file.name)
    return json(await fetch('/api/watermarks', { method: 'POST', body: fd }))
  },
  async estimateWatermark(files: File[]): Promise<{ job_id: string }> {
    const fd = new FormData()
    for (const f of files) fd.append('files', f, f.name)
    return json(await fetch('/api/watermarks/estimate', { method: 'POST', body: fd }))
  },
  async locateWatermark(id: string, version: number, watermarkId: string) {
    return json<{ top: number; left: number; score: number; width: number; height: number }>(
      await post(`/api/images/${id}/watermark/locate`, { version, watermark_id: watermarkId }),
    )
  },
  async export(
    id: string,
    req: ExportRequest,
  ): Promise<{ blob: Blob; filename: string; warnings: string[] }> {
    const r = await post(`/api/images/${id}/export`, req)
    if (!r.ok) await fail(r)
    const cd = r.headers.get('content-disposition') ?? ''
    const filename = /filename="([^"]+)"/.exec(cd)?.[1] ?? 'silgi-export'
    return {
      blob: await r.blob(),
      filename,
      warnings: JSON.parse(r.headers.get('x-silgi-warnings') ?? '[]'),
    }
  },
  async exportZip(
    items: { image_id: string; version: number }[],
    options: ExportRequest,
  ): Promise<Blob> {
    const r = await post('/api/export/zip', { items, options })
    if (!r.ok) await fail(r)
    return r.blob()
  },
}

type JobEvent =
  | {
      type: 'status'
      status: JobSnapshot['status']
      fraction: number
      message: string
    }
  | { type: 'progress'; fraction: number; message: string }
  | { type: 'preview'; index: number; box: [number, number, number, number] }
  | { type: 'done' | 'error' | 'cancelled' }

/** Follow a job over WebSocket until it ends; resolves with its final snapshot. */
export function watchJob(
  jobId: string,
  on: {
    progress?: (fraction: number, message: string) => void
    preview?: (index: number, box: Box) => void
  },
): Promise<JobSnapshot> {
  return new Promise((resolve, reject) => {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    const ws = new WebSocket(`${proto}://${location.host}/api/jobs/${jobId}/events`)
    let finished = false
    const finish = () => {
      if (finished) return
      finished = true
      api.job(jobId).then(resolve, reject)
    }
    ws.onmessage = (m) => {
      const ev = JSON.parse(m.data) as JobEvent
      if (ev.type === 'progress') on.progress?.(ev.fraction, ev.message)
      else if (ev.type === 'preview') {
        const [y0, y1, x0, x1] = ev.box
        on.preview?.(ev.index, { x: x0, y: y0, w: x1 - x0, h: y1 - y0 })
      } else if (ev.type === 'status') {
        on.progress?.(ev.fraction, ev.message)
        if (ev.status === 'done' || ev.status === 'error' || ev.status === 'cancelled') finish()
      } else finish()
    }
    ws.onerror = finish
    ws.onclose = finish
  })
}
