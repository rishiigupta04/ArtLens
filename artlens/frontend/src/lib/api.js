const BASE = import.meta.env.VITE_API_URL?.replace(/\/$/, '')

const TRANSIENT_STATUS = new Set([429, 502, 503, 504])

export class ApiError extends Error {
  constructor(message, { status = null, isTransient = false, detail = null } = {}) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.isTransient = isTransient
    this.detail = detail
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function toMilliseconds(seconds, fallbackMs) {
  const value = Number(seconds)
  if (!Number.isFinite(value) || value <= 0) return fallbackMs
  return Math.round(value * 1000)
}

async function parseErrorBody(res) {
  const contentType = res.headers.get('content-type') || ''

  if (contentType.includes('application/json')) {
    const json = await res.json().catch(() => null)
    if (!json) return null
    if (typeof json === 'string') return { message: json, estimatedTimeSeconds: null }

    const rawDetail = json.detail
    const detailMessage = typeof rawDetail === 'string'
      ? rawDetail
      : rawDetail?.message || rawDetail?.error || null

    return {
      message: detailMessage || json.message || json.error || null,
      estimatedTimeSeconds: rawDetail?.estimated_time ?? json.estimated_time ?? null,
    }
  }

  const text = await res.text().catch(() => '')
  if (!text) return null

  // HF Spaces can return an HTML 503 page while containers wake up.
  if (text.trim().startsWith('<')) return null
  return { message: text.slice(0, 200), estimatedTimeSeconds: null }
}

function toUserMessage(status, detail) {
  if (TRANSIENT_STATUS.has(status)) {
    return 'The ArtLens model is waking up. This can take 20-60 seconds on first request.'
  }
  return detail?.message || `Request failed (${status})`
}

async function requestJson(path, { method = 'GET', body, retries = 0, retryDelayMs = 1400 } = {}) {
  if (!BASE) {
    throw new ApiError('Missing VITE_API_URL in frontend environment.', { status: null, isTransient: false })
  }

  for (let attempt = 0; attempt <= retries; attempt += 1) {
    try {
      const res = await fetch(`${BASE}${path}`, { method, body })

      if (res.ok) {
        return res.json()
      }

      const detail = await parseErrorBody(res)
      const status = res.status
      const isTransient = TRANSIENT_STATUS.has(status)
      const error = new ApiError(toUserMessage(status, detail), { status, isTransient, detail })

      if (isTransient && attempt < retries) {
        const retryAfter = res.headers.get('retry-after')
        const delayFromHeaderMs = toMilliseconds(retryAfter, 0)
        const delayFromBodyMs = toMilliseconds(detail?.estimatedTimeSeconds, 0)
        const delayMs = Math.max(delayFromHeaderMs, delayFromBodyMs, retryDelayMs * (attempt + 1))
        await sleep(delayMs)
        continue
      }

      throw error
    } catch (error) {
      const isNetworkError = error instanceof TypeError
      const canRetryNetwork = isNetworkError && attempt < retries

      if (canRetryNetwork) {
        await sleep(retryDelayMs * (attempt + 1))
        continue
      }

      if (isNetworkError) {
        throw new ApiError('Network error. Check your connection and try again.', {
          status: null,
          isTransient: true,
          detail: error.message,
        })
      }

      throw error
    }
  }

  throw new ApiError('Unexpected request failure.', { status: null, isTransient: false })
}

// Single image prediction
export async function predict(file) {
  const form = new FormData()
  form.append('file', file)
  return requestJson('/predict', { method: 'POST', body: form, retries: 4, retryDelayMs: 2500 })
}

// Explanation endpoint
export async function explain(file) {
  const form = new FormData()
  form.append('file', file)
  return requestJson('/explain/upload', { method: 'POST', body: form, retries: 2, retryDelayMs: 2200 })
}

// Batch prediction
export async function batchPredict(files) {
  const form = new FormData()
  files.forEach((f) => form.append('files', f))
  return requestJson('/batch', { method: 'POST', body: form, retries: 2, retryDelayMs: 2200 })
}

// Health check
export async function health() {
  return requestJson('/health')
}