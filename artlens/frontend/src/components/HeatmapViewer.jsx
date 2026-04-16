import { useEffect, useState } from 'react'
import { Loader } from 'lucide-react'
import { explain } from '../lib/api'
import InfoTip from './InfoTip'

const API = import.meta.env.VITE_API_URL?.replace(/\/$/, '')

function resolveHeatmapUrl(path) {
  if (!path) return null
  if (/^https?:\/\//i.test(path)) return path
  if (path.startsWith('/')) return `${API}${path}`
  return `${API}/${path}`
}

function normalizeHeatmaps(value) {
  if (!value || typeof value !== 'object') return null

  const gradcam = typeof value.gradcam === 'string' && value.gradcam.trim() ? value.gradcam : null
  const rollout = typeof value.attention_rollout === 'string' && value.attention_rollout.trim()
    ? value.attention_rollout
    : null

  if (!gradcam && !rollout) return null

  return {
    ...(gradcam ? { gradcam } : {}),
    ...(rollout ? { attention_rollout: rollout } : {}),
  }
}

function getModePath(heatmaps, mode) {
  if (mode === 'original') return null
  const key = mode === 'rollout' ? 'attention_rollout' : 'gradcam'
  return heatmaps?.[key] || null
}

export default function HeatmapViewer({ file, originalSrc, analysisId, initialHeatmaps, onHeatmapsReady }) {
  const [mode, setMode]       = useState('original')  // original | rollout | gradcam
  const [heatmaps, setHeatmaps] = useState(null)
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState(null)
  const [frameRatio, setFrameRatio] = useState(1)

  useEffect(() => {
    setHeatmaps(normalizeHeatmaps(initialHeatmaps))
    setMode('original')
    setError(null)
    setLoading(false)
  }, [analysisId])

  useEffect(() => {
    const normalized = normalizeHeatmaps(initialHeatmaps)
    if (normalized) {
      setHeatmaps((current) => ({ ...(current || {}), ...normalized }))
    }
  }, [initialHeatmaps])

  useEffect(() => {
    if (!originalSrc) {
      setFrameRatio(1)
      return
    }

    let cancelled = false
    const image = new Image()
    image.onload = () => {
      if (cancelled) return
      if (image.naturalWidth > 0 && image.naturalHeight > 0) {
        setFrameRatio(image.naturalWidth / image.naturalHeight)
      }
    }
    image.src = originalSrc

    return () => {
      cancelled = true
    }
  }, [originalSrc])

  const loadHeatmaps = async (targetMode) => {
    if (getModePath(heatmaps, targetMode)) return true
    if (!file) {
      setError('Heatmaps are not available for this saved item yet.')
      return false
    }
    setLoading(true); setError(null)
    try {
      const data = await explain(file)
      const urls = normalizeHeatmaps(data?.heatmap_urls)
      if (!urls) {
        setError('Heatmaps could not be generated for this image.')
        return false
      }
      setHeatmaps((current) => ({ ...(current || {}), ...urls }))
      if (analysisId && urls) {
        onHeatmapsReady?.(analysisId, urls)
      }
      return Boolean(getModePath(urls, targetMode))
    } catch (e) {
      setError(e.message)
      return false
    } finally {
      setLoading(false)
    }
  }

  const currentSrc = mode === 'original'
    ? originalSrc
    : getModePath(heatmaps, mode)
      ? resolveHeatmapUrl(getModePath(heatmaps, mode))
      : originalSrc

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-sm font-semibold text-slate-200">Explanation views</p>
        <InfoTip
          title="How to read this"
          text="All views are kept in the same frame as your original image. Switch tabs to see what regions influenced the decision and where the model focused."
        />
      </div>

      <div
        className="relative rounded-xl overflow-hidden bg-gray-900"
        style={{ aspectRatio: frameRatio }}
      >
        {currentSrc && (
          <img
            src={currentSrc}
            alt={mode}
            className="h-full w-full object-contain"
          />
        )}
        {loading && (
          <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
            <div className="flex flex-col items-center gap-2 text-white">
              <Loader size={24} className="animate-spin" />
              <span className="text-sm">Generating heatmaps...</span>
            </div>
          </div>
        )}
      </div>

      <div className="flex gap-2">
        {[
          { id: 'original', label: 'Original image' },
          { id: 'gradcam',  label: 'Visual evidence' },
          { id: 'rollout',  label: 'Model attention' },
        ].map(({ id, label }) => (
          <button
            key={id}
            onClick={async () => {
              if (id === 'original') {
                setMode('original')
                return
              }

              if (getModePath(heatmaps, id)) {
                setMode(id)
                return
              }

              const loaded = await loadHeatmaps(id)
              if (loaded) {
                setMode(id)
              }
            }}
            className={`flex-1 py-2 text-sm font-medium rounded-lg border transition-colors
              ${mode === id
                ? 'bg-slate-900 text-white border-slate-800'
                : 'bg-white/5 text-slate-300 border-white/15 hover:border-white/40'
              }`}
          >
            {label}
          </button>
        ))}
      </div>
      {error && <p className="text-sm text-red-300">{error}</p>}
      <p className="text-sm text-slate-300 leading-relaxed">
        {mode === 'gradcam'  && 'Visual evidence highlights the image areas that most pushed the final verdict. Warmer regions mean stronger influence.'}
        {mode === 'rollout'  && 'Model attention shows where the system looked across the whole image before deciding. Brighter areas received more focus.'}
        {mode === 'original' && 'Start with the original image, then open Visual evidence or Model attention to see what drove ArtLens to its decision.'}
      </p>
    </div>
  )
}