import { useState, useCallback } from 'react'
import { predict } from '../lib/api'
import { logPrediction } from '../lib/supabase'
import UploadZone from '../components/UploadZone'
import ResultCard from '../components/ResultCard'
import { clearHistory, loadHistory, pushHistory, saveHistory } from '../lib/predictionHistory'

async function createThumbnail(file, maxSide = 320) {
  const dataUrl = await new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result)
    reader.onerror = reject
    reader.readAsDataURL(file)
  })

  return new Promise((resolve) => {
    const image = new Image()
    image.onload = () => {
      const longestSide = Math.max(image.naturalWidth, image.naturalHeight)
      const scale = longestSide > maxSide ? maxSide / longestSide : 1
      const width = Math.max(1, Math.round(image.naturalWidth * scale))
      const height = Math.max(1, Math.round(image.naturalHeight * scale))

      const canvas = document.createElement('canvas')
      canvas.width = width
      canvas.height = height
      const ctx = canvas.getContext('2d')
      if (!ctx) {
        resolve(null)
        return
      }

      ctx.drawImage(image, 0, 0, width, height)
      resolve(canvas.toDataURL('image/jpeg', 0.82))
    }
    image.onerror = () => resolve(null)
    image.src = String(dataUrl)
  })
}

function formatHistoryTime(value) {
  try {
    return new Date(value).toLocaleString()
  } catch {
    return ''
  }
}

export default function Home() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [history, setHistory] = useState(() => loadHistory())

  const handleFile = useCallback(async (f) => {
    setResult(null)
    setError(null)
    if (!f) {
      setFile(null)
      setPreview(null)
      return
    }

    setFile(f)
    setPreview(URL.createObjectURL(f))
    setLoading(true)
    try {
      const thumbnailPromise = createThumbnail(f)
      const res = await predict(f)
      setResult(res)
      const thumbnail = await thumbnailPromise

      const entry = {
        id: res.request_id || `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        createdAt: new Date().toISOString(),
        filename: f.name,
        thumbnail,
        result: res,
      }

      setHistory((current) => {
        const next = pushHistory(current, entry)
        saveHistory(next)
        return next
      })

      // Non-blocking analytics write.
      logPrediction(res, res.request_id).catch(() => {})
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  const showFullWidthUnknown = Boolean(result?.generator?.is_unknown)

  const reopenHistory = (entry) => {
    setError(null)
    setFile(null)
    setPreview(entry.thumbnail || null)
    setResult(entry.result)
  }

  const clearLocalHistory = () => {
    clearHistory()
    setHistory([])
  }

  const handleHeatmapsReady = useCallback((analysisId, heatmapUrls) => {
    if (!analysisId || !heatmapUrls) return

    setResult((current) => {
      if (!current || current.request_id !== analysisId) return current
      return { ...current, heatmap_urls: heatmapUrls }
    })

    setHistory((current) => {
      const next = current.map((item) => {
        if (item.result?.request_id !== analysisId) return item
        return {
          ...item,
          result: {
            ...item.result,
            heatmap_urls: heatmapUrls,
          },
        }
      })
      saveHistory(next)
      return next
    })
  }, [])

  return (
    <main className="max-w-6xl mx-auto px-4 py-6 sm:py-10 sm:mt-10 space-y-6 sm:space-y-8">
      <section className="soft-card p-5 sm:p-8 md:p-12 overflow-hidden relative">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_25%_20%,rgba(34,211,238,0.2),transparent_40%),radial-gradient(circle_at_85%_10%,rgba(168,85,247,0.16),transparent_35%)]" />
        <div className="relative z-10 grid md:grid-cols-2 gap-8 items-center">
          <div className="space-y-5">
            <span className="inline-flex text-xs border border-cyan-300/20 text-cyan-200 bg-cyan-500/10 px-3 py-1 rounded-full">
              AI Trust Infrastructure for Creators and Platforms
            </span>
            <h1 className="font-display text-3xl sm:text-4xl md:text-5xl font-light text-white leading-tight">
              Verify visual authenticity in seconds with <em className="text-cyan-300 not-italic">ArtLens</em>
            </h1>
            <p className="text-slate-300 text-sm md:text-base leading-relaxed max-w-xl">
              Upload one image and ArtLens checks whether it is likely human-made or AI-generated,
              identifies probable model family, and highlights regions that influenced the decision.
            </p>
          </div>

          <div className="soft-card p-3 sm:p-4 md:p-5 interactive-lift">
            <UploadZone onFile={handleFile} loading={loading} />
            {error && (
              <div className="mt-4 bg-red-500/10 border border-red-300/20 rounded-xl p-3">
                <p className="text-sm text-red-200">{error}</p>
              </div>
            )}
          </div>
        </div>
      </section>

      {showFullWidthUnknown && result ? (
        <section className="w-full">
          <ResultCard
            result={result}
            file={file}
            originalSrc={preview}
            onHeatmapsReady={handleHeatmapsReady}
          />
        </section>
      ) : (
        <section className="w-full">
          {result ? (
            <ResultCard
              result={result}
              file={file}
              originalSrc={preview}
              onHeatmapsReady={handleHeatmapsReady}
            />
          ) : (
            <div className="soft-card w-full min-h-56 sm:min-h-72 flex items-center justify-center text-slate-400 text-center px-4">
              Results and explainability layers appear here after upload.
            </div>
          )}
        </section>
      )}

      <section className="soft-card p-4 md:p-5 space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
          <h2 className="text-sm font-semibold text-slate-200">Recent analyses</h2>
          {history.length > 0 && (
            <button
              onClick={clearLocalHistory}
              className="text-xs text-slate-400 hover:text-slate-200"
            >
              Clear history
            </button>
          )}
        </div>

        {history.length === 0 ? (
          <p className="text-xs text-slate-400">Your last 5 results will appear here for quick revisit.</p>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
            {history.map((item) => {
              const verdict = item.result?.label === 'ai' ? 'AI-generated' : 'Human-made'
              const confidence = item.result?.confidence != null
                ? `${Math.round(item.result.confidence * 100)}%`
                : '-'
              const active = result?.request_id && item.result?.request_id === result.request_id

              return (
                <button
                  key={item.id}
                  onClick={() => reopenHistory(item)}
                  className={`text-left rounded-xl border p-3 transition-colors ${
                    active
                      ? 'border-cyan-300/40 bg-cyan-500/10'
                      : 'border-white/10 bg-white/5 hover:border-white/30'
                  }`}
                >
                  {item.thumbnail ? (
                    <img
                      src={item.thumbnail}
                      alt={item.filename || 'History preview'}
                      className="w-full h-24 object-cover rounded-lg mb-2"
                    />
                  ) : (
                    <div className="w-full h-24 rounded-lg mb-2 bg-slate-900/50 border border-white/10" />
                  )}
                  <p className="text-xs text-slate-300 truncate">{item.filename || 'Uploaded image'}</p>
                  <p className="text-xs text-slate-400 mt-1">{verdict} · {confidence}</p>
                  <p className="text-[11px] text-slate-500 mt-1">{formatHistoryTime(item.createdAt)}</p>
                </button>
              )
            })}
          </div>
        )}
      </section>

      <section className="soft-card p-4 sm:p-6 md:p-8 space-y-6">
        <div>
          <h2 className="text-2xl font-display text-slate-100">What is ArtLens?</h2>
          <p className="text-sm text-slate-300 mt-2 max-w-4xl leading-relaxed">
            Think of ArtLens as a three-step inspection pipeline. It first checks visual style patterns,
            then checks frequency fingerprints that are hard to fake, and finally explains what parts
            of the image influenced the verdict. This makes the system useful for regular users,
            creators, and moderation teams who want transparent decisions.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
          <div className="space-y-4">
            {[
              {
                title: '1) Visual understanding (what humans notice)',
                desc: 'A transformer model scans composition, brush texture, edges, and style consistency.'
              },
              {
                title: '2) Frequency analysis (what humans do not notice)',
                desc: 'FFT and wavelet signals detect tiny generation artifacts caused by upsampling and denoising.'
              },
              {
                title: '3) Explainable output (why the model decided)',
                desc: 'GradCAM++ and attention maps highlight image regions that drove the final classification.'
              },
            ].map((item) => (
              <article key={item.title} className="rounded-xl border border-white/10 bg-white/5 p-4 interactive-lift">
                <h3 className="text-sm font-semibold text-slate-100">{item.title}</h3>
                <p className="text-xs text-slate-300 mt-1 leading-relaxed">{item.desc}</p>
              </article>
            ))}
          </div>

          <div className="rounded-2xl border border-white/10 bg-slate-950/30 p-5">
            <p className="text-xs uppercase tracking-wide text-slate-400 mb-4">How a single upload flows</p>
            <div className="space-y-3">
              {[
                ['Upload image', 'User drops JPEG/PNG/WebP'],
                ['Deep + frequency checks', 'ViT + handcrafted signals are combined'],
                ['Verdict', 'AI-generated or human-made'],
                ['Generator attribution', 'Midjourney / Stable Diffusion / Unknown'],
                ['Explainability', 'Heatmaps show influencing regions'],
              ].map(([step, detail], idx) => (
                <div key={step} className="flex items-start gap-3">
                  <div className="mt-0.5 h-6 w-6 rounded-full bg-cyan-400/20 border border-cyan-300/30 text-cyan-200 text-xs flex items-center justify-center">
                    {idx + 1}
                  </div>
                  <div>
                    <p className="text-sm font-medium text-slate-100">{step}</p>
                    <p className="text-xs text-slate-400">{detail}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>
    </main>
  )
}