import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { batchPredict } from '../lib/api'
import { Upload } from 'lucide-react'

const LABEL_COLORS = {
  ai: 'text-rose-200 bg-rose-500/20 border border-rose-300/20',
  human: 'text-emerald-200 bg-emerald-500/20 border border-emerald-300/20',
}

export default function Batch() {
  const [files, setFiles] = useState([])
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const onDrop = useCallback((accepted) => {
    setFiles((prev) => [...prev, ...accepted].slice(0, 20))
    setResults([])
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/jpeg': [], 'image/png': [], 'image/webp': [] },
    maxFiles: 20,
    disabled: loading,
  })

  const run = async () => {
    if (!files.length) return
    setLoading(true)
    setError(null)
    setResults([])
    try {
      const data = await batchPredict(files)
      setResults(data.predictions || [])
    } catch (e) {
      setError(e.message || 'Batch request failed')
    } finally {
      setLoading(false)
    }
  }

  const clear = () => {
    setFiles([])
    setResults([])
    setError(null)
  }

  const aiCount = results.filter((r) => r.label === 'ai').length
  const humanCount = results.filter((r) => r.label === 'human').length

  return (
    <main className="max-w-6xl mx-auto px-4 py-6 sm:py-10 space-y-6">
      <div>
        <h1 className="text-3xl font-display text-slate-100">Batch analysis at scale</h1>
        <p className="text-sm text-slate-400 mt-1">
          Analyse up to 20 images in one request with full generator metadata.
        </p>
      </div>

      <div
        {...getRootProps()}
        className={`soft-card border-2 border-dashed p-6 sm:p-10 text-center cursor-pointer transition-colors ${
          isDragActive ? 'border-cyan-300/60 bg-cyan-400/10' : 'border-white/15'
        }`}
      >
        <input {...getInputProps()} />
        <Upload size={24} className="mx-auto text-slate-400 mb-2" />
        <p className="text-sm text-slate-200">Drop up to 20 images or click to browse</p>
        <p className="text-xs text-slate-400 mt-1">{files.length}/20 selected</p>
      </div>

      {files.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {files.map((f, i) => (
            <div
              key={i}
              className="text-xs bg-white/5 border border-white/10 px-2 py-1 rounded-md text-slate-300 truncate max-w-40"
            >
              {f.name}
            </div>
          ))}
        </div>
      )}

      <div className="flex flex-wrap gap-3">
        <button
          onClick={run}
          disabled={!files.length || loading}
          className="px-5 py-2 bg-cyan-400/20 text-cyan-100 border border-cyan-300/30 rounded-lg text-sm font-medium hover:bg-cyan-400/30 disabled:opacity-40 transition-colors"
        >
          {loading ? 'Analysing...' : `Analyse ${files.length} image${files.length !== 1 ? 's' : ''}`}
        </button>
        {files.length > 0 && (
          <button
            onClick={clear}
            className="px-5 py-2 border border-white/15 rounded-lg text-sm text-slate-300 hover:bg-white/10"
          >
            Clear
          </button>
        )}
      </div>

      {error && <p className="text-sm text-red-300">{error}</p>}

      {results.length > 0 && (
        <div className="space-y-3">
          <div className="flex flex-wrap gap-4 text-sm">
            <span className="text-emerald-300 font-medium">{humanCount} human</span>
            <span className="text-rose-300 font-medium">{aiCount} AI</span>
            <span className="text-slate-400">{results.length} total</span>
          </div>

          <div className="soft-card overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full min-w-[720px] text-sm">
              <thead className="bg-white/5 border-b border-white/10">
                <tr>
                  {[
                    'Filename',
                    'Verdict',
                    'Confidence',
                    'Generator',
                    'ViT (visual cues)',
                    'LGB (frequency cues)',
                  ].map((h) => (
                    <th key={h} className="text-left px-4 py-2.5 text-xs font-semibold text-slate-400">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-white/10">
                {results.map((r, i) => (
                  <tr key={i} className="hover:bg-white/5">
                    <td className="px-4 py-2.5 text-slate-300 max-w-36 truncate text-xs">{r.filename}</td>
                    <td className="px-4 py-2.5">
                      {r.label ? (
                        <span
                          className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                            LABEL_COLORS[r.label] || 'text-slate-300 bg-white/10 border border-white/10'
                          }`}
                        >
                          {r.label}
                        </span>
                      ) : (
                        <span className="text-red-300 text-xs">{r.error || 'error'}</span>
                      )}
                    </td>
                    <td className="px-4 py-2.5 text-slate-200 tabular-nums text-xs">
                      {r.confidence ? `${Math.round(r.confidence * 100)}%` : '-'}
                    </td>
                    <td className="px-4 py-2.5 text-slate-400 text-xs">
                      {r.generator?.is_unknown ? 'Unknown' : r.generator?.name || '-'}
                    </td>
                    <td className="px-4 py-2.5 text-slate-400 tabular-nums text-xs">
                      {r.model_signals?.vit_probability != null
                        ? `${Math.round(r.model_signals.vit_probability * 100)}%`
                        : '-'}
                    </td>
                    <td className="px-4 py-2.5 text-slate-400 tabular-nums text-xs">
                      {r.model_signals?.lgb_probability != null
                        ? `${Math.round(r.model_signals.lgb_probability * 100)}%`
                        : '-'}
                    </td>
                  </tr>
                ))}
              </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </main>
  )
}