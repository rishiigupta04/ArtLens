import { useState } from 'react'
import { motion } from 'framer-motion'
import { Flag, ChevronDown, ChevronUp } from 'lucide-react'
import ConfidenceBar  from './ConfidenceBar'
import HeatmapViewer  from './HeatmapViewer'
import SignalBreakdown from './SignalBreakdown'
import DistanceChart  from './DistanceChart'
import FlagModal      from './FlagModal'
import * as signalMessaging from '../lib/signalMessaging'

export default function ResultCard({ result, file, originalSrc, onHeatmapsReady }) {
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [showFlag,     setShowFlag]     = useState(false)

  const isAI      = result.label === 'ai'
  const isUnknown = result.generator?.is_unknown
  const summaryText = signalMessaging.buildResultNarrative
    ? signalMessaging.buildResultNarrative(result)
    : (result?.explanation?.summary || 'ArtLens completed the analysis for this image.')

  // Colour scheme based on verdict
  const scheme = isUnknown
    ? { bg: 'bg-amber-500/10',  border: 'border-amber-300/30', badge: 'bg-amber-400/20 text-amber-200' }
    : isAI
      ? { bg: 'bg-rose-500/10',    border: 'border-rose-300/30',   badge: 'bg-rose-400/20 text-rose-200' }
      : { bg: 'bg-emerald-500/10',  border: 'border-emerald-300/30', badge: 'bg-emerald-400/20 text-emerald-200' }

  return (
    <>
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className={`rounded-2xl border-2 ${scheme.bg} ${scheme.border} overflow-hidden`}
      >
        {/* ── Header ── */}
        <div className="p-4 sm:p-5 space-y-4">
          <div className="flex items-start justify-between gap-4 flex-wrap">
            <div className="space-y-1">
              <div className={`inline-block px-2.5 py-1 rounded-full text-xs font-semibold ${scheme.badge}`}>
                {isUnknown ? 'AI-generated (unknown family)' : isAI ? 'AI-generated' : 'Human-made'}
              </div>
              <p className="text-base md:text-lg text-slate-100 leading-relaxed mt-2">
                {summaryText}
              </p>
            </div>
          </div>

          <ConfidenceBar
            label={result.label}
            confidence={result.confidence}
            isUnknown={isUnknown}
          />
        </div>

        {/* ── Heatmap + Explanation detail ── */}
        <div className="px-4 sm:px-5 pb-4 sm:pb-5 grid grid-cols-1 md:grid-cols-2 gap-4">
          <HeatmapViewer
            file={file}
            originalSrc={originalSrc}
            analysisId={result.request_id}
            initialHeatmaps={result.heatmap_urls || null}
            onHeatmapsReady={onHeatmapsReady}
          />

          <div className="space-y-3">
            {result.explanation?.detail && (
              <div className="bg-slate-950/30 border border-white/10 rounded-xl p-3">
                <p className="text-sm text-slate-200 leading-relaxed">
                  {result.explanation.detail}
                </p>
              </div>
            )}
            <SignalBreakdown signals={result.model_signals} />
          </div>
        </div>

        {/* ── Advanced section (collapsed by default) ── */}
        <div className="border-t border-white/10">
          <button
            onClick={() => setShowAdvanced(v => !v)}
            className="w-full flex items-center justify-between px-5 py-3
                       text-xs font-medium text-slate-400 hover:text-slate-200"
          >
            <span>Advanced details</span>
            {showAdvanced ? <ChevronUp size={14}/> : <ChevronDown size={14}/>}
          </button>

          {showAdvanced && (
            <div className="px-4 sm:px-5 pb-4 sm:pb-5 space-y-3">
              <div className="bg-slate-950/30 border border-white/10 rounded-xl p-3">
                <p className="text-xs text-slate-300 leading-relaxed">
                  This section compares your image with known visual fingerprints. If every distance is high (especially above 3), ArtLens treats it as "unknown" because it does not clearly resemble known generator families.
                </p>
              </div>
              <DistanceChart distances={result.generator?.normalised_distances} />

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
                {[
                  ['Request ID',     result.request_id],
                  ['Model version',  result.model_version],
                  ['Inference time', `${result.inference_time_s}s`],
                  ['Cached',         result.cached ? 'Yes' : 'No'],
                  ['Temperature',    result.temperature],
                  ['ViT probability (visual cues)', result.model_signals?.vit_probability != null ? `${(result.model_signals.vit_probability * 100).toFixed(1)}%` : '-'],
                  ['LGB probability (frequency cues)', result.model_signals?.lgb_probability != null ? `${(result.model_signals.lgb_probability * 100).toFixed(1)}%` : '-'],
                ].map(([k, v]) => (
                  <div key={k} className="bg-slate-950/30 border border-white/10 rounded-lg px-3 py-2">
                    <span className="text-slate-400 block">{k}</span>
                    <span className="font-medium text-slate-100">{v}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* ── Footer ── */}
        <div className="px-4 sm:px-5 py-3 border-t border-white/10 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
          <span className="text-xs text-slate-400">
            {result.inference_time_s}s · {result.model_version}
          </span>
          <button
            onClick={() => setShowFlag(true)}
            className="flex items-center gap-1.5 text-xs text-slate-400
                       hover:text-amber-300 transition-colors"
          >
            <Flag size={12} />
            Flag as incorrect
          </button>
        </div>
      </motion.div>

      {showFlag && (
        <FlagModal
          result={result}
          imageHash={result.request_id}
          onClose={() => setShowFlag(false)}
        />
      )}
    </>
  )
}