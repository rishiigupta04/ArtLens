import * as signalMessaging from '../lib/signalMessaging'
import InfoTip from './InfoTip'

export default function SignalBreakdown({ signals }) {
  if (!signals) return null
  const vitProbability = signalMessaging.toProbability(signals.vit_probability)
  const lgbProbability = signalMessaging.toProbability(signals.lgb_probability)
  const ensembleProbability = signalMessaging.toProbability(signals.ensemble_probability)
  const consensus = signalMessaging.deriveSignalConsensus
    ? signalMessaging.deriveSignalConsensus(signals)
    : {
        tone: 'neutral',
        badge: 'Signal check unavailable',
        detail: 'ArtLens could not compare the two checks for this image.',
      }

  const toneClass = {
    good: 'bg-emerald-500/20 text-emerald-300',
    caution: 'bg-amber-500/20 text-amber-300',
    warn: 'bg-rose-500/20 text-rose-300',
    neutral: 'bg-slate-500/20 text-slate-300',
  }[consensus.tone] || 'bg-slate-500/20 text-slate-300'

  const Bar = ({ value, label, color }) => (
    <div className="space-y-1">
      <div className="flex justify-between text-sm text-slate-300">
        <span>{label}</span>
        <span className="font-medium tabular-nums">{value != null ? `${Math.round(value * 100)}%` : '-'}</span>
      </div>
      <div className="h-1.5 rounded-full bg-slate-800 overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${(value ?? 0) * 100}%`, background: color }}
        />
      </div>
    </div>
  )

  return (
    <div className="bg-slate-950/30 border border-white/10 rounded-xl p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wide">
            Signal breakdown
          </h3>
          <InfoTip
            title="ViT and LGB"
            text="ViT reads visual style and composition patterns. LGB reads texture and frequency clues. The final score combines both signals."
          />
        </div>
        <span className={`text-sm px-2 py-0.5 rounded-full font-medium ${toneClass}`}>{consensus.badge}</span>
      </div>
      <p className="text-sm text-slate-300 leading-relaxed">
        {consensus.detail}
      </p>
      <Bar value={vitProbability}     label="Deep visual (ViT: style and composition cues)"      color="#3b82f6" />
      <Bar value={lgbProbability}     label="Frequency analysis (LGB: texture and noise cues)" color="#8b5cf6" />
      <Bar value={ensembleProbability} label="Final combined score (ensemble)"       color="#1d4ed8" />
    </div>
  )
}