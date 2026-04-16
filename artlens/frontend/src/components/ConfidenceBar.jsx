import { motion } from 'framer-motion'

const COLORS = {
  ai      : { bar: '#dc2626', bg: '#fef2f2', text: '#991b1b' },
  human   : { bar: '#16a34a', bg: '#f0fdf4', text: '#14532d' },
  unknown : { bar: '#d97706', bg: '#fffbeb', text: '#92400e' },
}

export default function ConfidenceBar({ label, confidence, isUnknown }) {
  const type   = isUnknown ? 'unknown' : label
  const colors = COLORS[type] || COLORS.ai
  const pct    = Math.round(confidence * 100)

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-slate-300">Confidence</span>
        <span
          className="text-lg font-semibold tabular-nums"
          style={{ color: colors.text }}
        >
          {pct}%
        </span>
      </div>
      <div className="h-3 rounded-full bg-slate-800/80 overflow-hidden">
        <motion.div
          className="h-full rounded-full"
          style={{ backgroundColor: colors.bar }}
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
        />
      </div>
    </div>
  )
}