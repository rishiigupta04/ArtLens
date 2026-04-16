const BADGES = {
  'Midjourney'       : { bg: 'rgba(59,130,246,0.12)', text: '#bfdbfe', border: 'rgba(147,197,253,0.35)', dot: '#60a5fa' },
  'Stable Diffusion' : { bg: 'rgba(139,92,246,0.12)', text: '#ddd6fe', border: 'rgba(196,181,253,0.35)', dot: '#a78bfa' },
  'Unknown'          : { bg: 'rgba(245,158,11,0.12)', text: '#fcd34d', border: 'rgba(251,191,36,0.35)', dot: '#fbbf24' },
}

export default function GeneratorBadge({ generator }) {
  if (!generator) return null
  const { name, confidence, is_unknown, closest_known } = generator

  const display = is_unknown ? 'Unknown generator' : name
  const style   = BADGES[is_unknown ? 'Unknown' : name] || BADGES['Unknown']
  const subtitle = is_unknown
    ? `Closest match: ${closest_known}`
    : confidence ? `${Math.round(confidence * 100)}% confident` : null

  return (
    <div
      className="inline-flex flex-col px-3 py-2 rounded-lg border text-sm"
      style={{ background: style.bg, borderColor: style.border }}
    >
      <div className="flex items-center gap-2">
        <div
          className="w-2 h-2 rounded-full"
          style={{ background: style.dot }}
        />
        <span className="font-medium" style={{ color: style.text }}>
          {display}
        </span>
      </div>
      {subtitle && (
        <span className="text-xs mt-0.5 ml-4" style={{ color: style.text, opacity: 0.7 }}>
          {subtitle}
        </span>
      )}
    </div>
  )
}