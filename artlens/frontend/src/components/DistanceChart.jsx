import { BarChart, Bar, Cell, XAxis, YAxis, Tooltip, ReferenceLine, ResponsiveContainer } from 'recharts'
import InfoTip from './InfoTip'

export default function DistanceChart({ distances }) {
  if (!distances) return null

  const data = Object.entries(distances).map(([name, value]) => ({
    name : name.replace(' (WikiArt)', ''),
    value: parseFloat(value),
    fill : value < 1
      ? '#16a34a'
      : value < 3
        ? '#d97706'
        : '#dc2626',
  }))

  const closest = [...data].sort((a, b) => a.value - b.value)[0]

  let plainLanguageHint = 'Lower bars mean a closer match to examples the model has seen before.'
  if (closest) {
    if (closest.value < 1) {
      plainLanguageHint = `This image is a close match for ${closest.name.toLowerCase()} examples.`
    } else if (closest.value < 3) {
      plainLanguageHint = `This image somewhat matches ${closest.name.toLowerCase()} examples, but not strongly.`
    } else {
      plainLanguageHint = 'This image does not closely match known examples, so it may be treated as unknown.'
    }
  }

  return (
    <div className="bg-slate-950/30 border border-white/10 rounded-xl p-4 space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wide">
            Similarity check
          </h3>
          <InfoTip
            title="How to read similarity"
            text="Lower bars mean the image is closer to known examples. If most bars are above the red line (3.0), ArtLens treats the generator as unknown."
            align="right"
          />
        </div>
        <span className="text-xs text-slate-400">lower is closer</span>
      </div>
      <ResponsiveContainer width="100%" height={120}>
        <BarChart data={data} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
          <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#94a3b8' }} />
          <YAxis tick={{ fontSize: 10, fill: '#94a3b8' }} />
          <Tooltip
            formatter={(v) => [`${v.toFixed(2)} σ`, 'Distance']}
            contentStyle={{ fontSize: 12, background: '#0f172a', border: '1px solid #334155', color: '#e2e8f0' }}
          />
          <ReferenceLine y={3} stroke="#dc2626" strokeDasharray="3 3"
                          label={{ value: 'Unknown line (3.0)', fontSize: 10, fill: '#dc2626' }} />
          <Bar dataKey="value" fill="#6b7280" radius={[3,3,0,0]}
          >
            {data.map((d, i) => <Cell key={i} fill={d.fill} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <p className="text-xs text-slate-400">
        {plainLanguageHint}
      </p>
      <p className="text-xs text-slate-500">
        Rule of thumb: under 1 means strong match, over 3 means poor match and possible unknown generator.
      </p>
    </div>
  )
}