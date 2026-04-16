import { useMemo, useState } from 'react'
import { Info } from 'lucide-react'

export default function InfoTip({ title = 'About this', text, align = 'left' }) {
  const [open, setOpen] = useState(false)
  const placementClass = useMemo(() => {
    if (align === 'right') return 'left-0'
    return 'right-0'
  }, [align])

  return (
    <div className="relative inline-flex">
      <button
        type="button"
        aria-label={title}
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
        onBlur={() => setOpen(false)}
        className="inline-flex h-5 w-5 items-center justify-center rounded-full border border-white/20 text-slate-300 hover:text-white hover:border-white/40 transition-colors"
      >
        <Info size={12} />
      </button>

      {open && (
        <div className={`absolute ${placementClass} top-7 z-20 w-64 rounded-lg border border-white/15 bg-slate-950/95 p-3 text-xs text-slate-200 shadow-xl`}>
          <p className="font-semibold text-slate-100">{title}</p>
          <p className="mt-1 leading-relaxed text-slate-300">{text}</p>
        </div>
      )}
    </div>
  )
}