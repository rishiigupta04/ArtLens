import { Link, useLocation } from 'react-router-dom'

export default function Navbar() {
  const { pathname } = useLocation()
  const logoSrc = `${import.meta.env.BASE_URL}artlens-logo.png`

  const link = (to, label) => (
    <Link
      to={to}
      className={`text-sm font-medium px-3 py-1.5 rounded-lg transition-all
        ${pathname === to
          ? 'bg-white/20 text-slate-50 shadow'
          : 'text-slate-300 hover:text-white hover:bg-white/10'
        }`}
    >
      {label}
    </Link>
  )

  return (
    <>
    <nav className="z-50 border-b border-white/10 bg-slate-950/60 backdrop-blur-xl sm:sticky sm:top-0">
      <div className="max-w-6xl mx-auto px-4 h-16 sm:h-20 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-2 sm:gap-3 min-w-0">
          <img
            src={logoSrc}
            alt=""
            aria-hidden="true"
            className="h-10 w-10 sm:h-12 sm:w-12 object-contain shrink-0"
          />
          <span className="font-display text-xl sm:text-2xl font-light italic text-white whitespace-nowrap">
            Art<span className="text-cyan-300">Lens</span>
          </span>
          <span className="text-sm text-slate-400 font-medium hidden sm:block">
            AI Authenticity Engine
          </span>
        </Link>

        <div className="flex items-center gap-2 shrink-0">
          <div className="hidden sm:flex items-center gap-1 rounded-xl border border-white/10 bg-white/5 p-1">
            {link('/', 'Analyse')}
            {link('/batch', 'Batch')}
            {link('/about', 'About')}
          </div>
        </div>
      </div>
    </nav>
    <div className="sm:hidden fixed bottom-0 inset-x-0 z-50 border-t border-white/10 bg-slate-950/90 backdrop-blur-xl">
      <div className="px-4 py-3 grid grid-cols-3 gap-2">
        {[
          ['/', 'Analyse'],
          ['/batch', 'Batch'],
          ['/about', 'About'],
        ].map(([to, label]) => (
          <Link
            key={to}
            to={to}
            className={`text-center text-sm font-medium rounded-lg py-2 transition-all ${
              pathname === to
                ? 'bg-white/20 text-slate-50 shadow'
                : 'text-slate-300 bg-white/5 border border-white/10 hover:text-white hover:bg-white/10'
            }`}
          >
            {label}
          </Link>
        ))}
      </div>
    </div>
    </>
  )
}