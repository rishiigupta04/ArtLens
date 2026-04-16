export default function AnimatedGradientBg() {
  return (
    <div aria-hidden className="fixed inset-0 -z-10 overflow-hidden pointer-events-none">
      <div className="absolute -top-40 -left-24 h-80 w-80 rounded-full bg-cyan-400/20 blur-3xl animate-drift" />
      <div className="absolute top-1/4 -right-20 h-[24rem] w-[24rem] rounded-full bg-violet-500/20 blur-3xl animate-drift [animation-delay:1.5s]" />
      <div className="absolute bottom-[-8rem] left-1/3 h-[22rem] w-[22rem] rounded-full bg-fuchsia-400/10 blur-3xl animate-pulseGlow" />
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,_rgba(148,163,184,0.08),_transparent_45%)]" />
    </div>
  )
}