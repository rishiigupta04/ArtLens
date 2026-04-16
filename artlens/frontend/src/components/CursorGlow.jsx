import { useEffect, useRef } from 'react'

export default function CursorGlow() {
  const glowRef = useRef(null)

  useEffect(() => {
    const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const coarsePointer = window.matchMedia('(pointer: coarse)').matches
    if (prefersReduced || coarsePointer) return undefined

    const target = { x: window.innerWidth / 2, y: window.innerHeight / 2 }
    const current = { x: target.x, y: target.y }
    let rafId = 0

    const onMove = (event) => {
      target.x = event.clientX
      target.y = event.clientY
    }

    const tick = () => {
      // Smoothly follow pointer while staying snappy.
      current.x += (target.x - current.x) * 0.28
      current.y += (target.y - current.y) * 0.28

      if (glowRef.current) {
        glowRef.current.style.transform = `translate3d(${current.x - 140}px, ${current.y - 140}px, 0)`
      }

      rafId = window.requestAnimationFrame(tick)
    }

    window.addEventListener('pointermove', onMove, { passive: true })
    tick()

    return () => {
      window.removeEventListener('pointermove', onMove)
      window.cancelAnimationFrame(rafId)
    }
  }, [])

  return (
    <>
      <div
        ref={glowRef}
        aria-hidden
        className="fixed -z-10 h-72 w-72 rounded-full bg-cyan-300/20 blur-3xl pointer-events-none"
      />
      <div
        aria-hidden
        className="fixed -z-10 h-40 w-40 rounded-full bg-violet-400/10 blur-2xl pointer-events-none animate-pulseGlow"
        style={{ top: '22%', left: '76%' }}
      />
    </>
  )
}