import React, { useEffect, useMemo, useRef, useState } from 'react'

const STYLE_ID = 'hero3-animations'

const getRootTheme = () => {
  if (typeof document === 'undefined') {
    if (typeof window !== 'undefined' && window.matchMedia)
      return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
    return 'dark'
  }
  const root = document.documentElement
  if (root.classList.contains('dark')) return 'dark'
  if (root.getAttribute('data-theme') === 'dark' || root.dataset?.theme === 'dark') return 'dark'
  if (root.classList.contains('light')) return 'light'
  if (typeof window !== 'undefined' && window.matchMedia)
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  return 'dark'
}

const useThemeSync = () => {
  const [theme, setTheme] = useState(() => getRootTheme())
  useEffect(() => {
    if (typeof document === 'undefined') return
    const sync = () => {
      const next = getRootTheme()
      setTheme(prev => (prev === next ? prev : next))
    }
    sync()
    const observer = new MutationObserver(sync)
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class', 'data-theme'] })
    const media = typeof window !== 'undefined' && window.matchMedia
      ? window.matchMedia('(prefers-color-scheme: dark)') : null
    media?.addEventListener('change', sync)
    const onStorage = e => { if (e.key === 'hero-theme') sync() }
    if (typeof window !== 'undefined') window.addEventListener('storage', onStorage)
    return () => {
      observer.disconnect()
      media?.removeEventListener('change', sync)
      if (typeof window !== 'undefined') window.removeEventListener('storage', onStorage)
    }
  }, [])
  return [theme, setTheme]
}

const DeckGlyph = ({ theme = 'dark' }) => {
  const stroke = theme === 'dark' ? '#f5f5f5' : '#111111'
  const fill = theme === 'dark' ? 'rgba(255,255,255,0.08)' : 'rgba(17,17,17,0.08)'
  return (
    <svg viewBox="0 0 120 120" className="h-16 w-16" aria-hidden>
      <circle cx="60" cy="60" r="46" fill="none" stroke={stroke} strokeWidth="1.4"
        className="motion-safe:animate-[hero3-orbit_8.5s_linear_infinite] motion-reduce:animate-none"
        style={{ strokeDasharray: '18 14' }} />
      <rect x="34" y="34" width="52" height="52" rx="14" fill={fill} stroke={stroke} strokeWidth="1.2"
        className="motion-safe:animate-[hero3-grid_5.4s_ease-in-out_infinite] motion-reduce:animate-none" />
      <circle cx="60" cy="60" r="7" fill={stroke} />
      <path d="M60 30v10M60 80v10M30 60h10M80 60h10" stroke={stroke} strokeWidth="1.4" strokeLinecap="round"
        className="motion-safe:animate-[hero3-pulse_6s_ease-in-out_infinite] motion-reduce:animate-none" />
    </svg>
  )
}

export function HeroOrbitDeck({ onEnterDashboard }) {
  const [theme, setTheme] = useThemeSync()
  const [visible, setVisible] = useState(false)
  const [mode, setMode] = useState('prediction')
  const sectionRef = useRef(null)

  useEffect(() => {
    if (typeof document === 'undefined') return
    if (document.getElementById(STYLE_ID)) return
    const style = document.createElement('style')
    style.id = STYLE_ID
    style.innerHTML = `
      @keyframes hero3-intro {
        0% { opacity: 0; transform: translate3d(0, 64px, 0) scale(0.98); filter: blur(12px); }
        60% { filter: blur(0); }
        100% { opacity: 1; transform: translate3d(0, 0, 0) scale(1); filter: blur(0); }
      }
      @keyframes hero3-orbit {
        0% { stroke-dashoffset: 0; transform: rotate(0deg); }
        100% { stroke-dashoffset: -64; transform: rotate(360deg); }
      }
      @keyframes hero3-grid {
        0%, 100% { transform: rotate(-2deg); opacity: 0.7; }
        50% { transform: rotate(2deg); opacity: 1; }
      }
      @keyframes hero3-pulse {
        0%, 100% { stroke-dasharray: 0 200; opacity: 0.2; }
        45%, 60% { stroke-dasharray: 200 0; opacity: 1; }
      }
      @keyframes hero3-glow {
        0%, 100% { opacity: 0.45; transform: translate3d(0,0,0); }
        50% { opacity: 0.9; transform: translate3d(0,-8px,0); }
      }
      @keyframes hero3-drift {
        0%, 100% { transform: translate3d(0,0,0) rotate(-3deg); }
        50% { transform: translate3d(0,-12px,0) rotate(3deg); }
      }
    `
    document.head.appendChild(style)
    return () => { style.remove() }
  }, [])

  useEffect(() => {
    if (!sectionRef.current || typeof window === 'undefined') { setVisible(true); return }
    const observer = new IntersectionObserver(
      entries => { entries.forEach(e => { if (e.isIntersecting) { setVisible(true); observer.disconnect() } }) },
      { threshold: 0.2 }
    )
    observer.observe(sectionRef.current)
    return () => observer.disconnect()
  }, [])

  const toggleTheme = () => {
    if (typeof document === 'undefined') return
    const root = document.documentElement
    const current = getRootTheme()
    const next = current === 'dark' ? 'light' : 'dark'
    root.classList.toggle('dark', next === 'dark')
    root.classList.toggle('light', next === 'light')
    root.setAttribute('data-theme', next)
    try { window.localStorage?.setItem('hero-theme', next) } catch (_) {}
    setTheme(next)
  }

  const palette = useMemo(() =>
    theme === 'dark'
      ? {
          surface: 'bg-black text-white',
          subtle: 'text-white/60',
          border: 'border-white/12',
          card: 'bg-white/6',
          accent: 'bg-white/12',
          glow: 'rgba(255,255,255,0.14)',
          background: {
            color: '#040404',
            layers: [
              'radial-gradient(ellipse 80% 60% at 10% -10%, rgba(0,212,255,0.18), transparent 60%)',
              'radial-gradient(ellipse 90% 70% at 90% -20%, rgba(0,255,136,0.10), transparent 70%)',
            ],
            dots: 'radial-gradient(circle at 25% 25%, rgba(250,250,250,0.08) 0.7px, transparent 1px), radial-gradient(circle at 75% 75%, rgba(250,250,250,0.08) 0.7px, transparent 1px)',
          },
        }
      : {
          surface: 'bg-white text-neutral-950',
          subtle: 'text-neutral-600',
          border: 'border-neutral-200/80',
          card: 'bg-neutral-100/80',
          accent: 'bg-neutral-100',
          glow: 'rgba(17,17,17,0.08)',
          background: {
            color: '#f5f5f4',
            layers: [
              'radial-gradient(ellipse 80% 60% at 10% -10%, rgba(0,150,200,0.12), transparent 60%)',
              'radial-gradient(ellipse 90% 70% at 90% -20%, rgba(0,200,100,0.08), transparent 70%)',
            ],
            dots: 'radial-gradient(circle at 25% 25%, rgba(17,17,17,0.12) 0.7px, transparent 1px), radial-gradient(circle at 75% 75%, rgba(17,17,17,0.08) 0.7px, transparent 1px)',
          },
        },
  [theme])

  const metrics = [
    { label: 'AUC Score', value: '0.87+' },
    { label: 'Missions', value: '10K+' },
    { label: 'Compute saved', value: '80%' },
  ]

  const modes = useMemo(() => ({
    prediction: {
      title: 'Prediction mode',
      description: 'Deploy the early-exit LSTM to classify trajectory success from just 20–60% of telemetry — surfacing go/no-go signals before the simulation completes.',
      items: [
        'Bidirectional LSTM on partial trajectory',
        'P(success) score at configurable exit fraction',
        'Binary, multiclass, and regression targets',
      ],
    },
    analysis: {
      title: 'Analysis mode',
      description: 'Run the ablation sweep to map AUC vs telemetry fraction — the core research finding: how early can the model reliably predict failure?',
      items: [
        'AUC vs early-exit fraction curve',
        '13 physics-invariant feature diagnostics',
        'Cross-planetary generalization report',
      ],
    },
  }), [])

  const activeMode = modes[mode]

  const protocols = [
    { name: 'Pre-flight Screening', detail: 'Ingest GMAT trajectory CSV, validate 13 feature columns, normalize with mission scaler.', status: 'Active' },
    { name: 'Trajectory Analysis', detail: 'Feed partial sequence into LSTM at selected early-exit fraction, compute sigmoid probability.', status: 'Running' },
    { name: 'Mission Decision', detail: 'Return P(success) with confidence band — go/no-go threshold configurable per mission profile.', status: 'Live' },
  ]

  const setSpotlight = e => {
    const rect = e.currentTarget.getBoundingClientRect()
    e.currentTarget.style.setProperty('--hero3-x', `${e.clientX - rect.left}px`)
    e.currentTarget.style.setProperty('--hero3-y', `${e.clientY - rect.top}px`)
  }
  const clearSpotlight = e => {
    e.currentTarget.style.removeProperty('--hero3-x')
    e.currentTarget.style.removeProperty('--hero3-y')
  }

  return (
    <div className={`relative isolate min-h-screen w-full transition-colors duration-700 ${palette.surface}`}>
      <div className="pointer-events-none absolute inset-0 -z-30"
        style={{ backgroundColor: palette.background.color, backgroundImage: palette.background.layers.join(', '), backgroundRepeat: 'no-repeat', backgroundSize: 'cover' }} />
      <div className="pointer-events-none absolute inset-0 -z-20 opacity-80"
        style={{ backgroundImage: palette.background.dots, backgroundSize: '12px 12px', backgroundRepeat: 'repeat' }} />
      <div className="pointer-events-none absolute inset-0 -z-10"
        style={{ background: theme === 'dark' ? 'radial-gradient(60% 50% at 50% 10%, rgba(0,212,255,0.12), transparent 70%)' : 'radial-gradient(60% 50% at 50% 10%, rgba(0,180,255,0.1), transparent 70%)', filter: 'blur(22px)' }} />

      <section
        ref={sectionRef}
        className={`relative flex min-h-screen w-full flex-col gap-16 px-6 pt-24 pb-24 transition-opacity duration-700 md:gap-20 md:px-10 lg:px-16 xl:px-24 ${visible ? 'motion-safe:animate-[hero3-intro_1s_cubic-bezier(.22,.68,0,1)_forwards]' : 'opacity-0'}`}
      >
        <header className="grid gap-10 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,0.9fr)] lg:items-end">
          <div className="space-y-8">
            <div className="space-y-3">
              <div style={{ fontSize: '11px', letterSpacing: '0.4em', color: '#00d4ff', textTransform: 'uppercase', fontWeight: 600, opacity: 0.85 }}>
                NASA GMAT · AI Mission Prediction
              </div>
              <div style={{ fontSize: 'clamp(48px, 6vw, 80px)', fontWeight: 900, letterSpacing: '-0.02em', lineHeight: 1, background: 'linear-gradient(135deg, #ffffff 40%, rgba(0,212,255,0.85) 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text', fontFamily: "'Space Grotesk', sans-serif" }}>
                ORBITGUARD
              </div>
              <p className={`max-w-xl text-base md:text-lg ${palette.subtle}`} style={{ paddingTop: 4 }}>
                Predict spacecraft mission failure before it happens — a bidirectional LSTM trained on 85.3M trajectory points, classifying success from just 20% of telemetry.
              </p>
            </div>

            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div className={`inline-flex flex-wrap gap-3 rounded-full border px-5 py-3 text-xs uppercase tracking-[0.3em] transition ${palette.border} ${palette.accent}`}>
                <span className="flex items-center gap-2">
                  <span className="h-1.5 w-1.5 rounded-full bg-current animate-pulse" />
                  System online
                </span>
                <span className="opacity-60">∙</span>
                <span>Real-time inference</span>
              </div>
              <div className={`flex divide-x overflow-hidden rounded-full border text-xs uppercase tracking-[0.35em] ${palette.border} ${theme === 'dark' ? 'divide-white/10' : 'divide-neutral-200'}`}>
                {metrics.map(m => (
                  <div key={m.label} className="flex flex-col px-5 py-3">
                    <span className={`text-[11px] ${palette.subtle}`}>{m.label}</span>
                    <span className="text-lg font-semibold tracking-tight">{m.value}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className={`relative flex flex-col gap-6 rounded-3xl border p-8 transition ${palette.border} ${palette.card}`}>
            <div className="flex items-start justify-between gap-4">
              <div className="space-y-3">
                <p className="text-xs uppercase tracking-[0.35em]">Mode</p>
                <h2 className="text-xl font-semibold tracking-tight">{activeMode.title}</h2>
              </div>
              <DeckGlyph theme={theme} />
            </div>
            <p className={`text-sm leading-relaxed ${palette.subtle}`}>{activeMode.description}</p>
            <div className="flex gap-2">
              <button type="button" onClick={() => setMode('prediction')}
                className={`flex-1 rounded-full border px-4 py-2 text-xs font-semibold uppercase tracking-[0.35em] transition ${mode === 'prediction' ? 'bg-white text-black dark:bg-white/90 dark:text-black' : `${palette.border} ${palette.accent}`}`}>
                Prediction
              </button>
              <button type="button" onClick={() => setMode('analysis')}
                className={`flex-1 rounded-full border px-4 py-2 text-xs font-semibold uppercase tracking-[0.35em] transition ${mode === 'analysis' ? 'bg-white text-black dark:bg-white/90 dark:text-black' : `${palette.border} ${palette.accent}`}`}>
                Analysis
              </button>
            </div>
            <ul className="space-y-2 text-sm">
              {activeMode.items.map(item => (
                <li key={item} className={`flex items-start gap-3 ${palette.subtle}`}>
                  <span className="mt-1 h-2 w-2 rounded-full bg-current" />
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </div>
        </header>

        <div className="grid gap-10 xl:grid-cols-[minmax(0,0.9fr)_minmax(0,1.2fr)_minmax(0,0.9fr)] xl:items-stretch">
          {/* Left panel */}
          <div className={`order-2 flex flex-col gap-6 rounded-3xl border p-8 transition ${palette.border} ${palette.card} xl:order-1`}>
            <div className="flex items-center justify-between">
              <h3 className="text-xs uppercase tracking-[0.35em]">Model stack</h3>
              <span className="text-xs uppercase tracking-[0.35em] opacity-60">v4.0</span>
            </div>
            <p className={`text-sm leading-relaxed ${palette.subtle}`}>
              Physics-invariant features in the synodic reference frame allow the model to generalise across Earth–Moon, Mars, and Jupiter missions without retraining.
            </p>
            <div className="grid gap-3">
              {['Physics-invariant features', 'Multi-planetary inference', 'Early exit threshold control'].map(item => (
                <div key={item} className="relative overflow-hidden rounded-2xl border px-4 py-3 text-xs uppercase tracking-[0.3em] transition duration-500 hover:-translate-y-0.5 hover:shadow-[0_14px_40px_rgba(0,0,0,0.18)] dark:hover:shadow-[0_14px_40px_rgba(0,0,0,0.45)]">
                  <span>{item}</span>
                  <span className="pointer-events-none absolute inset-0 opacity-0 transition duration-500 hover:opacity-100"
                    style={{ background: `radial-gradient(180px circle at 50% 20%, ${palette.glow}, transparent 70%)` }} />
                </div>
              ))}
            </div>
            {onEnterDashboard && (
              <button onClick={onEnterDashboard}
                className={`mt-2 w-full rounded-full border px-4 py-3 text-xs font-semibold uppercase tracking-[0.3em] transition duration-300 hover:-translate-y-0.5 ${palette.border} ${palette.accent}`}>
                Open Mission Control →
              </button>
            )}
          </div>

          {/* Center image */}
          <figure className="order-1 overflow-hidden rounded-[32px] border transition xl:order-2" style={{ position: 'relative' }}>
            <div className="relative w-full pb-[120%] sm:pb-[90%] lg:pb-[72%]">
              <img
                src="https://images.unsplash.com/photo-1446776653964-20c1d3a81b06?w=800&h=960&fit=crop&q=80"
                alt="Earth from orbit — spacecraft mission perspective"
                loading="lazy"
                className="absolute inset-0 h-full w-full object-cover grayscale transition duration-700 ease-out hover:scale-[1.03]"
              />
              <span className="pointer-events-none absolute inset-0 bg-gradient-to-b from-black/40 via-transparent to-black/50 mix-blend-soft-light dark:from-white/10" />
              <div className="pointer-events-none absolute inset-0 border border-white/10 mix-blend-overlay dark:border-white/20" />
              <span className="pointer-events-none absolute -left-16 top-16 h-40 w-40 rounded-full border border-white/15 opacity-70 motion-safe:animate-[hero3-glow_9s_ease-in-out_infinite]" />
              <span className="pointer-events-none absolute -right-12 bottom-16 h-48 w-48 rounded-full border border-white/10 opacity-40 motion-safe:animate-[hero3-drift_12s_ease-in-out_infinite]" />
            </div>
            <figcaption className={`flex items-center justify-between px-6 py-5 text-xs uppercase tracking-[0.35em] ${palette.subtle}`}>
              <span>Trajectory composite</span>
              <span className="flex items-center gap-2">
                <span className="h-1 w-8 bg-current" />
                3-body RK4 physics
              </span>
            </figcaption>
          </figure>

          {/* Right panel — protocols */}
          <aside className={`order-3 flex flex-col gap-6 rounded-3xl border p-8 transition ${palette.border} ${palette.card} xl:order-3`}>
            <div className="flex items-center justify-between">
              <h3 className="text-xs uppercase tracking-[0.35em]">Mission protocols</h3>
              <span className="text-xs uppercase tracking-[0.35em] opacity-60">Indexed</span>
            </div>
            <ul className="space-y-4">
              {protocols.map((protocol, index) => (
                <li key={protocol.name} onMouseMove={setSpotlight} onMouseLeave={clearSpotlight}
                  className="group relative overflow-hidden rounded-2xl border px-5 py-4 transition duration-500 hover:-translate-y-0.5"
                  style={{ animationDelay: `${index * 0.1}s` }}>
                  <div className="pointer-events-none absolute inset-0 opacity-0 transition duration-500 group-hover:opacity-100"
                    style={{ background: theme === 'dark'
                      ? 'radial-gradient(190px circle at var(--hero3-x, 50%) var(--hero3-y, 50%), rgba(255,255,255,0.18), transparent 72%)'
                      : 'radial-gradient(190px circle at var(--hero3-x, 50%) var(--hero3-y, 50%), rgba(17,17,17,0.12), transparent 72%)' }} />
                  <div className="flex items-center justify-between">
                    <h4 className="text-sm font-semibold uppercase tracking-[0.25em]">{protocol.name}</h4>
                    <span className="text-[10px] uppercase tracking-[0.35em] opacity-70">{protocol.status}</span>
                  </div>
                  <p className={`mt-3 text-sm leading-relaxed ${palette.subtle}`}>{protocol.detail}</p>
                </li>
              ))}
            </ul>
          </aside>
        </div>
      </section>
    </div>
  )
}

export default HeroOrbitDeck
