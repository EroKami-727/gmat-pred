import { useState, useEffect, useRef } from 'react'
import { Zap, Globe2, BrainCircuit, Ruler, Target, Satellite, Rocket, Radio, Cpu, CheckCircle2, ArrowUpRight } from 'lucide-react'
import { Globe } from './components/ui/globe'
import { LogoCarousel } from './components/ui/logo-carousel'
import { FlowFieldBackground } from './components/ui/flow-field-background'
import { FaqSection } from './components/ui/faq-accordion'

// ── Fonts injected once ──────────────────────────────────────────────────────
const FontLink = () => (
  <style>{`
    @import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,9..144,300;0,9..144,700;0,9..144,800;0,9..144,900;1,9..144,400&family=Space+Mono:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    .lg-root { font-family: 'Geist', sans-serif; }
    .lg-heading { font-family: 'Newsreader', serif; font-optical-sizing: auto; }
    @keyframes fadeInUp { from { opacity:0; transform:translateY(40px); } to { opacity:1; transform:translateY(0); } }
    @keyframes fadeIn   { from { opacity:0; } to { opacity:1; } }
    @keyframes float    { 0%,100% { transform:translateY(0); } 50% { transform:translateY(-16px); } }
    @keyframes orbit    { from { transform:rotate(0deg) translateX(140px) rotate(0deg); } to { transform:rotate(360deg) translateX(140px) rotate(-360deg); } }
    @keyframes orbit2   { from { transform:rotate(120deg) translateX(100px) rotate(-120deg); } to { transform:rotate(480deg) translateX(100px) rotate(-480deg); } }
    @keyframes twinkle  { 0%,100% { opacity:1; transform:scale(1); } 50% { opacity:0.3; transform:scale(0.6); } }
    @keyframes pulseGlow { 0%,100% { box-shadow:0 0 8px #aaaaaa; } 50% { box-shadow:0 0 20px #aaaaaa; } }
    .nav-link { color:#888888; text-decoration:none; font-size:14px; font-weight:500; transition:color 0.2s; }
    .nav-link:hover { color:#ffffff; }
    .footer-link { color:#888888; text-decoration:none; font-size:14px; line-height:2.4; transition:color 0.2s; }
    .footer-link:hover { color:#ffffff; }
    .feat-card { background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.07); border-radius:24px; padding:36px 32px; transition:transform 0.35s cubic-bezier(0.16,1,0.3,1),border-color 0.3s,background 0.3s,box-shadow 0.35s; position:relative; overflow:hidden; }
    .feat-card::before { content:''; position:absolute; inset:0; background:radial-gradient(ellipse 80% 60% at 50% 0%,rgba(226,226,226,0.06),transparent 70%); opacity:0; transition:opacity 0.35s; pointer-events:none; }
    .feat-card:hover { transform:translateY(-10px); border-color:rgba(226,226,226,0.28); background:rgba(226,226,226,0.04); box-shadow:0 24px 64px rgba(0,0,0,0.5),0 0 0 1px rgba(226,226,226,0.1); }
    .feat-card:hover::before { opacity:1; }
    .partner-badge { padding:14px 28px; border-radius:12px; background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); font-family:'Geist',sans-serif; font-size:16px; font-weight:700; color:rgba(255,255,255,0.45); letter-spacing:0.04em; transition:color 0.3s,border-color 0.3s,background 0.3s; cursor:default; }
    .partner-badge:hover { color:rgba(255,255,255,0.9); border-color:rgba(226,226,226,0.4); background:rgba(226,226,226,0.06); }
    ::-webkit-scrollbar { width:6px; }
    ::-webkit-scrollbar-track { background:#090909; }
    ::-webkit-scrollbar-thumb { background:rgba(255,255,255,0.15); border-radius:3px; }
    ::-webkit-scrollbar-thumb:hover { background:rgba(255,255,255,0.3); }
    .clip-grad {
      background: linear-gradient(135deg, #e2e2e2 0%, #888888 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    @media (max-width:768px) {
      .feat-grid-3 { grid-template-columns:1fr!important; }
      .step-grid { grid-template-columns:1fr 1fr!important; }
      .footer-grid { grid-template-columns:1fr 1fr!important; }
    }
  `}</style>
)

// ── Animated counter ─────────────────────────────────────────────────────────
function Counter({ end, suffix = '', prefix = '', decimals = 0 }) {
  const [val, setVal] = useState(0)
  const ref = useRef(null)
  const started = useRef(false)
  useEffect(() => {
    const obs = new IntersectionObserver(([e]) => {
      if (e.isIntersecting && !started.current) {
        started.current = true
        const dur = 2200, start = performance.now()
        const tick = now => {
          const t = Math.min((now - start) / dur, 1)
          const ease = 1 - Math.pow(1 - t, 3)
          setVal(+(ease * end).toFixed(decimals))
          if (t < 1) requestAnimationFrame(tick)
        }
        requestAnimationFrame(tick)
      }
    }, { threshold: 0.3 })
    if (ref.current) obs.observe(ref.current)
    return () => obs.disconnect()
  }, [end, decimals])
  return <div ref={ref} style={{ fontFamily: "'Geist',sans-serif", fontSize: 'clamp(40px,5vw,62px)', fontWeight: 800, color: '#ffffff', lineHeight: 1 }}>
    {prefix}{decimals ? val.toFixed(decimals) : val.toLocaleString()}{suffix}
  </div>
}

// ── Fade-in on scroll wrapper ────────────────────────────────────────────────
function FadeIn({ children, style }) {
  const ref = useRef(null)
  const [v, setV] = useState(false)
  useEffect(() => {
    const obs = new IntersectionObserver(([e]) => { if (e.isIntersecting) setV(true) }, { threshold: 0.07 })
    if (ref.current) obs.observe(ref.current)
    return () => obs.disconnect()
  }, [])
  return <div ref={ref} style={{ transition: 'opacity 0.8s ease, transform 0.8s ease', opacity: v ? 1 : 0, transform: v ? 'translateY(0)' : 'translateY(36px)', ...style }}>{children}</div>
}

// ── Interactive Workflow Section ─────────────────────────────────────────
const STEPS = [
  {
    step: '01', Icon: Rocket, color: '#ffffff', title: 'Run GMAT Sim',
    desc: 'Generate trajectory telemetry via NASA GMAT with 3-body physics',
    detail: 'Configure your mission parameters — initial state vectors, target body, and propagation settings. GMAT runs a full 3-body RK4 integration, producing a time-series CSV of spacecraft state vectors across the entire trajectory arc.',
    tags: ['RK4 Integrator', '3-Body Physics', 'State Vectors', 'CSV Export'],
  },
  {
    step: '02', Icon: Radio, color: '#cccccc', title: 'Extract Features',
    desc: '13 physics-invariant features in synodic reference frame',
    detail: 'A preprocessing script converts raw Cartesian state vectors into 13 physics-invariant features: synodic-frame positions, specific orbital energy, eccentricity, flight path angle, SOI ratios, and velocity magnitude — all frame-invariant across planets.',
    tags: ['Synodic Frame', 'Orbital Energy', 'Eccentricity', '13-Dim Vector'],
  },
  {
    step: '03', Icon: Cpu, color: '#aaaaaa', title: 'LSTM Inference',
    desc: 'Feed partial trajectory into pre-trained model at exit fraction',
    detail: 'Select your exit fraction (10%–100%). The BiLSTM processes the truncated sequence — packed with variable-length masking — and outputs a binary classification probability P(success). Inference completes in under 50ms.',
    tags: ['BiLSTM', 'Early Exit', 'Pack Padded', 'Sub-50ms'],
  },
  {
    step: '04', Icon: CheckCircle2, color: '#888888', title: 'Go / No-Go',
    desc: 'Instant P(success) score — abort or proceed with confidence',
    detail: 'The dashboard displays P(success) with a confidence threshold slider. Missions below threshold are flagged for abort, saving up to 80% of simulation compute time. All predictions are logged with feature vectors for auditability.',
    tags: ['P(success)', 'Threshold', 'Audit Log', '80% Compute Saved'],
  },
]

function WorkflowSection() {
  const [active, setActive] = useState(0)
  const [hovered, setHovered] = useState(null)

  return (
    <section style={{ padding: '100px 5vw', background: 'linear-gradient(180deg,transparent,rgba(226,226,226,0.03) 50%,transparent)' }}>
      <style>{`
        .wf-step { cursor: pointer; transition: all 0.3s ease; }
        .wf-step:hover .wf-circle { transform: scale(1.1); }
        .wf-circle { transition: all 0.35s cubic-bezier(0.16,1,0.3,1); }
        .wf-detail { animation: wfSlide 0.35s cubic-bezier(0.16,1,0.3,1) forwards; }
        @keyframes wfSlide { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:translateY(0); } }
        .wf-tag { transition: background 0.2s, color 0.2s; }
        .wf-tag:hover { background: rgba(226,226,226,0.15) !important; color: #ffffff !important; }
      `}</style>

      <FadeIn>
        <div style={{ maxWidth: 1100, margin: '0 auto' }}>

          {/* Heading */}
          <div style={{ textAlign: 'center', marginBottom: 72 }}>
            <div className="lg-heading" style={{ fontSize: 13, fontWeight: 600, color: '#aaaaaa', letterSpacing: '0.15em', marginBottom: 14, textTransform: 'uppercase' }}>Workflow</div>
            <h2 className="lg-heading" style={{ fontSize: 'clamp(34px,4.5vw,58px)', fontWeight: 700, lineHeight: 1.1, color: '#ffffff' }}>
              From simulation to <span className="clip-grad">decision in seconds</span>
            </h2>
          </div>

          {/* Step nodes + animated connector */}
          <div style={{ position: 'relative', display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 0 }}>
            {/* Background track */}
            <div style={{ position: 'absolute', top: 44, left: '12.5%', right: '12.5%', height: 2, background: 'rgba(255,255,255,0.07)', zIndex: 0 }} />
            {/* Filled progress track */}
            <div style={{
              position: 'absolute', top: 44, left: '12.5%', height: 2,
              width: `${(active / 3) * 75}%`,
              background: 'linear-gradient(90deg,#ffffff,#888888)',
              zIndex: 0, transition: 'width 0.5s cubic-bezier(0.16,1,0.3,1)',
            }} />

            {STEPS.map(({ step, Icon, color, title }, i) => {
              const isActive = active === i
              const isPast = i < active
              const isHov = hovered === i
              return (
                <div
                  key={step}
                  className="wf-step"
                  onClick={() => setActive(i)}
                  onMouseEnter={() => setHovered(i)}
                  onMouseLeave={() => setHovered(null)}
                  style={{ textAlign: 'center', padding: '0 16px', position: 'relative', zIndex: 1 }}
                >
                  {/* Circle */}
                  <div className="wf-circle" style={{
                    width: 88, height: 88, borderRadius: '50%',
                    background: isActive ? 'rgba(255,255,255,0.08)' : isPast ? 'rgba(255,255,255,0.04)' : 'rgba(255,255,255,0.02)',
                    border: `2px solid ${isActive ? '#ffffff' : isPast ? 'rgba(255,255,255,0.4)' : 'rgba(255,255,255,0.15)'}`,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    margin: '0 auto 20px',
                    boxShadow: isActive ? '0 0 40px rgba(255,255,255,0.18), 0 0 0 6px rgba(255,255,255,0.05)' : isHov ? '0 0 24px rgba(255,255,255,0.1)' : 'none',
                  }}>
                    <Icon size={32} color={isActive ? '#ffffff' : isPast ? '#cccccc' : color} strokeWidth={isActive ? 2 : 1.5} />
                  </div>
                  {/* Step label */}
                  <div style={{ fontSize: 11, color: isActive ? '#ffffff' : '#666666', fontWeight: 700, letterSpacing: '0.16em', marginBottom: 8, fontFamily: "'Geist',sans-serif", transition: 'color 0.3s' }}>
                    STEP {step}
                  </div>
                  {/* Title */}
                  <div style={{ fontFamily: "'Newsreader',serif", fontSize: 17, fontWeight: 700, color: isActive ? '#ffffff' : isPast ? '#cccccc' : '#888888', marginBottom: 6, transition: 'color 0.3s' }}>
                    {title}
                  </div>
                  {/* Active dot indicator */}
                  {isActive && (
                    <div style={{ width: 6, height: 6, borderRadius: '50%', background: '#ffffff', margin: '8px auto 0', boxShadow: '0 0 8px #ffffff' }} />
                  )}
                </div>
              )
            })}
          </div>

          {/* Detail panel */}
          <div key={active} className="wf-detail" style={{
            marginTop: 52,
            padding: '36px 40px',
            borderRadius: 20,
            background: 'rgba(255,255,255,0.03)',
            border: '1px solid rgba(255,255,255,0.1)',
            boxShadow: '0 0 60px rgba(255,255,255,0.04)',
            display: 'grid', gridTemplateColumns: '1fr auto', gap: 40, alignItems: 'start',
          }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 18 }}>
                {(() => { const { Icon, color } = STEPS[active]; return <Icon size={22} color={color} strokeWidth={1.5} /> })()}
                <span style={{ fontFamily: "'Newsreader',serif", fontSize: 22, fontWeight: 700, color: '#ffffff' }}>
                  {STEPS[active].title}
                </span>
                <span style={{ fontSize: 11, color: '#666666', fontFamily: "'Geist',sans-serif", letterSpacing: '0.12em', fontWeight: 600, background: 'rgba(255,255,255,0.06)', padding: '3px 10px', borderRadius: 6 }}>
                  STEP {STEPS[active].step}
                </span>
              </div>
              <p style={{ fontSize: 14, color: 'rgba(200,200,200,0.75)', lineHeight: 1.95, fontFamily: "'Geist Mono',monospace", maxWidth: 640, margin: 0 }}>
                {STEPS[active].detail}
              </p>
            </div>
            {/* Tags + nav */}
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 16 }}>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'flex-end', maxWidth: 260 }}>
                {STEPS[active].tags.map(tag => (
                  <span key={tag} className="wf-tag" style={{
                    fontSize: 11, fontFamily: "'Geist',sans-serif", fontWeight: 600,
                    letterSpacing: '0.06em', padding: '5px 12px', borderRadius: 999,
                    background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.12)',
                    color: '#aaaaaa', cursor: 'default',
                  }}>{tag}</span>
                ))}
              </div>
              {/* Prev / Next buttons */}
              <div style={{ display: 'flex', gap: 8 }}>
                <button
                  onClick={() => setActive(a => Math.max(0, a - 1))}
                  disabled={active === 0}
                  style={{ padding: '8px 18px', borderRadius: 8, background: 'transparent', border: '1px solid rgba(255,255,255,0.12)', color: active === 0 ? '#333333' : '#aaaaaa', fontFamily: "'Geist',sans-serif", fontSize: 13, cursor: active === 0 ? 'default' : 'pointer', transition: 'all 0.2s' }}
                  onMouseEnter={e => { if (active > 0) { e.currentTarget.style.borderColor = 'rgba(255,255,255,0.4)'; e.currentTarget.style.color = '#ffffff' } }}
                  onMouseLeave={e => { e.currentTarget.style.borderColor = 'rgba(255,255,255,0.12)'; e.currentTarget.style.color = active === 0 ? '#333333' : '#aaaaaa' }}
                >← Prev</button>
                <button
                  onClick={() => setActive(a => Math.min(3, a + 1))}
                  disabled={active === 3}
                  style={{ padding: '8px 18px', borderRadius: 8, background: active === 3 ? 'transparent' : 'rgba(255,255,255,0.08)', border: '1px solid rgba(255,255,255,0.2)', color: active === 3 ? '#333333' : '#ffffff', fontFamily: "'Geist',sans-serif", fontSize: 13, cursor: active === 3 ? 'default' : 'pointer', transition: 'all 0.2s' }}
                  onMouseEnter={e => { if (active < 3) { e.currentTarget.style.background = 'rgba(255,255,255,0.14)' } }}
                  onMouseLeave={e => { e.currentTarget.style.background = active === 3 ? 'transparent' : 'rgba(255,255,255,0.08)' }}
                >Next →</button>
              </div>
            </div>
          </div>

        </div>
      </FadeIn>
    </section>
  )
}

// ═══════════════════════════════════════════════════════════════════════════
export default function LandingPage({ onEnterDashboard }) {
  // Force black background + suppress dashboard styles while landing page is shown
  useEffect(() => {
    document.documentElement.classList.add('dark', 'landing')
    document.documentElement.setAttribute('data-theme', 'dark')
    document.body.classList.add('landing')
    document.getElementById('root')?.classList.add('landing')
    document.body.style.background = '#090909'
    document.documentElement.style.background = '#090909'
    return () => {
      document.documentElement.classList.remove('dark', 'landing')
      document.documentElement.removeAttribute('data-theme')
      document.body.classList.remove('landing')
      document.getElementById('root')?.classList.remove('landing')
      document.body.style.background = ''
      document.documentElement.style.background = ''
    }
  }, [])

  const features = [
    { Icon: Zap, title: 'Early Exit Prediction', desc: 'Classify mission success/failure using just 20–60% of trajectory telemetry — saving up to 80% of simulation compute.', color: '#e2e2e2', glow: 'rgba(226,226,226,0.25)' },
    { Icon: Globe2, title: '3-Body RK4 Physics', desc: 'Full N-body gravitational simulation covering Earth–Moon, Mars, and Jupiter transfers with 4th-order Runge-Kutta integration.', color: '#aaaaaa', glow: 'rgba(170,170,170,0.25)' },
    { Icon: BrainCircuit, title: 'LSTM + Transformer', desc: 'Bidirectional LSTM and Transformer architectures trained on 850M+ trajectory rows across 80,000 Monte Carlo missions (8 planets).', color: '#888888', glow: 'rgba(136,136,136,0.25)' },
    { Icon: Ruler, title: '13 Physics Features', desc: 'Synodic frame coordinates, specific orbital energy, eccentricity, flight path angle — invariant across planetary systems.', color: '#666666', glow: 'rgba(102,102,102,0.22)' },
    { Icon: Target, title: 'AUC 0.984 Accuracy', desc: 'Transformer calibrated to AUC 0.984, F1 0.921 after threshold tuning. XGBoost baseline hits F1 0.992 ± 0.001 (5-seed CI).', color: '#e2e2e2', glow: 'rgba(226,226,226,0.25)' },
    { Icon: Satellite, title: 'NASA GMAT Integration', desc: "Interfaces directly with NASA's General Mission Analysis Tool for seamless end-to-end prediction pipelines.", color: '#aaaaaa', glow: 'rgba(170,170,170,0.25)' },
  ]

  return (
    <div className="lg-root" style={{ background: '#090909', color: '#e2e8f0', minHeight: '100vh', overflowX: 'hidden' }}>
      <FontLink />

      {/* ── Fixed Nav ─────────────────────────────────────────── */}
      <nav style={{ position: 'fixed', top: 0, left: 0, right: 0, zIndex: 999, display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 5vw', height: '80px', background: 'rgba(25,25,25,0.8)', backdropFilter: 'blur(32px)', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
        <div className="font-serif" style={{ fontSize: '24px', fontWeight: 600, color: '#FFFFFF', letterSpacing: '-0.02em', fontFamily: "'Newsreader', serif" }}>
          OrbitGuard.
        </div>
        <ul style={{ display: 'flex', gap: '48px', listStyle: 'none', fontFamily: "'Geist', sans-serif" }} className="hidden md:flex">
          {['Capabilities', 'Metrics', 'Research', 'Partners'].map(l => (
            <li key={l}><a href={`#${l.toLowerCase()}`} style={{ color: 'rgba(255,255,255,0.5)', textDecoration: 'none', fontSize: '14px', transition: 'color 0.2s' }} onMouseEnter={e => e.currentTarget.style.color = '#FFFFFF'} onMouseLeave={e => e.currentTarget.style.color = 'rgba(255,255,255,0.5)'}>{l}</a></li>
          ))}
        </ul>
        <button onClick={onEnterDashboard} style={{ display: 'inline-flex', alignItems: 'center', gap: '12px', background: '#FFFFFF', color: '#111111', padding: '10px 20px', borderRadius: '9999px', fontFamily: "'Geist', sans-serif", fontWeight: 600, fontSize: '14px', cursor: 'pointer', border: 'none', transition: 'transform 0.2s' }}
          onMouseEnter={e => e.currentTarget.style.transform = 'scale(1.05)'}
          onMouseLeave={e => e.currentTarget.style.transform = 'scale(1)'}>
          Mission Control
          <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: '24px', height: '24px', borderRadius: '50%', background: 'rgba(0,0,0,0.1)' }}>
            <ArrowUpRight size={14} strokeWidth={2} color="#111111" />
          </span>
        </button>
      </nav>

      {/* ── Hero ─────────────────────────────────────────────── */}
      <section style={{ paddingTop: 72, minHeight: '100vh', display: 'flex', flexDirection: 'column', justifyContent: 'center', position: 'relative', overflow: 'hidden' }}>
        <FlowFieldBackground color="#ffffff" particleCount={110} speed={0.8} trailOpacity={0.3} />
        {/* subtle radial glows */}
        <div style={{ pointerEvents: 'none', position: 'absolute', inset: 0, background: 'radial-gradient(ellipse 70% 60% at 70% 40%, rgba(226,226,226,0.09), transparent 70%)', zIndex: 0 }} />
        <div style={{ pointerEvents: 'none', position: 'absolute', inset: 0, background: 'radial-gradient(ellipse 50% 40% at 20% 60%, rgba(170,170,170,0.06), transparent 70%)', zIndex: 0 }} />

        <div style={{ maxWidth: 1280, margin: '0 auto', width: '100%', padding: '60px 5vw', display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(340px,1fr))', gap: 64, alignItems: 'center', position: 'relative', zIndex: 1 }}>
          {/* Left: text */}
          <div>
            <div style={{ fontSize: 11, letterSpacing: '0.4em', color: '#e2e2e2', textTransform: 'uppercase', fontWeight: 600, marginBottom: 20, opacity: 0.9 }}>
              NASA GMAT · AI Mission Prediction
            </div>
            <h1 className="lg-heading" style={{ fontSize: 'clamp(64px, 8vw, 110px)', fontWeight: 900, letterSpacing: '-0.03em', lineHeight: 0.92, color: '#ffffff', marginBottom: 28 }}>
              Orbit<br /><em style={{ fontStyle: 'italic', color: 'rgba(210,210,210,0.85)' }}>Guard</em>
            </h1>
            <p style={{ fontSize: 15, color: 'rgba(200,200,200,0.72)', lineHeight: 1.95, marginBottom: 44, maxWidth: 480, fontFamily: "'Geist Mono',monospace", fontWeight: 400, letterSpacing: '-0.01em' }}>
              Predict spacecraft mission failure before it happens. A bidirectional LSTM trained on 85.3M trajectory points — classifying success from just 20% of telemetry.
            </p>

            {/* Metric pills */}
            <div style={{ display: 'flex', gap: 0, marginBottom: 44, border: '1px solid rgba(255,255,255,0.08)', borderRadius: 16, overflow: 'hidden', width: 'fit-content' }}>
              {[
                { val: '0.984', label: 'AUC Score', color: '#e2e2e2' },
                { val: '80K', label: 'Missions', color: '#aaaaaa' },
                { val: '80%', label: 'Compute Saved', color: '#888888' },
              ].map(({ val, label, color }, i) => (
                <div key={label} style={{ padding: '16px 28px', borderRight: i < 2 ? '1px solid rgba(255,255,255,0.08)' : 'none', background: 'rgba(255,255,255,0.025)' }}>
                  <div className="lg-heading" style={{ fontSize: 26, fontWeight: 800, color, lineHeight: 1 }}>{val}</div>
                  <div style={{ fontSize: 10, color: '#475569', letterSpacing: '0.1em', textTransform: 'uppercase', marginTop: 4 }}>{label}</div>
                </div>
              ))}
            </div>

            {/* CTAs */}
            <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap' }}>
              <button onClick={onEnterDashboard}
                style={{ padding: '15px 36px', borderRadius: 50, background: 'linear-gradient(135deg,#ffffff,#cccccc)', color: '#0a0a0a', border: 'none', cursor: 'pointer', fontFamily: "'Geist',sans-serif", fontSize: 15, fontWeight: 700, boxShadow: '0 0 32px rgba(226,226,226,0.4)', transition: 'transform 0.2s,box-shadow 0.2s', letterSpacing: '0.02em' }}
                onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-2px)'; e.currentTarget.style.boxShadow = '0 0 48px rgba(226,226,226,0.6)' }}
                onMouseLeave={e => { e.currentTarget.style.transform = ''; e.currentTarget.style.boxShadow = '0 0 32px rgba(226,226,226,0.4)' }}>
                Open Mission Control →
              </button>
              <button style={{ padding: '15px 36px', borderRadius: 50, background: 'transparent', color: '#94a3b8', border: '1px solid rgba(255,255,255,0.14)', cursor: 'pointer', fontFamily: "'Geist',sans-serif", fontSize: 15, fontWeight: 500, transition: 'border-color 0.2s,color 0.2s' }}
                onMouseEnter={e => { e.currentTarget.style.borderColor = 'rgba(226,226,226,0.4)'; e.currentTarget.style.color = '#e2e2e2' }}
                onMouseLeave={e => { e.currentTarget.style.borderColor = 'rgba(255,255,255,0.14)'; e.currentTarget.style.color = '#94a3b8' }}>
                Read the Paper
              </button>
            </div>
          </div>

          {/* Right: Globe */}
          <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <Globe size={440} />
          </div>
        </div>
      </section>

      {/* ── Ready for Launch CTA — right under hero ───────────── */}
      <section style={{ padding: '80px 5vw', background: 'linear-gradient(180deg, rgba(226,226,226,0.04) 0%, transparent 100%)', borderTop: '1px solid rgba(226,226,226,0.08)' }}>
        <FadeIn>
          <div style={{ maxWidth: 860, margin: '0 auto', textAlign: 'center', padding: '64px 48px', borderRadius: 28, background: 'linear-gradient(135deg,rgba(226,226,226,0.07),rgba(170,170,170,0.05),rgba(124,58,237,0.07))', border: '1px solid rgba(226,226,226,0.18)', boxShadow: '0 0 80px rgba(226,226,226,0.07)' }}>
            <div className="lg-heading" style={{ fontSize: 12, fontWeight: 600, color: '#e2e2e2', letterSpacing: '0.2em', marginBottom: 18, textTransform: 'uppercase' }}>Ready for Launch</div>
            <h2 className="lg-heading" style={{ fontSize: 'clamp(30px,4vw,50px)', fontWeight: 800, lineHeight: 1.15, marginBottom: 18, color: '#f1f5f9' }}>
              Stop waiting for simulations.<br />
              <span className="clip-grad">Predict the outcome now.</span>
            </h2>
            <p style={{ fontSize: 14, color: 'rgba(180,180,180,0.62)', lineHeight: 1.95, marginBottom: 36, maxWidth: 480, margin: '0 auto 36px', fontFamily: "'Geist Mono',monospace" }}>
              Open Mission Control and run a live trajectory prediction with your own CSV data — no setup required.
            </p>
            <div style={{ display: 'flex', gap: 14, justifyContent: 'center', flexWrap: 'wrap' }}>
              <button onClick={onEnterDashboard} style={{ padding: '16px 40px', borderRadius: 50, background: 'linear-gradient(135deg,#ffffff,#cccccc)', color: '#0a0a0a', border: 'none', cursor: 'pointer', fontFamily: "'Geist',sans-serif", fontSize: 16, fontWeight: 700, boxShadow: '0 0 32px rgba(226,226,226,0.4)', transition: 'transform 0.2s,box-shadow 0.2s' }}
                onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-2px)'; e.currentTarget.style.boxShadow = '0 0 48px rgba(226,226,226,0.6)' }}
                onMouseLeave={e => { e.currentTarget.style.transform = ''; e.currentTarget.style.boxShadow = '0 0 32px rgba(226,226,226,0.4)' }}>
                Open Mission Control →
              </button>
              <button style={{ padding: '16px 40px', borderRadius: 50, background: 'transparent', color: '#aaaaaa', border: '2px solid rgba(170,170,170,0.5)', cursor: 'pointer', fontFamily: "'Geist',sans-serif", fontSize: 16, fontWeight: 600, transition: 'background 0.2s,transform 0.2s,border-color 0.2s' }}
                onMouseEnter={e => { e.currentTarget.style.background = 'rgba(170,170,170,0.08)'; e.currentTarget.style.transform = 'translateY(-2px)'; e.currentTarget.style.borderColor = '#aaaaaa' }}
                onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.transform = ''; e.currentTarget.style.borderColor = 'rgba(170,170,170,0.5)' }}>
                Read the Paper
              </button>
            </div>
          </div>
        </FadeIn>
      </section>

      {/* ── Partner logos ─────────────────────────────────────── */}
      <section id="partners" style={{ padding: '56px 0', borderTop: '1px solid rgba(255,255,255,0.05)', borderBottom: '1px solid rgba(255,255,255,0.05)', background: 'rgba(0,0,0,0.25)' }}>
        <div style={{ textAlign: 'center', marginBottom: 36, padding: '0 5vw' }}>
          <div style={{ fontSize: 11, fontWeight: 500, color: 'rgba(148,163,184,0.45)', letterSpacing: '0.22em', textTransform: 'uppercase', fontFamily: "'Geist',sans-serif" }}>
            Trusted by institutions pushing the boundaries of space exploration
          </div>
        </div>
        <LogoCarousel />
      </section>

      {/* ── Features ──────────────────────────────────────────── */}
      <section id="features" style={{ padding: '110px 5vw' }}>
        <FadeIn>
          <div style={{ maxWidth: 1200, margin: '0 auto' }}>
            <div style={{ textAlign: 'center', marginBottom: 72 }}>
              <div className="lg-heading" style={{ fontSize: 13, fontWeight: 600, color: '#e2e2e2', letterSpacing: '0.15em', marginBottom: 14, textTransform: 'uppercase' }}>Capabilities</div>
              <h2 className="lg-heading" style={{ fontSize: 'clamp(34px,4.5vw,58px)', fontWeight: 700, lineHeight: 1.1, color: '#ffffff' }}>Built for aerospace-grade <br /><span className="clip-grad">prediction at scale</span></h2>
              <div style={{ width: 60, height: 3, borderRadius: 2, background: 'linear-gradient(90deg,#e2e2e2,#aaaaaa)', margin: '16px auto 0' }} />
            </div>
            <div className="feat-grid-3" style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 20 }}>
              {features.map((f, i) => (
                <div key={i} className="feat-card">
                  {/* Icon container */}
                  <div style={{ marginBottom: 24 }}>
                    <div style={{
                      width: 52, height: 52, borderRadius: 16,
                      background: 'rgba(255,255,255,0.04)',
                      border: `1px solid ${f.color}33`,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      boxShadow: `0 0 18px ${f.glow}`,
                    }}>
                      <f.Icon size={24} color={f.color} strokeWidth={1.5} />
                    </div>
                  </div>
                  <h3 style={{ fontFamily: "'Newsreader', serif", fontSize: 20, fontWeight: 700, color: '#f1f5f9', marginBottom: 10, letterSpacing: '-0.02em', lineHeight: 1.2 }}>{f.title}</h3>
                  <p style={{ fontSize: 13, color: 'rgba(180,180,180,0.62)', lineHeight: 1.9, fontFamily: "'Geist Mono',monospace" }}>{f.desc}</p>
                  {/* Subtle bottom accent */}
                  <div style={{ position: 'absolute', bottom: 0, left: 24, right: 24, height: 1, background: `linear-gradient(90deg, transparent, ${f.color}22, transparent)` }} />
                </div>
              ))}
            </div>
          </div>
        </FadeIn>
      </section>

      {/* ── Stats — between Capabilities and Workflow ─────────── */}
      <section id="metrics" style={{ padding: '90px 5vw', background: 'rgba(226,226,226,0.03)', borderTop: '1px solid rgba(226,226,226,0.08)', borderBottom: '1px solid rgba(226,226,226,0.08)' }}>
        <FadeIn>
          <div style={{ maxWidth: 1100, margin: '0 auto', display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(200px,1fr))', gap: 48, textAlign: 'center' }}>
            {[
              { end: 10000, suffix: '+', label: 'Missions Simulated', sub: 'Monte Carlo RK4' },
              { end: 85.3, suffix: 'M', decimals: 1, label: 'Training Data Points', sub: 'Parquet compressed' },
              { end: 0.984, suffix: ' AUC', decimals: 3, label: 'Transformer (calibrated)', sub: 'Multi-planet, 80K missions' },
              { end: 80, suffix: '%', label: 'Compute Saved', sub: 'Early exit prediction' },
            ].map(({ end, suffix, prefix, decimals, label, sub }) => (
              <div key={label}>
                <Counter end={end} suffix={suffix} prefix={prefix} decimals={decimals || 0} />
                <div className="lg-heading" style={{ fontSize: 17, fontWeight: 600, color: '#cbd5e1', marginTop: 10 }}>{label}</div>
                <div style={{ fontSize: 12, color: '#e2e2e2', marginTop: 4, opacity: 0.6 }}>{sub}</div>
              </div>
            ))}
          </div>
        </FadeIn>
      </section>

      {/* ── How It Works (interactive) ────────────────────────── */}
      <WorkflowSection />

      {/* ── FAQ ───────────────────────────────────────────────── */}
      <FaqSection />

      {/* ── CTA ───────────────────────────────────────────────── */}
      <section style={{ padding: '160px 4vw', textAlign: 'center' }}>
        <FadeIn>
          <div style={{ display: 'inline-block', textAlign: 'left', maxWidth: '800px', background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.04)', padding: '6px', borderRadius: '24px' }}>
            <div style={{ padding: '80px', display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center', background: '#FFFFFF', border: '1px solid #EAEAEA', borderRadius: '18px', boxShadow: '0 4px 24px rgba(0,0,0,0.02)' }}>
              <div style={{ fontFamily: "'Geist Mono', monospace", color: '#787774', fontSize: '12px', letterSpacing: '0.1em', marginBottom: '24px', textTransform: 'uppercase' }}>Initialize Protocol</div>
              <h2 style={{ fontFamily: "'Newsreader', serif", fontSize: 'clamp(32px,4vw,56px)', color: '#111111', marginBottom: '32px', letterSpacing: '-0.02em', lineHeight: 1.1 }}>Avoid simulation delays.<br />Predict outcomes immediately.</h2>
              <button onClick={onEnterDashboard} className="group" style={{ display: 'inline-flex', alignItems: 'center', gap: '12px', background: '#111111', color: '#FFFFFF', padding: '12px 24px', borderRadius: '9999px', fontFamily: "'Geist', sans-serif", fontWeight: 500, fontSize: '15px', cursor: 'pointer', border: 'none', transition: 'transform 0.2s' }}
                onMouseEnter={e => e.currentTarget.style.transform = 'scale(1.03)'}
                onMouseLeave={e => e.currentTarget.style.transform = 'scale(1)'}
              >
                Open Mission Control
                <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: '28px', height: '28px', borderRadius: '50%', background: 'rgba(255,255,255,0.15)' }}>
                  <ArrowUpRight size={14} strokeWidth={2} />
                </span>
              </button>
            </div>
          </div>
        </FadeIn>
      </section>

      {/* ── Footer ────────────────────────────────────────────── */}
      <footer style={{ borderTop: '1px solid rgba(255,255,255,0.07)', padding: '72px 5vw 40px' }}>
        <div className="footer-grid" style={{ maxWidth: 1200, margin: '0 auto', display: 'grid', gridTemplateColumns: '2fr 1fr 1fr 1fr', gap: 48, marginBottom: 56 }}>
          <div>
            <div className="lg-heading clip-grad" style={{ fontSize: 22, fontWeight: 800, marginBottom: 16 }}>ORBITGUARD</div>
            <p style={{ fontSize: 12, color: 'rgba(160,160,160,0.85)', lineHeight: 1.85, maxWidth: 300, fontFamily: "'Geist Mono',monospace" }}>AI-powered spacecraft mission failure prediction. Built on NASA GMAT physics with LSTM + Transformer deep learning.</p>
            <div style={{ display: 'flex', gap: 12, marginTop: 24 }}>
              {['GitHub', 'arXiv', 'Twitter'].map(s => (
                <a key={s} href="#" className="lg-heading" style={{ padding: '7px 14px', borderRadius: 8, background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)', fontSize: 12, color: '#888888', textDecoration: 'none', fontWeight: 600, transition: 'all 0.2s' }}
                  onMouseEnter={e => { e.currentTarget.style.color = '#e2e2e2'; e.currentTarget.style.borderColor = 'rgba(226,226,226,0.3)' }}
                  onMouseLeave={e => { e.currentTarget.style.color = '#888888'; e.currentTarget.style.borderColor = 'rgba(255,255,255,0.08)' }}>
                  {s}
                </a>
              ))}
            </div>
          </div>
          {[
            { h: 'Product', links: ['Mission Control', 'API Docs', 'Changelog', 'Status Page'] },
            { h: 'Research', links: ['Ablation Study', 'Dataset', 'Publications', 'Open Source'] },
            { h: 'Company', links: ['About', 'Careers', 'Blog', 'Contact'] },
          ].map(({ h, links }) => (
            <div key={h}>
              <div className="lg-heading" style={{ fontSize: 13, fontWeight: 700, color: '#e2e8f0', letterSpacing: '0.06em', marginBottom: 18, textTransform: 'uppercase' }}>{h}</div>
              {links.map(l => <div key={l}><a href="#" className="footer-link">{l}</a></div>)}
            </div>
          ))}
        </div>
        <div style={{ maxWidth: 1200, margin: '0 auto', paddingTop: 28, borderTop: '1px solid rgba(255,255,255,0.05)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 16 }}>
          <div style={{ fontSize: 13, color: '#666666' }}>© 2026 OrbitGuard. Built on NASA GMAT research. All rights reserved.</div>
          <div style={{ display: 'flex', gap: 24 }}>
            {['Privacy', 'Terms', 'Security'].map(l => <a key={l} href="#" className="footer-link">{l}</a>)}
          </div>
        </div>
      </footer>
    </div>
  )
}
