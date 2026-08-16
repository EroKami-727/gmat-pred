import { useState, useEffect, lazy, Suspense } from 'react'
import LandingPage from './LandingPage'
import { API } from './lib/api'

// Panels are lazy so the landing page — which is what a first-time visitor
// actually sees — does not pull Recharts and all seven dashboards down with it.
// The dashboard is behind a click, and only one panel is mounted at a time.
const Overview    = lazy(() => import('./panels/Overview'))
const Training    = lazy(() => import('./panels/Training'))
const Ablation    = lazy(() => import('./panels/Ablation'))
const Dataset     = lazy(() => import('./panels/Dataset'))
const Simulator   = lazy(() => import('./panels/Simulator'))
const Experiments = lazy(() => import('./panels/Experiments'))
const Report      = lazy(() => import('./panels/Report'))

const TABS = ['REPORT', 'OVERVIEW', 'SIMULATOR', 'TRAINING', 'ABLATION', 'EXPERIMENTS', 'DATASET']
const PANELS = { REPORT: Report, OVERVIEW: Overview, SIMULATOR: Simulator, TRAINING: Training, ABLATION: Ablation, EXPERIMENTS: Experiments, DATASET: Dataset }

// Header status lamps.
//
// These three read-outs used to be hardcoded strings — "ALLOCATED // 98%",
// "MOUNTED // SECURE", "IDLE_READY" — rendered on every screen of the dashboard
// regardless of what the backend was doing, or whether there was a backend at
// all. A GPU utilisation figure that is a string literal is worse than no
// figure: it is the one number an operator would trust at a glance.
//
// They now poll /api/system and degrade honestly: unreachable backend shows
// OFFLINE in red rather than three reassuring green lamps.
const OFFLINE = [
  { dot: 'var(--red)', label: 'SYS_GPU_01',   value: 'OFFLINE' },
  { dot: 'var(--red)', label: 'DATASET_VOL',  value: 'UNREACHABLE' },
  { dot: 'var(--red)', label: 'MODEL_STATUS', value: 'NO_BACKEND' },
]

function useSystemStatus(pollMs = 5000) {
  const [lamps, setLamps] = useState(OFFLINE)

  useEffect(() => {
    let cancelled = false

    const read = async () => {
      try {
        const r = await fetch(`${API}/api/system`)
        if (!r.ok) throw new Error(String(r.status))
        const s = await r.json()
        if (cancelled) return

        const gpu = s.device === 'cuda'
          ? { dot: 'var(--green)', label: 'SYS_GPU_01',
              value: `${s.gpu_mem_used_gb ?? 0} / ${s.gpu_mem_total_gb ?? '?'} GB // ${s.gpu_util_pct ?? 0}%` }
          : { dot: '#ffaa00', label: 'SYS_GPU_01', value: 'CPU ONLY // NO CUDA' }

        const vol = s.dataset_mounted
          ? { dot: 'var(--green)', label: 'DATASET_VOL',
              value: `MOUNTED // ${s.dataset_size_gb ?? '?'} GB` }
          : { dot: 'var(--red)', label: 'DATASET_VOL', value: 'NOT MOUNTED' }

        const jobs = s.active_training_jobs?.length ?? 0
        const nPlanets = s.planets_loaded?.length ?? 0
        const model = jobs > 0
          ? { dot: '#ffaa00', label: 'MODEL_STATUS', value: `TRAINING // ${jobs} JOB${jobs > 1 ? 'S' : ''}` }
          : nPlanets > 0
            ? { dot: 'var(--green)', label: 'MODEL_STATUS', value: `READY // ${nPlanets} PLANETS` }
            : { dot: 'var(--red)', label: 'MODEL_STATUS', value: 'NO MODEL LOADED' }

        setLamps([gpu, vol, model])
      } catch {
        if (!cancelled) setLamps(OFFLINE)
      }
    }

    read()
    const id = setInterval(read, pollMs)
    return () => { cancelled = true; clearInterval(id) }
  }, [pollMs])

  return lamps
}

function Header({ onBack }) {
  const lamps = useSystemStatus()
  return (
    <header style={{
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      padding: '0 28px', height: '62px',
      background: 'var(--bg1)', borderBottom: '1px solid var(--border)',
      borderLeft: '3px solid var(--cyan)',
      position: 'sticky', top: 0, zIndex: 100, flexShrink: 0,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        <div style={{
          width: '5px', height: '34px',
          background: 'linear-gradient(180deg, #ffffff, #555555)',
          boxShadow: '0 0 14px rgba(226,226,226,0.4)',
          flexShrink: 0,
        }} />
        <div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '10px' }}>
            <span style={{ fontSize: '19px', color: 'var(--cyan)', letterSpacing: '0.14em', fontWeight: 700, textShadow: '0 0 24px rgba(226,226,226,0.3)' }}>
              ORBITGUARD
            </span>
            <span style={{ fontSize: '12px', color: 'var(--text-dim)', letterSpacing: '0.06em' }}>v4.0.9-rc</span>
          </div>
          <div style={{ fontSize: '11px', color: 'var(--text-dim)', letterSpacing: '0.08em', marginTop: '3px' }}>
            NASA GMAT EARLY EXIT PRESCREENING SYSTEM
          </div>
        </div>
      </div>

      <div style={{ display: 'flex' }}>
        {lamps.map(({ dot, label, value }) => (
          <div key={label} style={{ padding: '0 22px', borderLeft: '1px solid var(--border)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', color: 'var(--text-dim)', letterSpacing: '0.06em' }}>
              <span style={{ width: '7px', height: '7px', borderRadius: '50%', background: dot, boxShadow: `0 0 6px ${dot}`, animation: 'pulse 2s ease-in-out infinite', display: 'inline-block' }} />
              {label}
            </div>
            <div style={{ fontSize: '12px', color: 'var(--cyan)', letterSpacing: '0.04em', marginTop: '3px' }}>{value}</div>
          </div>
        ))}
        <div style={{ borderLeft: '1px solid var(--border)' }} />
        <button
          onClick={onBack}
          style={{
            marginLeft: '18px',
            padding: '7px 18px',
            background: 'transparent',
            border: '1px solid var(--border2)',
            color: 'var(--text-dim)',
            fontFamily: 'var(--mono)',
            fontSize: '11px',
            letterSpacing: '0.1em',
            textTransform: 'uppercase',
            cursor: 'pointer',
            transition: 'border-color 0.2s, color 0.2s',
          }}
          onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--cyan)'; e.currentTarget.style.color = 'var(--cyan)' }}
          onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border2)'; e.currentTarget.style.color = 'var(--text-dim)' }}
        >
          ← Landing
        </button>
      </div>

      <style>{`
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
      `}</style>
    </header>
  )
}

function TabBar({ active, setActive }) {
  return (
    <nav style={{
      display: 'flex', background: 'var(--bg1)',
      borderBottom: '1px solid var(--border)',
      padding: '0 28px', flexShrink: 0,
    }}>
      {TABS.map(tab => (
        <button
          key={tab}
          onClick={() => setActive(tab)}
          style={{
            padding: '14px 24px',
            fontSize: '12px',
            letterSpacing: '0.1em',
            color: active === tab ? 'var(--cyan)' : 'var(--text-dim)',
            background: 'none',
            border: 'none',
            borderBottom: active === tab ? '2px solid var(--cyan)' : '2px solid transparent',
            cursor: 'pointer',
            fontFamily: 'var(--mono)',
            textTransform: 'uppercase',
            transition: 'color 0.15s',
            textShadow: active === tab ? '0 0 14px rgba(226,226,226,0.4)' : 'none',
          }}
        >
          {tab}
        </button>
      ))}
    </nav>
  )
}

function PanelLoading() {
  return (
    <div style={{
      padding: '48px 28px', color: 'var(--text-dim)', fontFamily: 'var(--mono)',
      fontSize: '12px', letterSpacing: '0.1em',
    }}>
      LOADING PANEL…
    </div>
  )
}

export default function App() {
  const [showDashboard, setShowDashboard] = useState(false)
  const [active, setActive] = useState('REPORT')
  const Panel = PANELS[active]

  if (!showDashboard) {
    return <LandingPage onEnterDashboard={() => setShowDashboard(true)} />
  }

  return (
    <div className="app">
      <Header onBack={() => setShowDashboard(false)} />
      <TabBar active={active} setActive={setActive} />
      <main style={{ flex: 1, overflowY: 'auto' }}>
        <Suspense fallback={<PanelLoading />}>
          <Panel />
        </Suspense>
      </main>
    </div>
  )
}
