import { useState } from 'react'
import LandingPage from './LandingPage'
import Overview from './panels/Overview'
import Training from './panels/Training'
import Ablation from './panels/Ablation'
import Dataset from './panels/Dataset'

const TABS = ['OVERVIEW', 'TRAINING', 'ABLATION', 'DATASET']
const PANELS = { OVERVIEW: Overview, TRAINING: Training, ABLATION: Ablation, DATASET: Dataset }

function Header({ onBack }) {
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
        {[
          { dot: '#ffaa00', label: 'SYS_GPU_01', value: 'ALLOCATED // 98%' },
          { dot: 'var(--cyan)', label: 'DATASET_VOL', value: 'MOUNTED // SECURE' },
          { dot: 'var(--green)', label: 'MODEL_STATUS', value: 'IDLE_READY' },
        ].map(({ dot, label, value }) => (
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

export default function App() {
  const [showDashboard, setShowDashboard] = useState(false)
  const [active, setActive] = useState('OVERVIEW')
  const Panel = PANELS[active]

  if (!showDashboard) {
    return <LandingPage onEnterDashboard={() => setShowDashboard(true)} />
  }

  return (
    <div className="app">
      <Header onBack={() => setShowDashboard(false)} />
      <TabBar active={active} setActive={setActive} />
      <main style={{ flex: 1, overflowY: 'auto' }}>
        <Panel />
      </main>
    </div>
  )
}
