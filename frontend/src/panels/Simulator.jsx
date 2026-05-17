import { useEffect, useRef, useState, useCallback } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts'
import { API } from '../lib/api.js'

// Mission card status
const STATUS = {
  IDLE:      'IDLE',
  RUNNING:   'RUNNING',
  CANCELED:  'CANCELED',
  COMPLETED: 'COMPLETED',
  ERROR:     'ERROR',
}

const STATUS_COLOR = {
  IDLE:      'var(--text-dim)',
  RUNNING:   'var(--cyan)',
  CANCELED:  'var(--red)',
  COMPLETED: 'var(--green)',
  ERROR:     'var(--orange)',
}

function MissionCard({ mission, onStream, isActive }) {
  const { mission_id, label, failure_type, status, steps, probability, canceled, wasCorrect } = mission

  const lastProb = steps.length ? steps[steps.length - 1].probability : null
  const elapsed  = steps.length ? `${(steps[steps.length - 1].elapsed_pct * 100).toFixed(0)}%` : '0%'

  const chartData = steps.slice(-60)  // last 60 points for perf

  return (
    <div style={{
      background: 'var(--bg2)',
      border: `1px solid ${status === STATUS.CANCELED ? 'var(--red)' : status === STATUS.COMPLETED ? 'var(--green)' : 'var(--border)'}`,
      borderTop: `2px solid ${STATUS_COLOR[status]}`,
      padding: '12px',
      display: 'flex',
      flexDirection: 'column',
      gap: '8px',
      position: 'relative',
      transition: 'border-color 0.3s',
    }}>
      {/* Header row */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ fontSize: '11px', color: 'var(--text-dim)', fontFamily: 'var(--mono)', letterSpacing: '0.06em' }}>
          MID #{mission_id}
        </div>
        <div style={{ fontSize: '10px', color: STATUS_COLOR[status], letterSpacing: '0.08em', fontFamily: 'var(--mono)' }}>
          {status === STATUS.RUNNING ? '● ' : ''}{status}
        </div>
      </div>

      {/* Probability bar */}
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: 'var(--text-dim)', marginBottom: '4px', fontFamily: 'var(--mono)' }}>
          <span>P(FAIL)</span>
          <span style={{ color: lastProb != null ? (lastProb > 0.5 ? 'var(--red)' : 'var(--green)') : 'var(--text-dim)' }}>
            {lastProb != null ? lastProb.toFixed(3) : '—'}
          </span>
        </div>
        <div style={{ height: '4px', background: 'var(--bg3)', borderRadius: '2px' }}>
          <div style={{
            height: '100%',
            width: `${(lastProb || 0) * 100}%`,
            background: lastProb > 0.7 ? 'var(--red)' : lastProb > 0.4 ? 'var(--orange)' : 'var(--green)',
            borderRadius: '2px',
            transition: 'width 0.2s ease, background 0.2s ease',
          }} />
        </div>
      </div>

      {/* Mini sparkline */}
      <div style={{ height: '50px' }}>
        <ResponsiveContainer width="100%" height={50}>
          <LineChart data={chartData} margin={{ top: 2, right: 2, bottom: 2, left: 2 }}>
            <ReferenceLine y={0.5} stroke="#333333" strokeDasharray="3 2" />
            <Line
              type="monotone" dataKey="probability"
              stroke={status === STATUS.CANCELED ? 'var(--red)' : status === STATUS.COMPLETED ? 'var(--green)' : 'var(--cyan)'}
              strokeWidth={1.5} dot={false} isAnimationActive={false}
            />
            <YAxis domain={[0, 1]} hide />
            <XAxis dataKey="timestep" hide />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Footer */}
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: 'var(--text-dim)', fontFamily: 'var(--mono)' }}>
        <span>ELAPSED: {elapsed}</span>
        <span style={{ color: label === 1 ? 'var(--green)' : 'var(--red)' }}>
          TRUE: {label === 1 ? 'SUCCESS' : 'FAIL'}
        </span>
      </div>

      {/* Verdict overlay */}
      {(status === STATUS.CANCELED || status === STATUS.COMPLETED) && (
        <div style={{
          fontSize: '11px',
          fontFamily: 'var(--mono)',
          letterSpacing: '0.08em',
          padding: '4px 8px',
          background: 'var(--bg3)',
          border: `1px solid ${wasCorrect ? 'var(--green)' : 'var(--orange)'}`,
          color: wasCorrect ? 'var(--green)' : 'var(--orange)',
          textAlign: 'center',
        }}>
          {status === STATUS.CANCELED
            ? (wasCorrect ? '✓ CORRECTLY PRUNED' : '✗ FALSE POSITIVE')
            : (wasCorrect ? '✓ CORRECTLY PASSED' : '✗ MISSED FAILURE')}
        </div>
      )}

      {/* Run button (idle) */}
      {status === STATUS.IDLE && (
        <button
          onClick={() => onStream(mission_id)}
          style={{
            background: 'transparent',
            border: '1px solid var(--border2)',
            color: 'var(--text-dim)',
            fontSize: '10px',
            fontFamily: 'var(--mono)',
            letterSpacing: '0.08em',
            padding: '4px 0',
            cursor: 'pointer',
            transition: 'border-color 0.2s, color 0.2s',
          }}
          onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--cyan)'; e.currentTarget.style.color = 'var(--cyan)' }}
          onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border2)'; e.currentTarget.style.color = 'var(--text-dim)' }}
        >
          ▶ RUN
        </button>
      )}
    </div>
  )
}

export default function Simulator() {
  const [missions, setMissions]   = useState([])
  const [missionData, setMData]   = useState({})  // mission_id → {status, steps, ...}
  const [loading, setLoading]     = useState(false)
  const [apiOnline, setApiOnline] = useState(null)
  const [threshold, setThreshold]       = useState(0.5)
  const [minElapsed, setMinElapsed]     = useState(0.4)
  const [dataPath, setDataPath]         = useState('data/merged/missions.parquet')
  const [n, setN]                 = useState(12)
  const [runningAll, setRunningAll] = useState(false)
  const [stats, setStats]         = useState({ pruned: 0, passed: 0, correct: 0, total: 0 })

  const esRefs = useRef({})  // mission_id → EventSource

  useEffect(() => {
    fetch(`${API}/api/health`, { signal: AbortSignal.timeout(3000) })
      .then(r => r.json())
      .then(() => setApiOnline(true))
      .catch(() => setApiOnline(false))
  }, [])

  const loadMissions = async () => {
    setLoading(true)
    // Close any open streams
    Object.values(esRefs.current).forEach(es => es.close())
    esRefs.current = {}

    try {
      const r = await fetch(`${API}/api/simulator/missions?n=${n}&data=${encodeURIComponent(dataPath)}`)
      if (!r.ok) throw new Error(await r.text())
      const { missions: ms } = await r.json()
      setMissions(ms)
      const init = {}
      ms.forEach(m => {
        init[m.mission_id] = { status: STATUS.IDLE, steps: [], probability: null, canceled: false, wasCorrect: null }
      })
      setMData(init)
      setStats({ pruned: 0, passed: 0, correct: 0, total: 0 })
    } catch (err) {
      console.error('Failed to load missions:', err)
    } finally {
      setLoading(false)
    }
  }

  const streamMission = useCallback((mission_id) => {
    // Already streaming?
    if (esRefs.current[mission_id]) return

    setMData(prev => ({
      ...prev,
      [mission_id]: { ...prev[mission_id], status: STATUS.RUNNING, steps: [] }
    }))

    const url = `${API}/api/simulator/stream?mission_id=${mission_id}&data=${encodeURIComponent(dataPath)}&threshold=${threshold}&min_elapsed_pct=${minElapsed}&step_delay_ms=60`
    const es = new EventSource(url)
    esRefs.current[mission_id] = es

    es.onmessage = (e) => {
      let event
      try { event = JSON.parse(e.data) } catch { return }

      if (event.type === 'info') {
        // Header received
      }

      if (event.type === 'step') {
        setMData(prev => ({
          ...prev,
          [mission_id]: {
            ...prev[mission_id],
            steps: [...prev[mission_id].steps, { timestep: event.timestep, elapsed_pct: event.elapsed_pct, probability: event.probability }],
          }
        }))
      }

      if (event.type === 'cancel') {
        setMData(prev => ({
          ...prev,
          [mission_id]: { ...prev[mission_id], status: STATUS.CANCELED, canceled: true }
        }))
      }

      if (event.type === 'done') {
        es.close()
        delete esRefs.current[mission_id]
        const finalStatus = event.canceled ? STATUS.CANCELED : STATUS.COMPLETED
        setMData(prev => ({
          ...prev,
          [mission_id]: {
            ...prev[mission_id],
            status: finalStatus,
            wasCorrect: event.was_correct,
          }
        }))
        setStats(prev => ({
          pruned:   prev.pruned + (event.canceled ? 1 : 0),
          passed:   prev.passed + (event.canceled ? 0 : 1),
          correct:  prev.correct + (event.was_correct ? 1 : 0),
          total:    prev.total + 1,
        }))
      }

      if (event.type === 'error') {
        es.close()
        delete esRefs.current[mission_id]
        setMData(prev => ({
          ...prev,
          [mission_id]: { ...prev[mission_id], status: STATUS.ERROR }
        }))
      }
    }
    es.onerror = () => {
      es.close()
      delete esRefs.current[mission_id]
    }
  }, [dataPath, threshold])

  const runAll = useCallback(() => {
    setRunningAll(true)
    missions.forEach((m, i) => {
      setTimeout(() => streamMission(m.mission_id), i * 300)  // stagger by 300ms
    })
  }, [missions, streamMission])

  const resetAll = () => {
    Object.values(esRefs.current).forEach(es => es.close())
    esRefs.current = {}
    const init = {}
    missions.forEach(m => {
      init[m.mission_id] = { status: STATUS.IDLE, steps: [], probability: null, canceled: false, wasCorrect: null }
    })
    setMData(init)
    setRunningAll(false)
    setStats({ pruned: 0, passed: 0, correct: 0, total: 0 })
  }

  useEffect(() => () => Object.values(esRefs.current).forEach(es => es.close()), [])

  const anyRunning = Object.values(missionData).some(m => m.status === STATUS.RUNNING)

  return (
    <div className="panel-content">
      {/* Config bar */}
      <div className="card" style={{ marginBottom: '14px' }}>
        <div style={{ display: 'flex', gap: '16px', alignItems: 'flex-end', flexWrap: 'wrap' }}>
          <div>
            <div style={{ fontSize: '10px', color: 'var(--text-dim)', letterSpacing: '0.08em', marginBottom: '5px', fontFamily: 'var(--mono)' }}>DATASET PATH</div>
            <input
              className="field-input"
              style={{ width: '260px', fontSize: '11px' }}
              value={dataPath}
              onChange={e => setDataPath(e.target.value)}
              disabled={anyRunning || loading}
            />
          </div>
          <div>
            <div style={{ fontSize: '10px', color: 'var(--text-dim)', letterSpacing: '0.08em', marginBottom: '5px', fontFamily: 'var(--mono)' }}>MISSIONS</div>
            <input
              className="field-input"
              style={{ width: '60px' }}
              type="number" min="1" max="24"
              value={n}
              onChange={e => setN(parseInt(e.target.value) || 12)}
              disabled={anyRunning || loading}
            />
          </div>
          <div>
            <div style={{ fontSize: '10px', color: 'var(--text-dim)', letterSpacing: '0.08em', marginBottom: '5px', fontFamily: 'var(--mono)' }}>CONFIDENCE THRESHOLD</div>
            <input
              className="field-input"
              style={{ width: '70px' }}
              type="number" min="0.1" max="0.99" step="0.05"
              value={threshold}
              onChange={e => setThreshold(parseFloat(e.target.value) || 0.5)}
              disabled={anyRunning}
            />
          </div>
          <div>
            <div style={{ fontSize: '10px', color: 'var(--text-dim)', letterSpacing: '0.08em', marginBottom: '5px', fontFamily: 'var(--mono)' }}>
              MIN ELAPSED %
              <span style={{ color: '#555', marginLeft: '4px' }}>(match training)</span>
            </div>
            <input
              className="field-input"
              style={{ width: '70px' }}
              type="number" min="0.05" max="1.0" step="0.05"
              value={minElapsed}
              onChange={e => setMinElapsed(parseFloat(e.target.value) || 0.4)}
              disabled={anyRunning}
            />
          </div>

          <div style={{ display: 'flex', gap: '8px', marginLeft: 'auto', alignItems: 'center' }}>
            <div style={{ fontSize: '10px', color: apiOnline ? 'var(--green)' : apiOnline === false ? 'var(--red)' : 'var(--text-dim)', fontFamily: 'var(--mono)', letterSpacing: '0.06em' }}>
              ● API {apiOnline == null ? '...' : apiOnline ? 'ONLINE' : 'OFFLINE'}
            </div>
            <button className="launch-btn" style={{ padding: '8px 16px', fontSize: '11px' }}
              onClick={loadMissions} disabled={loading || anyRunning || apiOnline === false}>
              {loading ? 'LOADING...' : '↺ LOAD MISSIONS'}
            </button>
            {missions.length > 0 && (
              <>
                <button className="launch-btn" style={{ padding: '8px 16px', fontSize: '11px' }}
                  onClick={runAll} disabled={anyRunning || loading}>
                  ⚡ RUN ALL
                </button>
                <button className="launch-btn" style={{ padding: '8px 16px', fontSize: '11px', background: 'transparent' }}
                  onClick={resetAll} disabled={anyRunning}>
                  ↺ RESET
                </button>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Stats bar */}
      {stats.total > 0 && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px', marginBottom: '14px' }}>
          {[
            { label: 'MISSIONS RUN',    value: stats.total,                           color: 'var(--cyan)'  },
            { label: 'PRUNED (CANCEL)', value: stats.pruned,                          color: 'var(--red)'   },
            { label: 'PASSED THROUGH',  value: stats.passed,                          color: 'var(--green)' },
            { label: 'CORRECT CALLS',   value: `${stats.correct}/${stats.total}`,     color: 'var(--orange)'},
          ].map(s => (
            <div key={s.label} style={{ background: 'var(--bg2)', border: '1px solid var(--border)', padding: '10px 14px' }}>
              <div style={{ fontSize: '10px', color: 'var(--text-dim)', letterSpacing: '0.06em', fontFamily: 'var(--mono)', marginBottom: '4px' }}>{s.label}</div>
              <div style={{ fontSize: '22px', color: s.color, fontFamily: 'var(--mono)' }}>{s.value}</div>
            </div>
          ))}
        </div>
      )}

      {/* Empty state */}
      {missions.length === 0 && (
        <div style={{
          border: '1px solid var(--border)', background: 'var(--bg2)',
          padding: '60px 24px', textAlign: 'center',
        }}>
          <div style={{ fontSize: '13px', color: 'var(--text-dim)', letterSpacing: '0.1em', fontFamily: 'var(--mono)', marginBottom: '12px' }}>
            SIMULATOR READY
          </div>
          <div style={{ fontSize: '12px', color: 'var(--text-dim)', lineHeight: 1.8 }}>
            {apiOnline === false
              ? 'API is offline. Start it with: uvicorn src.api.main:app --port 8000'
              : 'Click "LOAD MISSIONS" to sample trajectories from the dataset.\nThe model will predict each mission\'s outcome in realtime.'}
          </div>
          {apiOnline === false && (
            <div style={{ marginTop: '16px', fontSize: '11px', fontFamily: 'var(--mono)', color: 'var(--orange)', letterSpacing: '0.06em' }}>
              Then train a model: python3 -m src.ml.train --data data/merged/missions.parquet
            </div>
          )}
        </div>
      )}

      {/* Mission grid */}
      {missions.length > 0 && (
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))',
          gap: '10px',
        }}>
          {missions.map(m => {
            const d = missionData[m.mission_id] || { status: STATUS.IDLE, steps: [], canceled: false, wasCorrect: null }
            return (
              <MissionCard
                key={m.mission_id}
                mission={{ ...m, ...d }}
                onStream={streamMission}
                isActive={d.status === STATUS.RUNNING}
              />
            )
          })}
        </div>
      )}

      {/* Legend */}
      {missions.length > 0 && (
        <div style={{ marginTop: '16px', display: 'flex', gap: '20px', fontSize: '10px', color: 'var(--text-dim)', fontFamily: 'var(--mono)', flexWrap: 'wrap' }}>
          {[
            ['var(--cyan)',   '● RUNNING'],
            ['var(--red)',    '● CANCELED (model predicted failure, stopped sim)'],
            ['var(--green)',  '● COMPLETED (model passed sim through)'],
            ['var(--orange)', '● ERROR'],
          ].map(([col, label]) => (
            <span key={label} style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
              <span style={{ color: col }}>{label}</span>
            </span>
          ))}
        </div>
      )}
    </div>
  )
}
