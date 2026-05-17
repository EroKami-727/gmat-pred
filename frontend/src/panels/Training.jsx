import { useEffect, useRef, useState } from 'react'
import {
  AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from 'recharts'
import { API } from '../lib/api.js'

// Parse a python3 -m src.ml.train ... command into form params
function parseTrainCommand(cmd) {
  const flags = {
    '--lr':           v => ({ lr: v }),
    '--batch-size':   v => ({ batch: v }),
    '--epochs':       v => ({ epochs: v }),
    '--hidden-dim':   v => ({ hidden: v }),
    '--early-exit':   v => ({ earlyExit: v }),
    '--model':        v => ({ arch: v === 'transformer' ? 'TrajectoryTransformer' : 'TrajectoryLSTM' }),
    '--data':         v => ({ data: v }),
    '--output-dir':   v => ({ outputDir: v }),
  }
  const tokens = cmd.trim().split(/\s+/)
  const parsed = {}
  for (let i = 0; i < tokens.length; i++) {
    const fn = flags[tokens[i]]
    if (fn && tokens[i + 1] && !tokens[i + 1].startsWith('--')) {
      Object.assign(parsed, fn(tokens[++i]))
    }
  }
  return parsed
}

const INIT_LOGS = [
  ['info', '> ORBITGUARD TRAINING MODULE READY'],
  ['dim',  '> DATASET: data/merged/missions.parquet'],
  ['dim',  '> DEVICE: detecting...'],
  ['dim',  '> GRAD CLIP: max_norm=1.0   LR SCHEDULER: ReduceLROnPlateau'],
  ['dim',  ''],
  ['dim',  'Configure hyperparameters and press LAUNCH TRAINING.'],
]

const chartStyle = {
  cartesian: { strokeDasharray: '3 3', stroke: '#1a1a1a' },
  axis: { stroke: '#333333', tick: { fill: '#555555', fontSize: 9, fontFamily: 'Share Tech Mono, Courier New, monospace' } },
  tooltip: {
    contentStyle: { background: '#0f0f0f', border: '1px solid #2a2a2a', fontSize: 10, fontFamily: 'Share Tech Mono, Courier New, monospace' },
    labelStyle: { color: '#888888' },
  },
}

export default function Training() {
  const [logs, setLogs]       = useState(INIT_LOGS)
  const [running, setRunning] = useState(false)
  const [liveData, setLiveData] = useState([])
  const [jobId, setJobId]     = useState(null)
  const [apiOnline, setApiOnline] = useState(null)  // null=checking, true/false

  const esRef     = useRef(null)   // EventSource
  const termRef   = useRef(null)

  const [cmdInput, setCmdInput] = useState('')
  const [cmdError, setCmdError] = useState(null)

  const [params, setParams] = useState({
    lr: '1e-3', batch: '32', epochs: '30',
    optimizer: 'Adam', arch: 'TrajectoryTransformer', earlyExit: '1.0', hidden: '128',
    data: 'data/merged/missions.parquet', outputDir: 'models/transformer_production',
  })
  const set = k => e => setParams(p => ({ ...p, [k]: e.target.value }))

  const applyCommand = () => {
    if (!cmdInput.trim()) return
    if (!cmdInput.includes('src.ml.train')) {
      setCmdError('Not a train command — must contain src.ml.train')
      return
    }
    const parsed = parseTrainCommand(cmdInput)
    if (Object.keys(parsed).length === 0) {
      setCmdError('No recognizable flags found')
      return
    }
    setParams(p => ({ ...p, ...parsed }))
    setCmdError(null)
    setCmdInput('')
  }

  // Check API health on mount
  useEffect(() => {
    fetch(`${API}/api/health`, { signal: AbortSignal.timeout(3000) })
      .then(r => r.json())
      .then(d => {
        setApiOnline(true)
        setLogs(prev => [
          ...prev.slice(0, 2),
          ['dim', `> DEVICE: ${d.device === 'cuda' ? `cuda (${d.gpu}, ${d.gpu_mem_gb} GB)` : 'cpu'}`],
          ...prev.slice(3),
        ])
      })
      .catch(() => {
        setApiOnline(false)
        setLogs(prev => [
          ...prev.slice(0, 2),
          ['warn', '> API OFFLINE — start: uvicorn src.api.main:app --port 8000'],
          ...prev.slice(3),
        ])
      })
  }, [])

  const appendLog = (type, msg) =>
    setLogs(prev => [...prev.slice(-120), [type, msg]])

  const startTraining = async () => {
    const maxEpochs = Math.max(1, parseInt(params.epochs) || 30)
    setLiveData([])

    // Start training job
    let jid
    try {
      const res = await fetch(`${API}/api/train/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...params, epochs: maxEpochs }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const { job_id, command } = await res.json()
      jid = job_id
      setJobId(jid)
    } catch (err) {
      appendLog('warn', `> FAILED TO START: ${err.message}`)
      setRunning(false)
      return
    }

    setLogs([
      ['info', `> LAUNCHING TRAINING JOB ${jid}`],
      ['dim',  `> MODEL: ${params.arch}`],
      ['dim',  `> LR=${params.lr}  BATCH=${params.batch}  EPOCHS=${maxEpochs}  EXIT=${params.earlyExit}`],
      ['dim',  ''],
    ])
    setRunning(true)

    // Connect to SSE stream
    const es = new EventSource(`${API}/api/train/stream/${jid}`)
    esRef.current = es

    es.onmessage = (e) => {
      let event
      try { event = JSON.parse(e.data) } catch { return }

      if (event.type === 'log') {
        const typeMap = { '▸': 'info', 'Epoch': 'ok', 'WARN': 'warn', 'ERROR': 'warn' }
        const cls = Object.entries(typeMap).find(([k]) => event.text.includes(k))?.[1] || 'dim'
        appendLog(cls, event.text)
      }

      if (event.type === 'epoch') {
        setLiveData(prev => [...prev, {
          epoch: event.epoch,
          train: event.train_loss,
          val:   event.val_loss,
          auc:   event.auc,
        }])
      }

      if (event.type === 'done') {
        es.close()
        setRunning(false)
        setJobId(null)
        if (event.exit_code === 0) {
          appendLog('info', '▸ TRAINING COMPLETE — model saved.')
        } else if (event.exit_code === -1) {
          appendLog('warn', '> TRAINING STOPPED BY USER.')
        } else {
          appendLog('warn', `> TRAINING FAILED (exit ${event.exit_code})`)
        }
      }
    }

    es.onerror = () => {
      appendLog('warn', '> SSE CONNECTION LOST')
      setRunning(false)
    }
  }

  const stopTraining = async () => {
    esRef.current?.close()
    if (jobId) {
      await fetch(`${API}/api/train/stop/${jobId}`).catch(() => {})
    }
    appendLog('warn', '> TRAINING STOPPED BY USER.')
    setRunning(false)
    setJobId(null)
  }

  const toggleTraining = () => {
    if (running) {
      stopTraining()
    } else {
      startTraining()
    }
  }

  useEffect(() => () => esRef.current?.close(), [])

  useEffect(() => {
    if (termRef.current) termRef.current.scrollTop = termRef.current.scrollHeight
  }, [logs])

  const lossData = liveData.map(d => ({ epoch: d.epoch, train: d.train, val: d.val }))
  const aucData  = liveData.map(d => ({ epoch: d.epoch, auc: d.auc }))
  const bestAuc  = liveData.length ? Math.max(...liveData.map(d => d.auc)).toFixed(4) : '—'
  const lastEp   = liveData.length

  return (
    <div className="panel-content">
      <div className="g-left-right">

        {/* ── Left: hyperparams ── */}
        <div className="card">
          <div className="card-hdr">
            HYPERPARAMETERS <span className="badge">CONF.02</span>
          </div>

          {/* Command paste box */}
          <div style={{ marginBottom: '16px', paddingBottom: '16px', borderBottom: '1px solid var(--border)' }}>
            <div style={{ fontSize: '10px', color: 'var(--text-dim)', letterSpacing: '0.08em', marginBottom: '6px', fontFamily: 'var(--mono)' }}>
              PASTE TRAIN COMMAND → AUTO-FILL
            </div>
            <textarea
              className="field-input"
              rows={2}
              style={{ width: '100%', resize: 'vertical', fontSize: '10px', fontFamily: 'var(--mono)', lineHeight: 1.6 }}
              placeholder="python3 -m src.ml.train --model transformer --epochs 30 ..."
              value={cmdInput}
              onChange={e => { setCmdInput(e.target.value); setCmdError(null) }}
              onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); applyCommand() } }}
              disabled={running}
            />
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '6px' }}>
              <button
                onClick={applyCommand}
                disabled={running || !cmdInput.trim()}
                style={{
                  background: 'transparent', border: '1px solid var(--border2)',
                  color: 'var(--cyan)', fontFamily: 'var(--mono)', fontSize: '10px',
                  letterSpacing: '0.08em', padding: '4px 12px', cursor: 'pointer',
                  opacity: (!cmdInput.trim() || running) ? 0.4 : 1,
                }}
              >
                APPLY
              </button>
              {cmdError && <span style={{ fontSize: '10px', color: 'var(--red)', fontFamily: 'var(--mono)' }}>{cmdError}</span>}
              {!cmdError && !cmdInput && <span style={{ fontSize: '10px', color: 'var(--text-dim)', fontFamily: 'var(--mono)' }}>or press Enter</span>}
            </div>
          </div>

          {/* API status indicator */}
          <div style={{ marginBottom: '14px', padding: '7px 10px', background: 'var(--bg3)', border: '1px solid var(--border)', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{
              width: '7px', height: '7px', borderRadius: '50%', flexShrink: 0,
              background: apiOnline == null ? '#888888' : apiOnline ? 'var(--green)' : 'var(--red)',
              boxShadow: apiOnline ? '0 0 6px var(--green)' : 'none',
            }} />
            <span style={{ color: 'var(--text-dim)', letterSpacing: '0.06em' }}>
              API: {apiOnline == null ? 'CHECKING...' : apiOnline ? 'ONLINE' : 'OFFLINE'}
            </span>
          </div>

          {[
            ['LEARNING RATE',   'lr'],
            ['BATCH SIZE',      'batch'],
            ['EPOCHS',          'epochs'],
            ['HIDDEN DIM',      'hidden'],
            ['EARLY EXIT FRAC', 'earlyExit'],
          ].map(([label, key]) => (
            <div key={key} className="field">
              <div className="field-label">{label}</div>
              <input
                className="field-input"
                type="text"
                value={params[key]}
                onChange={set(key)}
                disabled={running}
              />
            </div>
          ))}

          <div className="field">
            <div className="field-label">OPTIMIZER</div>
            <select className="field-select" value={params.optimizer} onChange={set('optimizer')} disabled={running}>
              <option>Adam</option><option>AdamW</option><option>SGD</option>
            </select>
          </div>

          <div className="field">
            <div className="field-label">MODEL ARCH</div>
            <select className="field-select" value={params.arch} onChange={set('arch')} disabled={running}>
              <option>TrajectoryTransformer</option>
              <option>TrajectoryLSTM</option>
            </select>
          </div>

          <button
            className={`launch-btn${running ? ' running' : ''}`}
            onClick={toggleTraining}
            disabled={apiOnline === false}
            title={apiOnline === false ? 'Start the FastAPI backend first' : ''}
          >
            {running ? '⏹  STOP TRAINING' : '⚡  LAUNCH TRAINING'}
          </button>

          {/* Live progress */}
          {(running || liveData.length > 0) && (
            <div style={{ marginTop: '12px', background: 'var(--bg3)', border: '1px solid var(--border)', padding: '10px 12px' }}>
              <div style={{ fontSize: '11px', color: 'var(--text-dim)', letterSpacing: '0.08em', marginBottom: '6px' }}>
                PROGRESS
              </div>
              <div style={{ height: '3px', background: 'var(--border2)', marginBottom: '8px' }}>
                <div style={{
                  height: '100%',
                  width: `${(lastEp / (parseInt(params.epochs) || 30)) * 100}%`,
                  background: running ? 'var(--cyan)' : 'var(--green)',
                  transition: 'width 0.4s ease',
                }} />
              </div>
              <div className="stat-row">
                EPOCH <span style={{ color: 'var(--cyan)' }}>{lastEp} / {params.epochs}</span>
              </div>
              <div className="stat-row">
                BEST AUC <span style={{ color: 'var(--green)' }}>{bestAuc}</span>
              </div>
              {jobId && (
                <div className="stat-row">
                  JOB <span style={{ color: 'var(--text-dim)' }}>{jobId}</span>
                </div>
              )}
            </div>
          )}

          <div style={{ marginTop: '14px', borderTop: '1px solid var(--border)', paddingTop: '12px' }}>
            {[
              ['GRAD CLIP',    'max_norm=1.0 ✓'],
              ['SCHEDULER',   'ReduceLROnPlateau ✓'],
              ['POS_WEIGHT',  'dynamic ✓'],
              ['METRICS JSON','saved per epoch ✓'],
              ['PYARROW FIX', 'pa.array() filter ✓'],
            ].map(([k, v]) => (
              <div key={k} className="stat-row">{k} <span style={{ color: 'var(--green)' }}>{v}</span></div>
            ))}
          </div>
        </div>

        {/* ── Right: terminal + live charts ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>

          {/* Terminal */}
          <div className="card" style={{ padding: '14px 16px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
              <span style={{ fontSize: '11px', color: 'var(--text-dim)', letterSpacing: '0.08em', textTransform: 'uppercase' }}>
                LIVE OUTPUT
              </span>
              <span className="badge-live" style={{ background: running ? 'var(--green)' : 'var(--cyan)' }}>
                {running ? '● RUNNING' : 'LIVE OUTPUT'}
              </span>
            </div>
            <div ref={termRef} className="terminal" style={{ height: '200px' }}>
              {logs.map(([type, msg], i) => (
                <div key={i} className={type}>{msg}</div>
              ))}
            </div>
          </div>

          {/* Charts */}
          <div className="g2">
            <div className="chart-box">
              <div className="chart-label">
                TRAINING &amp; VAL LOSS
                <div>
                  <span className="legend-item">
                    <span className="legend-dot" style={{ background: '#aaaaaa' }} /> TRAIN
                  </span>
                  {' '}
                  <span className="legend-item">
                    <span className="legend-dot" style={{ background: '#666666' }} /> VAL
                  </span>
                </div>
              </div>
              <ResponsiveContainer width="100%" height={160}>
                <AreaChart data={lossData} isAnimationActive={false}>
                  <defs>
                    <linearGradient id="gTrain" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor="#aaaaaa" stopOpacity={0.18} />
                      <stop offset="95%" stopColor="#aaaaaa" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid {...chartStyle.cartesian} />
                  <XAxis dataKey="epoch" {...chartStyle.axis} />
                  <YAxis {...chartStyle.axis} domain={['auto', 'auto']} />
                  <Tooltip {...chartStyle.tooltip} />
                  <Area type="monotone" dataKey="train" stroke="#aaaaaa" fill="url(#gTrain)" strokeWidth={2} dot={false} name="Train Loss" isAnimationActive={false} />
                  <Area type="monotone" dataKey="val"   stroke="#666666" fill="none"          strokeWidth={2} dot={false} name="Val Loss"   isAnimationActive={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            <div className="chart-box">
              <div className="chart-label">
                ROC AUC SCORE
                <span style={{ fontSize: '11px', color: 'var(--text-dim)' }}>TARGET: 0.90 ─ ─</span>
              </div>
              <ResponsiveContainer width="100%" height={160}>
                <AreaChart data={aucData} isAnimationActive={false}>
                  <defs>
                    <linearGradient id="gAuc" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor="#e2e2e2" stopOpacity={0.2} />
                      <stop offset="95%" stopColor="#e2e2e2" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid {...chartStyle.cartesian} />
                  <XAxis dataKey="epoch" {...chartStyle.axis} />
                  <YAxis domain={[0.45, 1.0]} {...chartStyle.axis} tickFormatter={v => v.toFixed(2)} />
                  <Tooltip {...chartStyle.tooltip} formatter={v => v.toFixed(4)} />
                  <ReferenceLine y={0.90} stroke="#333333" strokeDasharray="4 4" />
                  <Area type="monotone" dataKey="auc" stroke="#e2e2e2" fill="url(#gAuc)" strokeWidth={2.5} dot={false} name="AUC" isAnimationActive={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

        </div>
      </div>
    </div>
  )
}
