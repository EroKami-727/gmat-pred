import { useEffect, useRef, useState } from 'react'
import {
  AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from 'recharts'

// ── 50-epoch ground-truth curves ────────────────────────────
function makeCurves(n = 50) {
  return Array.from({ length: n }, (_, i) => {
    const t = i / (n - 1)
    const train = 2.45 * Math.exp(-3.8 * t) + 0.18 + (Math.random() - 0.5) * 0.012
    const val   = 2.62 * Math.exp(-3.1 * t) + 0.62 + (Math.random() - 0.5) * 0.015
    const auc   = 0.51 + 0.38 / (1 + Math.exp(-10 * (t - 0.35))) + (Math.random() - 0.5) * 0.004
    return { train: +train.toFixed(4), val: +val.toFixed(4), auc: +Math.min(auc, 0.895).toFixed(4) }
  })
}
const CURVES = makeCurves(50)

const INIT_LOGS = [
  ['info', '> ORBITGUARD TRAINING MODULE READY'],
  ['dim',  '> DATASET: data/merged/missions.parquet'],
  ['dim',  '> DEVICE: cuda (NVIDIA RTX 4060, 8.0 GB)'],
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
  const [liveData, setLiveData] = useState([])   // grows one point per epoch

  const epochRef    = useRef(1)
  const intervalRef = useRef(null)
  const termRef     = useRef(null)

  const [params, setParams] = useState({
    lr: '1e-3', batch: '32', epochs: '50',
    optimizer: 'Adam', arch: 'TrajectoryLSTM', earlyExit: '1.0', hidden: '128',
  })
  const set = k => e => setParams(p => ({ ...p, [k]: e.target.value }))

  // called once per simulated epoch
  const nextEpoch = (maxEpochs) => {
    const ep = epochRef.current
    if (ep > maxEpochs) return   // safety guard (interval already cleared in effect)

    const idx   = Math.min(ep - 1, CURVES.length - 1)
    const { train, val, auc } = CURVES[idx]
    const lr    = ep > Math.floor(maxEpochs * 0.6) ? '5.00e-4' : '1.00e-3'
    const epStr = String(ep).padStart(2, ' ')

    // append to chart
    setLiveData(prev => [...prev, { epoch: ep, train, val, auc }])

    // append to terminal
    const line = `Epoch [${epStr}/${maxEpochs}] | Loss: ${train} | Val: ${val} | AUC: ${auc} | lr=${lr}`
    setLogs(prev => [...prev.slice(-100), ['ok', line]])

    epochRef.current = ep + 1

    // done?
    if (ep >= maxEpochs) {
      clearInterval(intervalRef.current)
      setRunning(false)
      setLogs(prev => [
        ...prev,
        ['dim', ''],
        ['info', `▸ TRAINING COMPLETE — best AUC: ${auc} at epoch ${ep}`],
        ['dim',  `▸ Model saved → models/production/best_model_${params.arch.includes('LSTM') ? 'lstm' : 'transformer'}_binary.pt`],
        ['dim',  `▸ Metrics saved → models/production/metrics_${params.arch.includes('LSTM') ? 'lstm' : 'transformer'}_binary.json`],
      ])
    }
  }

  const toggleTraining = () => {
    if (running) {
      clearInterval(intervalRef.current)
      setLogs(prev => [...prev, ['warn', '> TRAINING STOPPED BY USER.']])
      setRunning(false)
      return
    }

    const maxEpochs = Math.max(1, parseInt(params.epochs) || 50)

    // reset for a fresh run
    epochRef.current = 1
    setLiveData([])
    setLogs([
      ['info', `> LAUNCHING: python3 -m src.ml.train --epochs ${maxEpochs} --model ${params.arch.includes('LSTM') ? 'lstm' : 'transformer'} --lr ${params.lr} --batch-size ${params.batch} --early-exit ${params.earlyExit}`],
      ['dim',  `> BATCHES PER EPOCH: ${Math.ceil(7000 / parseInt(params.batch || 32))}`],
      ['dim',  `> DEVICE: cuda (NVIDIA RTX 4060, 8.0 GB)`],
      ['dim',  `> POS_WEIGHT: computing from training set...`],
      ['dim',  ''],
    ])

    setRunning(true)
    // kick off first epoch immediately, then interval
    setTimeout(() => nextEpoch(maxEpochs), 400)
    intervalRef.current = setInterval(() => nextEpoch(maxEpochs), 750 + Math.random() * 250)
  }

  useEffect(() => () => clearInterval(intervalRef.current), [])

  useEffect(() => {
    if (termRef.current) termRef.current.scrollTop = termRef.current.scrollHeight
  }, [logs])

  // split liveData for the two charts
  const lossData = liveData.map(d => ({ epoch: d.epoch, train: d.train, val: d.val }))
  const aucData  = liveData.map(d => ({ epoch: d.epoch, auc: d.auc }))
  const bestAuc  = liveData.length ? Math.max(...liveData.map(d => d.auc)).toFixed(4) : '—'
  const lastEp   = epochRef.current - 1

  return (
    <div className="panel-content">
      <div className="g-left-right">

        {/* ── Left: hyperparams ── */}
        <div className="card">
          <div className="card-hdr">HYPERPARAMETERS <span className="badge">CONF.02</span></div>

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
              <option>TrajectoryLSTM</option>
              <option>TrajectoryTransformer</option>
            </select>
          </div>

          <button
            className={`launch-btn${running ? ' running' : ''}`}
            onClick={toggleTraining}
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
                  width: `${(lastEp / (parseInt(params.epochs) || 50)) * 100}%`,
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

          {/* Charts — both start empty, grow as epochs come in */}
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
