import { useEffect, useRef, useState } from 'react'

const LOG_MESSAGES = [
  ['ok', '[14:34:56.823] HEARTBEAT OK. SYNC LATENCY: 12ms.'],
  ['ok', '[14:34:59.831] HEARTBEAT OK. SYNC LATENCY: 11ms.'],
  ['ok', '[14:35:02.933] HEARTBEAT OK. SYNC LATENCY: 12ms.'],
  ['ok', '[14:35:05.826] HEARTBEAT OK. SYNC LATENCY: 13ms.'],
  ['dim', '[14:35:08.824] DATASET STREAM: batch 1/219 processed.'],
  ['ok', '[14:35:11.823] HEARTBEAT OK. SYNC LATENCY: 10ms.'],
  ['ok', '[14:35:14.837] HEARTBEAT OK. SYNC LATENCY: 12ms.'],
  ['dim', '[14:35:17.826] GC CYCLE EXECUTED. FREED 124MB.'],
  ['ok', '[14:35:21.823] HEARTBEAT OK. SYNC LATENCY: 12ms.'],
  ['warn', '[14:35:44.824] WARN: MINOR FLUCTUATION IN THERMAL BAND.'],
  ['ok', '[14:35:47.831] HEARTBEAT OK. SYNC LATENCY: 12ms.'],
  ['info', '[14:36:01.002] MODEL WEIGHTS LOADED. AUC_LAST: 0.870'],
  ['info', '[14:36:04.111] DATASET: data/merged/missions.parquet'],
  ['dim', '[14:36:04.112] ROWS: 85,282,712   MISSIONS: 10,000'],
  ['ok', '[14:36:07.823] HEARTBEAT OK. SYNC LATENCY: 11ms.'],
  ['dim', '[14:36:12.001] SCALER: models/test/scaler_lstm_binary.pkl'],
  ['ok', '[14:36:15.831] HEARTBEAT OK. SYNC LATENCY: 12ms.'],
  ['info', '[14:36:19.220] PYARROW FILTER: train_ids ✓  val_ids ✓  test_ids ✓'],
  ['ok', '[14:36:22.823] HEARTBEAT OK. SYNC LATENCY: 12ms.'],
  ['dim', '[14:36:25.440] POS_WEIGHT: 1.832 (n_neg/n_pos from training set)'],
]

function Cube() {
  return (
    <div className="cube-scene">
      <div className="cube">
        {[
          ['face-front', 'ATTN_HEAD'],
          ['face-back', 'EMBED_IN'],
          ['face-left', 'LSTM_01'],
          ['face-right', 'LSTM_02'],
          ['face-top', 'DENSE_OUT'],
          ['face-bottom', 'DROPOUT'],
        ].map(([cls, label]) => (
          <div key={cls} className={`face ${cls}`}>{label}</div>
        ))}
      </div>
    </div>
  )
}

function Gauge({ value = 0.87 }) {
  const circumference = 2 * Math.PI * 28
  const offset = circumference * (1 - value)
  return (
    <div className="gauge-wrap">
      <div className="gauge">
        <svg viewBox="0 0 64 64" width="64" height="64">
          <circle className="gauge-track" cx="32" cy="32" r="28" />
          <circle
            className="gauge-fill"
            cx="32" cy="32" r="28"
            style={{ strokeDasharray: circumference, strokeDashoffset: offset }}
          />
        </svg>
        <div className="gauge-text">
          {value.toFixed(2)}
          <small>AUC</small>
        </div>
      </div>
      <div>
        <div style={{ fontSize: '26px', color: 'var(--green)', lineHeight: 1 }}>{value.toFixed(2)}</div>
        <div style={{ fontSize: '11px', color: 'var(--text-dim)', letterSpacing: '0.06em', marginTop: '3px' }}>EVAL.LAST</div>
        <div style={{ fontSize: '11px', color: 'var(--green)', marginTop: '4px' }}>ROC-AUC SCORE</div>
      </div>
    </div>
  )
}

function Sparkline() {
  const data = [0.28, 0.32, 0.35, 0.33, 0.36, 0.38, 0.35, 0.37, 0.353]
  const w = 120, h = 30
  const max = Math.max(...data), min = Math.min(...data)
  const points = data.map((v, i) => {
    const x = (i / (data.length - 1)) * w
    const y = h - ((v - min) / (max - min + 0.01)) * h * 0.8 - h * 0.08
    return `${x},${y}`
  }).join(' ')
  return (
    <svg width={w} height={h} style={{ display: 'block', marginTop: '6px' }}>
      <polyline points={points} fill="none" stroke="var(--red)" strokeWidth="1.5" />
    </svg>
  )
}

export default function Overview() {
  const [logs, setLogs] = useState([LOG_MESSAGES[0], LOG_MESSAGES[1], LOG_MESSAGES[2]])
  const idxRef = useRef(3)
  const termRef = useRef(null)

  useEffect(() => {
    const tick = () => {
      const msg = LOG_MESSAGES[idxRef.current % LOG_MESSAGES.length]
      setLogs(prev => [...prev.slice(-70), msg])
      idxRef.current++
    }
    const id = setInterval(tick, 900 + Math.random() * 800)
    return () => clearInterval(id)
  }, [])

  useEffect(() => {
    if (termRef.current) termRef.current.scrollTop = termRef.current.scrollHeight
  }, [logs])

  const tiles = [
    {
      label: 'TELEMETRY RECORDS', sub: 'SYS.DB.1',
      value: '10', unit: ',000',
      note: '↑ 85.3M rows in corpus', noteColor: 'var(--green)',
      borderColor: 'var(--cyan)',
    },
    {
      label: 'ACTIVE PAYLOAD', sub: 'VOL.04',
      value: '13', unit: '.44 GB',
      note: '⚠  Zstd compressed Parquet', noteColor: 'var(--orange)',
      borderColor: 'var(--cyan-dim)',
    },
    {
      label: 'ANOMALY DETECTION', sub: 'HEURISTIC',
      value: '35', unit: '.3%',
      note: null, // sparkline
      noteColor: 'var(--red)',
      borderColor: 'var(--red)',
      isSparkline: true,
    },
    {
      label: 'MODEL CONFIDENCE', sub: '',
      isGauge: true,
      borderColor: 'var(--green)',
    },
  ]

  return (
    <div className="panel-content">
      {/* Metric tiles */}
      <div className="g4" style={{ marginBottom: '16px' }}>
        {tiles.map(t => (
          <div key={t.label} className="tile" style={{ borderTopColor: t.borderColor }}>
            <div className="tile-label">
              {t.label}
              <span>{t.sub}</span>
            </div>
            {t.isGauge ? (
              <Gauge />
            ) : t.isSparkline ? (
              <>
                <div className="tile-value" style={{ color: 'var(--red)' }}>
                  {t.value}<em>{t.unit}</em>
                </div>
                <Sparkline />
                <div className="tile-sub" style={{ color: 'var(--red)', marginTop: '4px' }}>
                  Failure threshold balanced
                </div>
              </>
            ) : (
              <>
                <div className="tile-value">{t.value}<em>{t.unit}</em></div>
                <div className="tile-sub" style={{ color: t.noteColor }}>{t.note}</div>
              </>
            )}
          </div>
        ))}
      </div>

      {/* Topology + Log */}
      <div className="g-left-right">
        {/* Left: architecture */}
        <div className="card">
          <div className="card-hdr">
            ARCHITECTURE TOPOLOGY
            <span className="badge">LSTM_BINARY</span>
          </div>

          <Cube />

          <div style={{ textAlign: 'center', marginTop: '8px' }}>
            <div style={{ fontSize: '13px', color: 'var(--green)', letterSpacing: '0.12em', fontWeight: 700 }}>
              MODEL LOADED
            </div>
            <div style={{ fontSize: '11px', color: 'var(--text-dim)', marginTop: '4px', letterSpacing: '0.04em' }}>
              CHECKSUM: 0X8A7B9C... VERIFIED
            </div>
          </div>

          <div style={{ marginTop: '16px', borderTop: '1px solid var(--border)', paddingTop: '13px' }}>
            {[
              ['INPUT DIM', '13 physics-invariant features'],
              ['HIDDEN DIM', '128 units × 2 LSTM layers'],
              ['TOTAL PARAMS', '~250K'],
              ['GRAD CLIP', 'max_norm = 1.0  ✓'],
              ['LR SCHEDULER', 'ReduceLROnPlateau ✓'],
              ['OPTIMIZER', 'Adam (lr = 1e-3)'],
              ['POS_WEIGHT', 'dynamic (n_neg / n_pos)'],
            ].map(([k, v]) => (
              <div key={k} className="stat-row">{k} <span>{v}</span></div>
            ))}
          </div>
        </div>

        {/* Right: terminal */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
            <span style={{ fontSize: '11px', color: 'var(--text-dim)', letterSpacing: '0.08em', textTransform: 'uppercase' }}>
              SYS.LOG // TTY1
            </span>
            <div className="dots">
              <span className="dot dot-r" />
              <span className="dot dot-y" />
              <span className="dot dot-g" />
            </div>
          </div>
          <div
            ref={termRef}
            className="terminal"
            style={{ flex: 1, minHeight: '320px' }}
          >
            {logs.map(([type, msg], i) => (
              <div key={i} className={type}>{msg}</div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
