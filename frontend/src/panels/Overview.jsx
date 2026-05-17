import { useEffect, useRef, useState } from 'react'
import { API } from '../lib/api.js'

function Gauge({ value = null }) {
  const display = value != null ? value : 0.0
  const circumference = 2 * Math.PI * 28
  const offset = circumference * (1 - display)
  const color = value == null ? '#444444' : display >= 0.80 ? 'var(--green)' : display >= 0.65 ? 'var(--orange)' : 'var(--red)'
  return (
    <div className="gauge-wrap">
      <div className="gauge">
        <svg viewBox="0 0 64 64" width="64" height="64">
          <circle className="gauge-track" cx="32" cy="32" r="28" />
          <circle
            className="gauge-fill"
            cx="32" cy="32" r="28"
            style={{ strokeDasharray: circumference, strokeDashoffset: offset, stroke: color }}
          />
        </svg>
        <div className="gauge-text">
          {value != null ? display.toFixed(2) : '—'}
          <small>AUC</small>
        </div>
      </div>
      <div>
        <div style={{ fontSize: '26px', color, lineHeight: 1 }}>{value != null ? display.toFixed(2) : '—'}</div>
        <div style={{ fontSize: '11px', color: 'var(--text-dim)', letterSpacing: '0.06em', marginTop: '3px' }}>EVAL.LAST</div>
        <div style={{ fontSize: '11px', color, marginTop: '4px' }}>ROC-AUC SCORE</div>
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

function Cube({ arch }) {
  const faces = arch === 'transformer'
    ? [
        ['face-front', 'ATTN_HEAD'],
        ['face-back',  'EMBED_IN'],
        ['face-left',  'ENC_L1'],
        ['face-right', 'ENC_L2'],
        ['face-top',   'CLS_TOK'],
        ['face-bottom', 'DROPOUT'],
      ]
    : [
        ['face-front', 'ATTN_HEAD'],
        ['face-back',  'EMBED_IN'],
        ['face-left',  'LSTM_01'],
        ['face-right', 'LSTM_02'],
        ['face-top',   'DENSE_OUT'],
        ['face-bottom', 'DROPOUT'],
      ]
  return (
    <div className="cube-scene">
      <div className="cube">
        {faces.map(([cls, label]) => (
          <div key={cls} className={`face ${cls}`}>{label}</div>
        ))}
      </div>
    </div>
  )
}

// Generate a real-looking log line from system snapshot
function snapshotToLogs(snap, prev) {
  const ts = snap.timestamp
  const lines = []

  if (!prev || prev.model_exists !== snap.model_exists) {
    if (snap.model_exists) {
      lines.push(['info', `[${ts}] MODEL LOADED: ${snap.arch?.toUpperCase()}_BINARY (${(snap.model_params / 1e6).toFixed(2)}M params)`])
    } else {
      lines.push(['warn', `[${ts}] NO MODEL FOUND — run training first`])
    }
  }

  if (snap.device === 'cuda') {
    const pct = snap.gpu_util_pct
    const type = pct > 70 ? 'warn' : 'ok'
    lines.push([type, `[${ts}] GPU: ${snap.gpu_mem_used_gb}/${snap.gpu_mem_total_gb} GB  (${pct}% alloc)`])
  } else {
    lines.push(['dim', `[${ts}] DEVICE: cpu (no GPU detected)`])
  }

  if (snap.active_training_jobs?.length > 0) {
    lines.push(['info', `[${ts}] TRAINING IN PROGRESS — job ${snap.active_training_jobs[0]}`])
  }

  if (snap.ablation_fractions_done > 0) {
    lines.push(['dim', `[${ts}] ABLATION: ${snap.ablation_fractions_done}/6 fractions done — best AUC ${snap.ablation_best_auc}`])
  } else {
    lines.push(['dim', `[${ts}] ABLATION: no sweep data yet`])
  }

  return lines
}

export default function Overview() {
  const [logs, setLogs]       = useState([['dim', '> CONNECTING TO API...']])
  const [bestAuc, setBestAuc] = useState(null)
  const [modelArch, setModelArch] = useState('transformer')
  const [snap, setSnap]       = useState(null)
  const [archStats, setArchStats] = useState([])
  const prevSnapRef = useRef(null)
  const termRef = useRef(null)

  const poll = () => {
    Promise.all([
      fetch(`${API}/api/system`, { signal: AbortSignal.timeout(3000) }).then(r => r.json()).catch(() => null),
      fetch(`${API}/api/status`, { signal: AbortSignal.timeout(3000) }).then(r => r.json()).catch(() => null),
    ]).then(([sys, status]) => {
      if (sys) {
        const newLines = snapshotToLogs(sys, prevSnapRef.current)
        setLogs(prev => [...prev.slice(-80), ...newLines])
        prevSnapRef.current = sys
        setSnap(sys)
      }
      if (status?.best_auc != null) setBestAuc(status.best_auc)
      if (status?.arch) {
        const a = status.arch
        setModelArch(a)
        if (a === 'transformer') {
          setArchStats([
            ['INPUT DIM',   '13 physics-invariant features'],
            ['D_MODEL',     '128 — token embedding size'],
            ['ATTN HEADS',  '8 — multi-head self-attention'],
            ['ENC LAYERS',  '4 — Pre-LN transformer blocks'],
            ['FF DIM',      '512 — feedforward width'],
            ['POOLING',     'CLS token (learnable)'],
            ['TOTAL PARAMS', status.total_params ? `${(status.total_params / 1e6).toFixed(2)}M` : '—'],
          ])
        } else {
          setArchStats([
            ['INPUT DIM',   '13 physics-invariant features'],
            ['HIDDEN DIM',  '128 units'],
            ['LSTM LAYERS', '2'],
            ['TOTAL PARAMS', status.total_params ? `${(status.total_params / 1e6).toFixed(2)}M` : '—'],
          ])
        }
      }
      if (!sys && !status) {
        setLogs(prev => [...prev, ['warn', `[${new Date().toLocaleTimeString()}] API OFFLINE — start: uvicorn src.api.main:app --port 8000`]])
      }
    })
  }

  useEffect(() => {
    poll()
    const id = setInterval(poll, 5000)
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
      note: null,
      noteColor: 'var(--red)',
      borderColor: 'var(--red)',
      isSparkline: true,
    },
    {
      label: 'MODEL CONFIDENCE', sub: '',
      isGauge: true,
      borderColor: bestAuc != null ? (bestAuc >= 0.80 ? 'var(--green)' : 'var(--orange)') : 'var(--border)',
    },
  ]

  return (
    <div className="panel-content">
      <div className="g4" style={{ marginBottom: '16px' }}>
        {tiles.map(t => (
          <div key={t.label} className="tile" style={{ borderTopColor: t.borderColor }}>
            <div className="tile-label">
              {t.label}
              <span>{t.sub}</span>
            </div>
            {t.isGauge ? (
              <Gauge value={bestAuc} />
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

      <div className="g-left-right">
        {/* Left: architecture */}
        <div className="card">
          <div className="card-hdr">
            ARCHITECTURE TOPOLOGY
            <span className="badge">{modelArch.toUpperCase()}_BINARY</span>
          </div>

          <Cube arch={modelArch} />

          <div style={{ textAlign: 'center', marginTop: '8px' }}>
            <div style={{ fontSize: '13px', color: snap?.model_exists ? 'var(--green)' : '#555555', letterSpacing: '0.12em', fontWeight: 700 }}>
              {snap?.model_exists ? 'MODEL LOADED' : snap ? 'NO MODEL' : 'CHECKING...'}
            </div>
            {snap?.model_params && (
              <div style={{ fontSize: '11px', color: 'var(--text-dim)', marginTop: '4px', letterSpacing: '0.04em' }}>
                {(snap.model_params / 1e6).toFixed(2)}M PARAMETERS
              </div>
            )}
          </div>

          <div style={{ marginTop: '16px', borderTop: '1px solid var(--border)', paddingTop: '13px' }}>
            {archStats.map(([k, v]) => (
              <div key={k} className="stat-row">{k} <span>{v}</span></div>
            ))}
            {archStats.length === 0 && (
              <div style={{ color: 'var(--text-dim)', fontSize: '11px', letterSpacing: '0.06em' }}>API OFFLINE</div>
            )}
          </div>
        </div>

        {/* Right: live terminal */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
            <span style={{ fontSize: '11px', color: 'var(--text-dim)', letterSpacing: '0.08em', textTransform: 'uppercase' }}>
              SYS.LOG // POLL 5s
            </span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              {snap?.active_training_jobs?.length > 0 && (
                <span style={{ fontSize: '10px', color: 'var(--cyan)', fontFamily: 'var(--mono)', letterSpacing: '0.06em', animation: 'pulse 1.5s infinite' }}>
                  ● TRAINING
                </span>
              )}
              <div className="dots">
                <span className="dot dot-r" />
                <span className="dot dot-y" />
                <span className="dot dot-g" />
              </div>
            </div>
          </div>

          {/* GPU bar if available */}
          {snap?.gpu && (
            <div style={{ marginBottom: '10px', padding: '8px 10px', background: 'var(--bg3)', border: '1px solid var(--border)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: 'var(--text-dim)', fontFamily: 'var(--mono)', marginBottom: '5px' }}>
                <span>{snap.gpu}</span>
                <span style={{ color: snap.gpu_util_pct > 70 ? 'var(--orange)' : 'var(--green)' }}>
                  {snap.gpu_mem_used_gb}/{snap.gpu_mem_total_gb} GB
                </span>
              </div>
              <div style={{ height: '3px', background: 'var(--border2)' }}>
                <div style={{
                  height: '100%',
                  width: `${snap.gpu_util_pct}%`,
                  background: snap.gpu_util_pct > 70 ? 'var(--orange)' : 'var(--green)',
                  transition: 'width 1s ease',
                }} />
              </div>
            </div>
          )}

          <div
            ref={termRef}
            className="terminal"
            style={{ flex: 1, minHeight: '260px' }}
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
