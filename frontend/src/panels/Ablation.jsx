import { useEffect, useState } from 'react'
import {
  AreaChart, Area, LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts'

const API = 'http://localhost:8000'

// Shown until real data loads
const PLACEHOLDER = [
  { pct: 10,  steps: 58,   auc: null, lat: '—',       w: 7,  col: '#333333' },
  { pct: 20,  steps: 115,  auc: null, lat: '—',       w: 13, col: '#333333' },
  { pct: 30,  steps: 173,  auc: null, lat: '—',       w: 20, col: '#333333' },
  { pct: 40,  steps: 230,  auc: null, lat: '—',       w: 27, col: '#333333', hl: true },
  { pct: 60,  steps: 346,  auc: null, lat: '—',       w: 40, col: '#333333' },
  { pct: 90,  steps: 519,  auc: null, lat: '—',       w: 42, col: '#333333' },
  { pct: 100, steps: 576,  auc: null, lat: '—',       w: 45, col: '#333333' },
]

const tooltipStyle = {
  contentStyle: { background: '#0f0f0f', border: '1px solid #2a2a2a', fontSize: 10, fontFamily: 'Share Tech Mono, Courier New, monospace' },
  labelStyle: { color: '#888888' },
}

function aucColor(v) {
  if (v == null) return '#444444'
  if (v >= 0.80) return 'var(--green)'
  if (v >= 0.65) return 'var(--orange)'
  return 'var(--red)'
}

function rowColor(v, idx) {
  const COLORS = ['#ff5555', '#ff7700', '#ffaa00', '#e2e2e2', '#cccccc', '#aaaaaa', '#888888']
  return v != null ? aucColor(v) : COLORS[idx] || '#888888'
}

function inferLatency(steps) {
  // Rough estimate: ~0.078ms per timestep for transformer on GPU
  return `${(steps * 0.078).toFixed(1)} ms`
}

function computeBarWidth(auc, maxAuc) {
  if (auc == null || maxAuc == null) return 10
  return Math.round((auc / maxAuc) * 45)
}

export default function Ablation() {
  const [rows, setRows] = useState(PLACEHOLDER)
  const [rawData, setRawData] = useState([])   // full API response with epoch_history
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [modelType, setModelType] = useState('TRANSFORMER')
  const [expandedPct, setExpandedPct] = useState(null)  // which row is drilled into

  useEffect(() => {
    fetch(`${API}/api/ablation`)
      .then(r => {
        if (!r.ok) throw new Error(r.status === 404 ? 'pending' : 'error')
        return r.json()
      })
      .then(data => {
        setRawData(data)
        const maxAuc = Math.max(...data.map(d => d.test_auc))
        // Find optimal (best AUC / time tradeoff: biggest AUC gain before diminishing)
        const optIdx = findOptimalIdx(data)

        const mapped = data.map((d, i) => ({
          pct:   d.early_exit_pct,
          steps: Math.round(d.early_exit_pct * 5.76),  // 576 total steps * fraction
          auc:   d.test_auc,
          lat:   inferLatency(Math.round(d.early_exit_pct * 5.76)),
          w:     computeBarWidth(d.test_auc, maxAuc),
          col:   aucColor(d.test_auc),
          hl:    i === optIdx,
        }))
        setRows(mapped)
        if (data[0]?.model_type) {
          setModelType(data[0].model_type.toUpperCase())
        }
        setError(null)
      })
      .catch(err => {
        if (err.message === 'pending') {
          setError('pending')
        } else {
          setError('fetch_failed')
        }
      })
      .finally(() => setLoading(false))
  }, [])

  const realRows = rows.filter(r => r.auc != null)
  const chartData = (realRows.length ? realRows : rows).map(d => ({
    pct: `${d.pct}%`,
    auc: d.auc,
  }))

  const optimalRow = rows.find(r => r.hl) || rows[3]
  const bestAuc    = realRows.length ? Math.max(...realRows.map(r => r.auc)) : null
  const minViable  = realRows.length ? realRows.find(r => r.auc >= 0.60)?.auc : null
  const hasData    = realRows.length > 0

  return (
    <div className="panel-content">
      <div className="g-3-7">

        {/* Left */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
          <div className="finding">
            <div className="finding-title">KEY FINDING // ABL-X</div>
            <div className="finding-tag">
              {hasData ? 'REAL DATA' : loading ? 'LOADING...' : 'AWAITING RUN'}
            </div>
            <div className="finding-body">
              {hasData ? (
                <>
                  Model performance exhibits diminishing returns beyond 70% telemetry
                  utilization. Using only{' '}
                  <strong style={{ color: 'var(--cyan)' }}>{optimalRow?.pct}%</strong> of
                  trajectory data yields AUC {optimalRow?.auc?.toFixed(3)} while reducing
                  computational load by{' '}
                  <strong style={{ color: 'var(--cyan)' }}>{100 - (optimalRow?.pct || 40)}%</strong>.
                </>
              ) : error === 'pending' ? (
                <>Ablation sweep not yet run. See command below.</>
              ) : (
                <>Connecting to API…</>
              )}
            </div>
            {hasData && (
              <div className="finding-action">
                ACTION: Recommend pruning at {optimalRow?.pct}%.&nbsp;
                <strong>AUTH: ORBITGUARD</strong>
              </div>
            )}
            {error === 'pending' && (
              <div style={{ marginTop: '10px', fontSize: '11px', color: 'var(--orange)', fontFamily: 'var(--mono)', lineHeight: 1.8 }}>
                Run to generate data:<br />
                <span style={{ color: 'var(--cyan)' }}>
                  python3 -m src.ml.ablation \<br />
                  &nbsp;&nbsp;--data data/merged/missions.parquet \<br />
                  &nbsp;&nbsp;--epochs 30
                </span>
              </div>
            )}
          </div>

          <div className="card">
            <div className="card-hdr">ABLATION CURVE</div>
            <ResponsiveContainer width="100%" height={200}>
              <AreaChart data={chartData} margin={{ top: 8, right: 8, bottom: 0, left: -10 }}>
                <defs>
                  <linearGradient id="gAbl" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor="#e2e2e2" stopOpacity={0.18} />
                    <stop offset="95%" stopColor="#e2e2e2" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1a1a1a" />
                <XAxis dataKey="pct" stroke="#333333" tick={{ fill: '#555555', fontSize: 9, fontFamily: 'Share Tech Mono' }} />
                <YAxis domain={[0.45, 1.0]} stroke="#333333" tick={{ fill: '#555555', fontSize: 9, fontFamily: 'Share Tech Mono' }} tickFormatter={v => v.toFixed(2)} />
                <Tooltip {...tooltipStyle} formatter={v => v != null ? v.toFixed(3) : '—'} />
                {optimalRow?.auc && (
                  <ReferenceLine
                    y={optimalRow.auc}
                    stroke="#e2e2e233"
                    strokeDasharray="4 3"
                    label={{ value: 'OPTIMAL', fill: '#aaaaaa', fontSize: 8, position: 'insideTopRight' }}
                  />
                )}
                <Area
                  type="monotone" dataKey="auc"
                  stroke={hasData ? '#aaaaaa' : '#333333'}
                  fill="url(#gAbl)" strokeWidth={2.5}
                  dot={{ r: 4, fill: '#e2e2e2', stroke: '#0f0f0f', strokeWidth: 1 }}
                  name="AUC"
                  connectNulls={false}
                />
              </AreaChart>
            </ResponsiveContainer>

            <div style={{ marginTop: '12px', borderTop: '1px solid var(--border)', paddingTop: '12px' }}>
              {optimalRow && [
                [`${optimalRow.pct}% telemetry`, `${100 - optimalRow.pct}% sim time saved`, 'var(--cyan)'],
                ['100% telemetry', '0% (baseline)', 'var(--text-dim)'],
              ].map(([k, v, c]) => (
                <div key={k} className="stat-row">
                  {k} <span style={{ color: c }}>{v}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right: table */}
        <div className="card">
          <div className="card-hdr">
            EARLY-EXIT SWEEP RESULTS
            <span className="badge">BINARY CLASSIFICATION · {modelType}</span>
          </div>

          {!hasData && (
            <div style={{ padding: '20px 0', textAlign: 'center', color: 'var(--text-dim)', fontSize: '12px', letterSpacing: '0.08em' }}>
              {loading ? 'FETCHING RESULTS...' : error === 'pending' ? 'NO ABLATION DATA YET — RUN SWEEP FIRST' : 'COULD NOT REACH API'}
            </div>
          )}

          <table className="abl-table">
            <thead>
              <tr>
                <th>RETENTION %</th>
                <th>TIMESTEPS</th>
                <th>AUC SCORE</th>
                <th>EST. LATENCY</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((d, i) => {
                const epochHistory = rawData.find(r => r.early_exit_pct === d.pct)?.epoch_history || []
                const isExpanded = expandedPct === d.pct && epochHistory.length > 0
                return [
                  <tr
                    key={d.pct}
                    className={d.hl ? 'hl' : ''}
                    style={{ cursor: epochHistory.length ? 'pointer' : 'default' }}
                    onClick={() => epochHistory.length && setExpandedPct(isExpanded ? null : d.pct)}
                    title={epochHistory.length ? 'Click to see training curve' : ''}
                  >
                    <td>
                      {d.pct}%{d.hl ? ' (Rec)' : ''}
                      {epochHistory.length > 0 && (
                        <span style={{ marginLeft: '6px', fontSize: '9px', color: 'var(--text-dim)' }}>
                          {isExpanded ? '▲' : '▼'}
                        </span>
                      )}
                    </td>
                    <td>{d.steps.toLocaleString()}</td>
                    <td style={{ color: d.auc != null ? aucColor(d.auc) : '#444444' }}>
                      {d.auc != null ? d.auc.toFixed(3) : '—'}
                    </td>
                    <td>
                      <div className="bar-wrap">
                        <div className="bar-track">
                          <div className="bar-fill" style={{ width: `${d.w}%`, background: rowColor(d.auc, i) }} />
                        </div>
                        <span className="bar-val" style={{ color: rowColor(d.auc, i) }}>{d.lat}</span>
                      </div>
                    </td>
                  </tr>,
                  isExpanded && (
                    <tr key={`${d.pct}-expand`}>
                      <td colSpan={4} style={{ padding: '10px 6px', background: 'var(--bg3)' }}>
                        <div style={{ fontSize: '10px', color: 'var(--text-dim)', fontFamily: 'var(--mono)', marginBottom: '6px', letterSpacing: '0.06em' }}>
                          TRAINING CURVE — {d.pct}% RETENTION ({epochHistory.length} epochs)
                        </div>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                          <div>
                            <div style={{ fontSize: '9px', color: 'var(--text-dim)', fontFamily: 'var(--mono)', marginBottom: '4px' }}>VAL LOSS</div>
                            <ResponsiveContainer width="100%" height={90}>
                              <LineChart data={epochHistory} margin={{ top: 2, right: 4, bottom: 2, left: -20 }}>
                                <CartesianGrid strokeDasharray="2 2" stroke="#1a1a1a" />
                                <XAxis dataKey="epoch" tick={{ fill: '#444', fontSize: 8 }} />
                                <YAxis tick={{ fill: '#444', fontSize: 8 }} domain={['auto', 'auto']} />
                                <Tooltip contentStyle={{ background: '#0f0f0f', border: '1px solid #2a2a2a', fontSize: 9 }} formatter={v => v.toFixed(4)} />
                                <Line type="monotone" dataKey="val_loss" stroke="#888888" strokeWidth={1.5} dot={false} name="Val Loss" isAnimationActive={false} />
                              </LineChart>
                            </ResponsiveContainer>
                          </div>
                          <div>
                            <div style={{ fontSize: '9px', color: 'var(--text-dim)', fontFamily: 'var(--mono)', marginBottom: '4px' }}>AUC OVER EPOCHS</div>
                            <ResponsiveContainer width="100%" height={90}>
                              <LineChart data={epochHistory} margin={{ top: 2, right: 4, bottom: 2, left: -20 }}>
                                <CartesianGrid strokeDasharray="2 2" stroke="#1a1a1a" />
                                <XAxis dataKey="epoch" tick={{ fill: '#444', fontSize: 8 }} />
                                <YAxis tick={{ fill: '#444', fontSize: 8 }} domain={[0.5, 1.0]} tickFormatter={v => v.toFixed(2)} />
                                <Tooltip contentStyle={{ background: '#0f0f0f', border: '1px solid #2a2a2a', fontSize: 9 }} formatter={v => v.toFixed(4)} />
                                <Line type="monotone" dataKey="auc" stroke={aucColor(d.auc)} strokeWidth={1.5} dot={false} name="AUC" isAnimationActive={false} />
                              </LineChart>
                            </ResponsiveContainer>
                          </div>
                        </div>
                      </td>
                    </tr>
                  ),
                ]
              })}
            </tbody>
          </table>

          {/* Summary metrics */}
          <div style={{ marginTop: '20px', display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '10px' }}>
            {[
              { label: 'BEST AUC',    value: bestAuc ? bestAuc.toFixed(3) : '—',          sub: '100% telemetry',          color: 'var(--green)' },
              { label: 'OPTIMAL AUC', value: optimalRow?.auc ? optimalRow.auc.toFixed(3) : '—', sub: `At ${optimalRow?.pct || 40}% (Rec)`, color: 'var(--cyan)' },
              { label: 'MIN VIABLE',  value: minViable ? minViable.toFixed(3) : '—',       sub: 'AUC ≥ 0.60 cutoff',      color: 'var(--orange)' },
            ].map(m => (
              <div key={m.label} style={{ background: 'var(--bg3)', border: '1px solid var(--border)', padding: '12px 14px' }}>
                <div style={{ fontSize: '11px', color: 'var(--text-dim)', letterSpacing: '0.06em', marginBottom: '6px' }}>{m.label}</div>
                <div style={{ fontSize: '22px', color: m.color }}>{m.value}</div>
                <div style={{ fontSize: '11px', color: 'var(--text-dim)', marginTop: '4px' }}>{m.sub}</div>
              </div>
            ))}
          </div>

          {hasData && optimalRow && (
            <div style={{ marginTop: '16px', background: 'var(--bg3)', border: '1px solid var(--border)', borderLeft: '2px solid var(--cyan)', padding: '12px 14px' }}>
              <div style={{ fontSize: '11px', color: 'var(--text-dim)', letterSpacing: '0.06em', marginBottom: '8px' }}>RESEARCH IMPLICATION</div>
              <div style={{ fontSize: '13px', color: 'var(--text)', lineHeight: 1.8 }}>
                A {modelType} monitoring only the{' '}
                <span style={{ color: 'var(--cyan)' }}>first {optimalRow.pct}% of a trajectory</span> achieves
                AUC {optimalRow.auc?.toFixed(3)} — sufficient to prune failing missions,
                saving up to{' '}
                <span style={{ color: 'var(--green)' }}>{100 - optimalRow.pct}% of simulation compute time</span> per Monte Carlo run.
              </div>
            </div>
          )}
        </div>

      </div>
    </div>
  )
}

// Find index of best tradeoff: largest AUC gain per % of trajectory used
function findOptimalIdx(data) {
  if (data.length < 2) return 0
  let bestScore = -Infinity
  let bestIdx = 0
  for (let i = 1; i < data.length; i++) {
    const aucGain = data[i].test_auc - data[i - 1].test_auc
    const pctCost = data[i].early_exit_pct - data[i - 1].early_exit_pct
    const score = aucGain / pctCost  // AUC per % of trajectory
    if (score > bestScore) {
      bestScore = score
      bestIdx = i
    }
  }
  return bestIdx
}
