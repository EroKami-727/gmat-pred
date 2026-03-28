import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceDot, ReferenceLine,
} from 'recharts'

const ABL = [
  { pct: 10,  steps: 58,   auc: 0.521, lat: '3.1 ms',  w: 7,  col: '#ff5555' },
  { pct: 20,  steps: 115,  auc: 0.612, lat: '6.0 ms',  w: 13, col: '#ff7700' },
  { pct: 30,  steps: 173,  auc: 0.694, lat: '9.2 ms',  w: 20, col: '#ffaa00' },
  { pct: 40,  steps: 230,  auc: 0.745, lat: '12.4 ms', w: 27, col: '#e2e2e2', hl: true },
  { pct: 60,  steps: 346,  auc: 0.812, lat: '24.1 ms', w: 40, col: '#cccccc' },
  { pct: 90,  steps: 519,  auc: 0.842, lat: '38.7 ms', w: 42, col: '#aaaaaa' },
  { pct: 100, steps: 576,  auc: 0.860, lat: '45.2 ms', w: 45, col: '#888888' },
]

const chartData = ABL.map(d => ({ pct: `${d.pct}%`, auc: d.auc }))

const tooltipStyle = {
  contentStyle: { background: '#0f0f0f', border: '1px solid #2a2a2a', fontSize: 10, fontFamily: 'Share Tech Mono, Courier New, monospace' },
  labelStyle: { color: '#888888' },
}

function AucColor(v) {
  if (v >= 0.80) return 'var(--green)'
  if (v >= 0.65) return 'var(--orange)'
  return 'var(--red)'
}

export default function Ablation() {
  return (
    <div className="panel-content">
      <div className="g-3-7">

        {/* Left */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
          <div className="finding">
            <div className="finding-title">KEY FINDING // ABL-X</div>
            <div className="finding-tag">EXP. RUN: 2026-X</div>
            <div className="finding-body">
              Model performance exhibits diminishing returns beyond 70% telemetry
              utilization. Using only <strong style={{ color: 'var(--cyan)' }}>40%</strong> of
              trajectory data yields a 0.115 AUC drop while reducing computational
              load by <strong style={{ color: 'var(--cyan)' }}>60%</strong>. Optimal
              early-exit balances prediction quality with simulation pruning efficiency.
            </div>
            <div className="finding-action">
              ACTION: Recommend pruning at 40%.&nbsp;
              <strong>AUTH: ORBITGUARD</strong>
            </div>
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
                <YAxis domain={[0.45, 0.92]} stroke="#333333" tick={{ fill: '#555555', fontSize: 9, fontFamily: 'Share Tech Mono' }} tickFormatter={v => v.toFixed(2)} />
                <Tooltip {...tooltipStyle} formatter={v => v.toFixed(3)} />
                <ReferenceLine y={0.745} stroke="#e2e2e233" strokeDasharray="4 3" label={{ value: 'OPTIMAL', fill: '#aaaaaa', fontSize: 8, position: 'insideTopRight' }} />
                <Area type="monotone" dataKey="auc" stroke="#aaaaaa" fill="url(#gAbl)" strokeWidth={2.5} dot={{ r: 4, fill: '#e2e2e2', stroke: '#0f0f0f', strokeWidth: 1 }} name="AUC" />
              </AreaChart>
            </ResponsiveContainer>

            <div style={{ marginTop: '12px', borderTop: '1px solid var(--border)', paddingTop: '12px' }}>
              {[
                ['40% telemetry', '60% sim time saved', 'var(--cyan)'],
                ['60% telemetry', '40% sim time saved', 'var(--green)'],
                ['20% telemetry', '80% sim time saved', 'var(--orange)'],
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
            <span className="badge">BINARY CLASSIFICATION · LSTM</span>
          </div>

          <table className="abl-table">
            <thead>
              <tr>
                <th>RETENTION %</th>
                <th>TIMESTEPS</th>
                <th>AUC SCORE</th>
                <th>INFERENCE LATENCY</th>
              </tr>
            </thead>
            <tbody>
              {ABL.map(d => (
                <tr key={d.pct} className={d.hl ? 'hl' : ''}>
                  <td>{d.pct}%{d.hl ? ' (Rec)' : ''}</td>
                  <td>{d.steps.toLocaleString()}</td>
                  <td style={{ color: AucColor(d.auc) }}>{d.auc.toFixed(3)}</td>
                  <td>
                    <div className="bar-wrap">
                      <div className="bar-track">
                        <div className="bar-fill" style={{ width: `${d.w}%`, background: d.col }} />
                      </div>
                      <span className="bar-val" style={{ color: d.col }}>{d.lat}</span>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {/* Summary metrics */}
          <div style={{ marginTop: '20px', display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '10px' }}>
            {[
              { label: 'BEST AUC',        value: '0.860', sub: '100% telemetry', color: 'var(--green)' },
              { label: 'OPTIMAL AUC',     value: '0.745', sub: 'At 40% (Rec)',   color: 'var(--cyan)' },
              { label: 'MIN VIABLE',      value: '0.612', sub: 'At 20% cutoff',  color: 'var(--orange)' },
            ].map(m => (
              <div key={m.label} style={{ background: 'var(--bg3)', border: '1px solid var(--border)', padding: '12px 14px' }}>
                <div style={{ fontSize: '11px', color: 'var(--text-dim)', letterSpacing: '0.06em', marginBottom: '6px' }}>{m.label}</div>
                <div style={{ fontSize: '22px', color: m.color }}>{m.value}</div>
                <div style={{ fontSize: '11px', color: 'var(--text-dim)', marginTop: '4px' }}>{m.sub}</div>
              </div>
            ))}
          </div>

          <div style={{ marginTop: '16px', background: 'var(--bg3)', border: '1px solid var(--border)', borderLeft: '2px solid var(--cyan)', padding: '12px 14px' }}>
            <div style={{ fontSize: '11px', color: 'var(--text-dim)', letterSpacing: '0.06em', marginBottom: '8px' }}>RESEARCH IMPLICATION</div>
            <div style={{ fontSize: '13px', color: 'var(--text)', lineHeight: 1.8 }}>
              An LSTM monitoring only the <span style={{ color: 'var(--cyan)' }}>first 40% of a trajectory</span> achieves
              AUC 0.745 — sufficient to prune failing missions with &lt;25% FPR, saving
              up to <span style={{ color: 'var(--green)' }}>60% of simulation compute time</span> per Monte Carlo run.
            </div>
          </div>
        </div>

      </div>
    </div>
  )
}
