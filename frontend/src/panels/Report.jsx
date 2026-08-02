import { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, ReferenceLine,
} from 'recharts'
import { API } from '../lib/api.js'

// ── Data ─────────────────────────────────────────────────────────────────────

const MODEL_COMPARISON = [
  { name: 'Majority',         f1: 0.000, auc: 0.500 },
  { name: 'Energy\nThreshold',f1: 0.536, auc: 0.535 },
  { name: 'XGBoost\nInitial', f1: 0.974, auc: 0.998 },
  { name: 'XGBoost\nSummary', f1: 0.992, auc: 1.000 },
  { name: 'Transformer\n(calibrated)', f1: 0.921, auc: 0.984 },
]

// HISTORICAL — these are leave-one-target-out results for the SINGLE GLOBAL
// model that production no longer uses. Kept because the experiment is real and
// it is the evidence that motivated the per-planet split. Current production
// numbers are fetched live from /api/simulator/model_report.
const LOTO_TARGETS = [
  { name: 'Saturn',  auc: 1.000, f1: 1.000, status: 'good',    note: 'Perfect generalisation' },
  { name: 'Neptune', auc: 1.000, f1: 1.000, status: 'good',    note: 'Perfect generalisation' },
  { name: 'Jupiter', auc: 0.998, f1: 0.997, status: 'good',    note: 'Near-perfect' },
  { name: 'Uranus',  auc: 0.992, f1: 0.000, status: 'calib',   note: 'High AUC — threshold off' },
  { name: 'Venus',   auc: 0.856, f1: 0.000, status: 'calib',   note: 'Ranking OK — threshold off' },
  { name: 'Mars',    auc: 0.509, f1: 0.000, status: 'fail',    note: 'Regime shift (dist features)' },
  { name: 'Mercury', auc: 0.496, f1: 0.000, status: 'fail',    note: 'Regime shift (dist features)' },
  { name: 'Moon',    auc: 0.296, f1: 0.000, status: 'fail',    note: 'Hardest — structurally different' },
]

const CALIBRATION_DATA = [
  { label: 'Default\n(@0.5)',     f1: 0.838, ece: 0.052 },
  { label: 'Threshold\nTuned',   f1: 0.921, ece: 0.052 },
  { label: 'Isotonic\nCalibrated', f1: 0.919, ece: 0.0045 },
]

const DOMAIN_GEN = [
  { name: 'Moon',    unbal: 0.577, bal: 0.826, delta: '+0.249', good: true },
  { name: 'Mars',    unbal: 0.899, bal: 0.923, delta: '+0.024', good: true },
  { name: 'Jupiter', unbal: 0.965, bal: 0.976, delta: '+0.011', good: true },
  { name: 'Mercury', unbal: 0.531, bal: 0.500, delta: '-0.031', good: false },
  { name: 'Saturn',  unbal: 0.997, bal: 0.997, delta: '0.000',  good: null },
  { name: 'Neptune', unbal: 0.998, bal: 0.999, delta: '+0.001', good: null },
  { name: 'Uranus',  unbal: 0.981, bal: 0.977, delta: '-0.004', good: false },
  { name: 'Venus',   unbal: 0.902, bal: 0.000, delta: '-0.902', good: false },
]

const JIT_DATA = [
  { name: 'Normal\n(estimated)', hours: 20, col: '#555555' },
  { name: 'Numba JIT', hours: 0.85, col: 'var(--cyan)' },
]

// ── Helpers ───────────────────────────────────────────────────────────────────

const TT = {
  contentStyle: { background: '#0f0f0f', border: '1px solid #2a2a2a', fontSize: 10, fontFamily: 'Share Tech Mono, monospace' },
  labelStyle: { color: '#888888' },
}

// Live production metrics — read from each model's meta.json via the API so
// this table cannot go stale the way the hardcoded LOTO one did.
function ProductionTable() {
  const [rows, setRows] = useState(null)
  const [frac, setFrac] = useState(0.4)
  const [err,  setErr]  = useState(null)

  useEffect(() => {
    fetch(`${API}/api/simulator/model_report`)
      .then(r => r.json())
      .then(d => { setRows(d.planets || []); setFrac(d.operating_frac ?? 0.4) })
      .catch(e => setErr(String(e)))
  }, [])

  if (err)   return <Card style={{ padding: '14px 20px' }}>
    <span style={{ fontSize: '10px', color: 'var(--red)', fontFamily: 'Share Tech Mono,monospace' }}>
      API OFFLINE — {err}</span></Card>
  if (!rows) return <Card style={{ padding: '14px 20px' }}>
    <span style={{ fontSize: '10px', color: '#555', fontFamily: 'Share Tech Mono,monospace' }}>
      LOADING…</span></Card>

  const num = (v, d = 3) => (v == null ? '—' : Number(v).toFixed(d))
  const col = (v, hi = 0.99, mid = 0.95) =>
    v == null ? '#555' : v >= hi ? 'var(--green)' : v >= mid ? '#ffaa00' : 'var(--red)'

  return (
    <Card style={{ padding: '14px 0' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '11px', fontFamily: 'Share Tech Mono,monospace' }}>
        <thead>
          <tr style={{ color: '#555', fontSize: '10px', letterSpacing: '0.08em' }}>
            <th style={{ padding: '4px 20px', textAlign: 'left',  fontWeight: 400 }}>TARGET</th>
            <th style={{ padding: '4px 10px', textAlign: 'right', fontWeight: 400 }}>AUC</th>
            <th style={{ padding: '4px 10px', textAlign: 'right', fontWeight: 400 }}>RECALL</th>
            <th style={{ padding: '4px 10px', textAlign: 'right', fontWeight: 400 }}>PREC</th>
            <th style={{ padding: '4px 10px', textAlign: 'right', fontWeight: 400 }}>F1</th>
            <th style={{ padding: '4px 10px', textAlign: 'right', fontWeight: 400 }}>MODE</th>
            <th style={{ padding: '4px 20px 4px 10px', textAlign: 'right', fontWeight: 400 }}>THR</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(r => (
            <tr key={r.planet} style={{ borderTop: '1px solid #1a1a1a' }}>
              <td style={{ padding: '7px 20px', color: '#ccc' }}>
                {r.planet.charAt(0).toUpperCase() + r.planet.slice(1)}
                {r.has_assist && <span style={{ fontSize: '8px', color: '#4a7', marginLeft: '7px',
                  border: '1px solid #2a4a35', padding: '1px 3px' }}>+TREE</span>}
                {!r.trained && <span style={{ fontSize: '8px', color: 'var(--red)', marginLeft: '7px' }}>NOT TRAINED</span>}
              </td>
              <td style={{ padding: '7px 10px', textAlign: 'right', color: col(r.auc) }}>{num(r.auc, 4)}</td>
              <td style={{ padding: '7px 10px', textAlign: 'right', color: col(r.recall) }}>{num(r.recall, 4)}</td>
              <td style={{ padding: '7px 10px', textAlign: 'right', color: col(r.precision) }}>{num(r.precision, 4)}</td>
              <td style={{ padding: '7px 10px', textAlign: 'right', color: col(r.f1) }}>{num(r.f1, 4)}</td>
              <td style={{ padding: '7px 10px', textAlign: 'right', color: col(r.mode_acc, 0.95, 0.90) }}>{num(r.mode_acc, 3)}</td>
              <td style={{ padding: '7px 20px 7px 10px', textAlign: 'right', color: '#888' }}>{num(r.threshold, 3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div style={{ fontSize: '10px', color: '#555', marginTop: '10px', padding: '0 20px', lineHeight: 1.6 }}>
        One model per target, evaluated on its own held-out split at {Math.round(frac * 100)}% of
        trajectory observed. MODE = accuracy of predicting <em>how</em> a mission fails.
        +TREE marks planets where a gradient-boosted assist is fused at the decision window to
        recover rare failure modes the sequence model misses.
      </div>
    </Card>
  )
}

function Section({ title, children }) {
  return (
    <div style={{ marginBottom: '36px' }}>
      <div style={{
        fontSize: '11px', letterSpacing: '0.12em', color: 'var(--cyan)',
        borderBottom: '1px solid var(--border)', paddingBottom: '8px', marginBottom: '18px',
        textTransform: 'uppercase',
      }}>
        {title}
      </div>
      {children}
    </div>
  )
}

function Card({ children, style }) {
  return (
    <div style={{
      background: 'var(--bg2)', border: '1px solid var(--border)',
      borderRadius: '4px', padding: '18px 20px', ...style,
    }}>
      {children}
    </div>
  )
}

function StatusDot({ status }) {
  const colors = { good: 'var(--green)', calib: '#ffaa00', fail: 'var(--red)' }
  return (
    <span style={{
      display: 'inline-block', width: '8px', height: '8px', borderRadius: '50%',
      background: colors[status] || '#555', boxShadow: `0 0 5px ${colors[status] || '#555'}`,
      flexShrink: 0,
    }} />
  )
}

// ── Chart: model F1 ───────────────────────────────────────────────────────────

function ModelChart() {
  const colors = ['#444444', '#666666', '#aaaaaa', '#cccccc', 'var(--cyan)']
  return (
    <Card>
      <div style={{ fontSize: '12px', color: 'var(--text-dim)', marginBottom: '14px' }}>
        F1 Score — random split, early exit 40%
      </div>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={MODEL_COMPARISON} barCategoryGap="30%" margin={{ left: -20, right: 10, bottom: 0 }}>
          <CartesianGrid strokeDasharray="2 4" stroke="#222" vertical={false} />
          <XAxis dataKey="name" tick={{ fontSize: 9, fill: '#888', fontFamily: 'Share Tech Mono,monospace' }} />
          <YAxis domain={[0, 1]} tick={{ fontSize: 9, fill: '#888' }} />
          <Tooltip {...TT} formatter={v => [v.toFixed(3), 'F1']} />
          <ReferenceLine y={1} stroke="#333" strokeDasharray="2 4" />
          <Bar dataKey="f1" radius={[2, 2, 0, 0]}>
            {MODEL_COMPARISON.map((_, i) => <Cell key={i} fill={colors[i]} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </Card>
  )
}

// ── Chart: calibration ────────────────────────────────────────────────────────

function CalibChart() {
  return (
    <Card>
      <div style={{ fontSize: '12px', color: 'var(--text-dim)', marginBottom: '14px' }}>
        Transformer calibration — F1 and ECE (lower is better)
      </div>
      <ResponsiveContainer width="100%" height={180}>
        <BarChart data={CALIBRATION_DATA} barCategoryGap="30%" margin={{ left: -20, right: 10 }}>
          <CartesianGrid strokeDasharray="2 4" stroke="#222" vertical={false} />
          <XAxis dataKey="label" tick={{ fontSize: 9, fill: '#888', fontFamily: 'Share Tech Mono,monospace' }} />
          <YAxis domain={[0, 1]} tick={{ fontSize: 9, fill: '#888' }} />
          <Tooltip {...TT} />
          <Bar dataKey="f1" name="F1" fill="#aaaaaa" radius={[2, 2, 0, 0]} />
          <Bar dataKey="ece" name="ECE" fill="#ff5555" radius={[2, 2, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
      <div style={{ fontSize: '10px', color: '#888', marginTop: '10px', lineHeight: 1.6 }}>
        Threshold tuning alone: F1 <span style={{ color: 'var(--cyan)' }}>0.838 → 0.921</span>. Isotonic calibration: ECE{' '}
        <span style={{ color: 'var(--cyan)' }}>0.052 → 0.0045</span> (11.5× reduction).
      </div>
    </Card>
  )
}

// ── Chart: JIT speedup ────────────────────────────────────────────────────────

function JitChart() {
  return (
    <Card>
      <div style={{ fontSize: '12px', color: 'var(--text-dim)', marginBottom: '14px' }}>
        Neptune dataset generation — hours (10,000 missions)
      </div>
      <ResponsiveContainer width="100%" height={160}>
        <BarChart data={JIT_DATA} barCategoryGap="40%" margin={{ left: -20, right: 10 }}>
          <CartesianGrid strokeDasharray="2 4" stroke="#222" vertical={false} />
          <XAxis dataKey="name" tick={{ fontSize: 9, fill: '#888', fontFamily: 'Share Tech Mono,monospace' }} />
          <YAxis tick={{ fontSize: 9, fill: '#888' }} unit="h" />
          <Tooltip {...TT} formatter={v => [`${v.toFixed(2)} h`, 'Time']} />
          <Bar dataKey="hours" radius={[2, 2, 0, 0]}>
            {JIT_DATA.map((d, i) => <Cell key={i} fill={d.col} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div style={{ fontSize: '10px', color: '#888', marginTop: '10px' }}>
        Numba @njit JIT compilation on RK4 hot loop:{' '}
        <span style={{ color: 'var(--cyan)' }}>~26–37× speedup</span>. 51 min vs ~20 hours.
      </div>
    </Card>
  )
}

// ── LOTO table ────────────────────────────────────────────────────────────────

function LotoTable() {
  const statusLabel = { good: 'OK', calib: 'THRESHOLD', fail: 'FAIL' }
  const statusColor = { good: 'var(--green)', calib: '#ffaa00', fail: 'var(--red)' }
  return (
    <Card style={{ padding: '14px 0' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '11px', fontFamily: 'Share Tech Mono,monospace' }}>
        <thead>
          <tr style={{ color: '#555', fontSize: '10px', letterSpacing: '0.08em' }}>
            <th style={{ padding: '4px 20px', textAlign: 'left', fontWeight: 400 }}>TARGET</th>
            <th style={{ padding: '4px 10px', textAlign: 'right', fontWeight: 400 }}>AUC</th>
            <th style={{ padding: '4px 10px', textAlign: 'right', fontWeight: 400 }}>F1@0.5</th>
            <th style={{ padding: '4px 10px', textAlign: 'left', fontWeight: 400 }}>STATUS</th>
            <th style={{ padding: '4px 20px 4px 10px', textAlign: 'left', fontWeight: 400 }}>NOTE</th>
          </tr>
        </thead>
        <tbody>
          {LOTO_TARGETS.map(t => (
            <tr key={t.name} style={{ borderTop: '1px solid #1a1a1a' }}>
              <td style={{ padding: '7px 20px', color: '#ccc', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <StatusDot status={t.status} />
                {t.name}
              </td>
              <td style={{ padding: '7px 10px', textAlign: 'right', color: t.auc >= 0.9 ? 'var(--green)' : t.auc >= 0.7 ? '#ffaa00' : 'var(--red)' }}>
                {t.auc.toFixed(3)}
              </td>
              <td style={{ padding: '7px 10px', textAlign: 'right', color: t.f1 > 0.5 ? 'var(--green)' : '#666' }}>
                {t.f1.toFixed(3)}
              </td>
              <td style={{ padding: '7px 10px', color: statusColor[t.status], fontSize: '10px', letterSpacing: '0.06em' }}>
                {statusLabel[t.status]}
              </td>
              <td style={{ padding: '7px 20px 7px 10px', color: '#555', fontSize: '10px' }}>
                {t.note}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div style={{ padding: '10px 20px 0', fontSize: '10px', color: '#555', lineHeight: 1.7 }}>
        <span style={{ color: 'var(--green)' }}>● OK</span> — generalises well &nbsp;
        <span style={{ color: '#ffaa00' }}>● THRESHOLD</span> — AUC fine, just needs threshold tuning &nbsp;
        <span style={{ color: 'var(--red)' }}>● FAIL</span> — physically different regime, model doesn't transfer
      </div>
    </Card>
  )
}

// ── Domain generalisation table ───────────────────────────────────────────────

function DomainGenTable() {
  return (
    <Card style={{ padding: '14px 0' }}>
      <div style={{ padding: '0 20px 12px', fontSize: '12px', color: 'var(--text-dim)' }}>
        F1 score per target — unbalanced vs balanced sampling (both 30 epochs)
      </div>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '11px', fontFamily: 'Share Tech Mono,monospace' }}>
        <thead>
          <tr style={{ color: '#555', fontSize: '10px', letterSpacing: '0.08em' }}>
            <th style={{ padding: '4px 20px', textAlign: 'left', fontWeight: 400 }}>TARGET</th>
            <th style={{ padding: '4px 10px', textAlign: 'right', fontWeight: 400 }}>UNBALANCED</th>
            <th style={{ padding: '4px 10px', textAlign: 'right', fontWeight: 400 }}>BALANCED</th>
            <th style={{ padding: '4px 20px 4px 10px', textAlign: 'right', fontWeight: 400 }}>DELTA</th>
          </tr>
        </thead>
        <tbody>
          {DOMAIN_GEN.map(r => (
            <tr key={r.name} style={{ borderTop: '1px solid #1a1a1a' }}>
              <td style={{ padding: '7px 20px', color: '#ccc' }}>{r.name}</td>
              <td style={{ padding: '7px 10px', textAlign: 'right', color: '#888' }}>{r.unbal.toFixed(3)}</td>
              <td style={{ padding: '7px 10px', textAlign: 'right', color: r.bal === 0 ? 'var(--red)' : '#888' }}>{r.bal.toFixed(3)}</td>
              <td style={{ padding: '7px 20px 7px 10px', textAlign: 'right',
                color: r.good === true ? 'var(--green)' : r.good === false ? 'var(--red)' : '#555',
                fontWeight: r.good !== null ? 600 : 400,
              }}>
                {r.delta}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div style={{ padding: '10px 20px 0', fontSize: '10px', color: '#555', lineHeight: 1.7 }}>
        4 weak targets were upweighted 2× during training (Moon, Mars, Mercury, Venus). Moon improved greatly; Venus collapsed to F1=0. Net aggregate effect was negative.
      </div>
    </Card>
  )
}

// ── Timeline ──────────────────────────────────────────────────────────────────

const TIMELINE = [
  {
    who: 'EARLIER',
    color: '#aaaaaa',
    items: [
      'Built the GMAT simulation pipeline and feature engineering (13 physics-invariant features)',
      'Generated 10,000 Moon missions — first dataset',
      'Trained original Transformer and LSTM models on Moon data',
      'Established baseline results: AUC 0.936, F1 0.745',
    ],
  },
  {
    who: 'THIS SESSION',
    color: 'var(--cyan)',
    items: [
      'Generated 7 more planet datasets (Mercury → Neptune) — 70,000 more missions',
      'Neptune took 37 years simulated time per mission — built Numba JIT compiler (26–37× faster, 51 min vs ~20 h)',
      'Merged all 8 datasets → 80,000 missions total on local hardware',
      'Re-trained Transformer on full 8-planet data — 50 epochs, AUC 0.984',
      'Ran 8 paper-grade evaluations: calibration, multi-seed CIs, formal ablation, error analysis, domain generalisation, LOTO, parameter holdout, calibration plots',
    ],
  },
]

function Timeline() {
  return (
    <div style={{ display: 'flex', gap: '16px', flexDirection: 'column' }}>
      {TIMELINE.map(block => (
        <Card key={block.who} style={{ borderLeftWidth: '3px', borderLeftColor: block.color }}>
          <div style={{ fontSize: '11px', letterSpacing: '0.1em', color: block.color, marginBottom: '12px' }}>
            {block.who}
          </div>
          <ul style={{ margin: 0, paddingLeft: '16px', listStyle: 'none' }}>
            {block.items.map((item, i) => (
              <li key={i} style={{
                fontSize: '12px', color: '#aaa', lineHeight: 1.7, paddingBottom: '6px',
                paddingLeft: '12px', borderLeft: `1px solid ${block.color}33`, marginLeft: '-1px',
              }}>
                {item}
              </li>
            ))}
          </ul>
        </Card>
      ))}
    </div>
  )
}

// ── Key stats strip ───────────────────────────────────────────────────────────

function StatsStrip() {
  const stats = [
    { label: 'TOTAL MISSIONS', value: '80K', note: '8 planets' },
    { label: 'BEST MODEL F1',  value: '0.992', note: 'XGBoost-summary (5-seed CI ±0.001)' },
    { label: 'TRANSFORMER AUC', value: '0.984', note: 'Post-calibration' },
    { label: 'JIT SPEEDUP',    value: '26–37×', note: 'Neptune generation' },
    { label: 'CALIBRATION ECE', value: '0.0045', note: '11.5× better with isotonic' },
  ]
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5,1fr)', gap: '12px', marginBottom: '28px' }}>
      {stats.map(s => (
        <div key={s.label} className="tile" style={{ borderTopColor: 'var(--cyan)' }}>
          <div className="tile-label">{s.label}</div>
          <div className="tile-value" style={{ fontSize: '18px' }}>{s.value}</div>
          <div className="tile-sub" style={{ color: 'var(--text-dim)' }}>{s.note}</div>
        </div>
      ))}
    </div>
  )
}

// ── Root ──────────────────────────────────────────────────────────────────────

export default function Report() {
  return (
    <div className="panel-content">

      {/* Header */}
      <div style={{ marginBottom: '28px' }}>
        <div style={{ fontSize: '18px', color: 'var(--cyan)', letterSpacing: '0.08em', fontWeight: 700, marginBottom: '6px' }}>
          PROGRESS REPORT
        </div>
        <div style={{ fontSize: '12px', color: 'var(--text-dim)', lineHeight: 1.8, maxWidth: '720px' }}>
          OrbitGuard predicts whether a spacecraft trajectory will succeed — using just 40% of the simulated
          flight path. This lets us stop wasted NASA GMAT simulations early, saving up to 80% of compute.
          The model watches real physics telemetry (13 features per timestep) and outputs a Go/No-Go verdict.
        </div>
      </div>

      {/* Key stats */}
      <StatsStrip />

      {/* Two-column layout */}
      <div className="g2" style={{ gap: '24px' }}>

        {/* Left column */}
        <div>
          <Section title="What we built — who did what">
            <Timeline />
          </Section>

          <Section title="Model performance — F1 score by model">
            <ModelChart />
            <div style={{ fontSize: '10px', color: '#555', marginTop: '8px', lineHeight: 1.6 }}>
              XGBoost outperforms Transformer because the task is highly separable from
              initial trajectory statistics. Transformer is still valuable as the primary
              online classifier (it sees one timestep at a time, not the full trajectory).
            </div>
          </Section>

          <Section title="Neptune generation speedup — Numba JIT">
            <JitChart />
          </Section>
        </div>

        {/* Right column */}
        <div>
          <Section title="Production models — per planet (live)">
            <ProductionTable />
          </Section>

          <Section title="Superseded — leave-one-target-out on the single global model">
            <div style={{
              fontSize: '10px', color: '#c08a2e', fontFamily: 'Share Tech Mono,monospace',
              border: '1px solid #3a2a0a', background: '#140f03',
              padding: '7px 10px', marginBottom: '10px', lineHeight: 1.6,
            }}>
              ⚠ HISTORICAL — describes the one-model-for-all-planets architecture that
              production replaced. These numbers are why the split happened; they are not
              current performance. See the table above.
            </div>
            <LotoTable />
            <div style={{ fontSize: '10px', color: '#555', marginTop: '8px', lineHeight: 1.6 }}>
              Held out one planet at a time, trained on the other 7, tested on the held-out one.
              Stable across 5 seeds (std = 0.000) — the result is real, not noise. The failures
              here are regime shift: one shared scaler cannot span all targets, which is exactly
              what the per-planet models fix.
            </div>
          </Section>

          <Section title="Calibration — Transformer predictions">
            <CalibChart />
          </Section>

          <Section title="Domain generalisation — oversampling weak targets">
            <DomainGenTable />
          </Section>
        </div>

      </div>

      {/* Finding summary */}
      <Section title="Key findings summary">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: '12px' }}>
          {[
            {
              title: 'XGBoost nearly perfect',
              body: 'XGBoost-summary hits F1=0.992 and AUC=1.000 on 80K missions across 8 planets. This works because mission outcome is mostly determined by initial conditions — summary statistics are enough.',
              color: 'var(--green)',
            },
            {
              title: 'Generalisation: 2 failure modes',
              body: 'Uranus & Venus fail only because the threshold is wrong — their ranking is fine (AUC > 0.85). Mars, Mercury, Moon fail genuinely — the model has never seen physics at their scale.',
              color: '#ffaa00',
            },
            {
              title: 'Balancing helps some, hurts others',
              body: 'Oversampling Moon/Mars/Mercury/Venus (2×) improved Moon by +0.249 F1 and Mars by +0.024 — but Venus completely collapsed (0.902 → 0.000). No free lunch.',
              color: 'var(--red)',
            },
          ].map(f => (
            <Card key={f.title} style={{ borderTopWidth: '2px', borderTopColor: f.color }}>
              <div style={{ fontSize: '11px', color: f.color, letterSpacing: '0.06em', marginBottom: '8px' }}>
                {f.title.toUpperCase()}
              </div>
              <div style={{ fontSize: '11px', color: '#999', lineHeight: 1.75 }}>{f.body}</div>
            </Card>
          ))}
        </div>
      </Section>

    </div>
  )
}
