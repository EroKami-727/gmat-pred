import { useEffect, useRef, useState } from 'react'
import { API } from '../lib/api.js'

// ─── Shared Hook ────────────────────────────────────────────────────────────

function useExperiment(startUrl, onComplete) {
  const [logs, setLogs]         = useState([['dim', '> READY']])
  const [running, setRunning]   = useState(false)
  const [jobId, setJobId]       = useState(null)
  const [apiOnline, setApiOnline] = useState(null)
  const esRef   = useRef(null)
  const termRef = useRef(null)

  useEffect(() => {
    fetch(`${API}/api/health`, { signal: AbortSignal.timeout(3000) })
      .then(() => setApiOnline(true))
      .catch(() => setApiOnline(false))
  }, [])

  useEffect(() => {
    if (termRef.current) termRef.current.scrollTop = termRef.current.scrollHeight
  }, [logs])

  useEffect(() => () => esRef.current?.close(), [])

  const appendLog = (type, msg) =>
    setLogs(prev => [...prev.slice(-150), [type, msg]])

  const start = async (payload) => {
    try {
      const res = await fetch(`${API}${startUrl}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const { job_id } = await res.json()
      setJobId(job_id)
      setRunning(true)
      setLogs([['info', `> JOB ${job_id} STARTED`]])

      const es = new EventSource(`${API}/api/jobs/stream/${job_id}`)
      esRef.current = es
      es.onmessage = (e) => {
        let event
        try { event = JSON.parse(e.data) } catch { return }
        if (event.type === 'log') {
          const text = event.text
          const cls = text.includes('Traceback') || text.includes('Error') ? 'warn'
                    : text.startsWith('▸') || text.startsWith('═') || text.includes('COMPLETE') ? 'info'
                    : text.includes('WARN') || text.includes('FAILED') ? 'warn'
                    : 'dim'
          appendLog(cls, text)
        }
        if (event.type === 'done') {
          es.close()
          setRunning(false)
          if (event.exit_code === 0) {
            appendLog('info', '▸ EXPERIMENT COMPLETE ✓')
            onComplete?.()
          } else if (event.exit_code === -1) {
            appendLog('warn', '> STOPPED BY USER')
          } else {
            appendLog('warn', `> FAILED (exit ${event.exit_code})`)
          }
        }
      }
      es.onerror = () => { appendLog('warn', '> SSE CONNECTION LOST'); setRunning(false) }
    } catch (err) {
      appendLog('warn', `> FAILED TO START: ${err.message}`)
    }
  }

  const stop = async () => {
    esRef.current?.close()
    if (jobId) await fetch(`${API}/api/jobs/stop/${jobId}`).catch(() => {})
    appendLog('warn', '> STOPPED')
    setRunning(false)
    setJobId(null)
  }

  return { logs, running, apiOnline, start, stop, termRef }
}


// ─── Shared UI Primitives ───────────────────────────────────────────────────

function Terminal({ logs, termRef, height = '220px' }) {
  return (
    <div ref={termRef} className="terminal" style={{ height }}>
      {logs.map(([type, msg], i) => (
        <div key={i} className={type}>{msg || ' '}</div>
      ))}
    </div>
  )
}

function Field({ label, value, onChange, disabled, rows }) {
  const baseStyle = {
    fontFamily: 'var(--mono)',
    fontSize: '11px',
    background: 'var(--bg3)',
    border: '1px solid var(--border)',
    color: 'var(--text)',
    padding: '6px 8px',
    width: '100%',
    boxSizing: 'border-box',
    letterSpacing: '0.04em',
  }
  return (
    <div className="field">
      <div className="field-label">{label}</div>
      {rows
        ? <textarea className="field-input" rows={rows} style={{ resize: 'none', ...baseStyle }} value={value} onChange={onChange} disabled={disabled} />
        : <input    className="field-input"              style={baseStyle}                       value={value} onChange={onChange} disabled={disabled} />
      }
    </div>
  )
}

function ApiStatus({ apiOnline }) {
  const col = apiOnline == null ? '#888888' : apiOnline ? 'var(--green)' : 'var(--red)'
  const label = apiOnline == null ? 'CHECKING...' : apiOnline ? 'ONLINE' : 'OFFLINE'
  return (
    <div style={{ marginBottom: '14px', padding: '7px 10px', background: 'var(--bg3)', border: '1px solid var(--border)', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '8px' }}>
      <span style={{ width: '7px', height: '7px', borderRadius: '50%', flexShrink: 0, background: col, boxShadow: apiOnline ? `0 0 6px ${col}` : 'none' }} />
      <span style={{ color: 'var(--text-dim)', letterSpacing: '0.06em' }}>API: {label}</span>
    </div>
  )
}

function RunButton({ running, apiOnline, onStart, onStop, label = 'RUN EXPERIMENT' }) {
  return (
    <button
      className={`launch-btn${running ? ' running' : ''}`}
      onClick={running ? onStop : onStart}
      disabled={apiOnline === false}
      title={apiOnline === false ? 'Start the FastAPI backend first' : ''}
    >
      {running ? '⏹  STOP' : `▶  ${label}`}
    </button>
  )
}

function ResultsTable({ headers, rows, emptyMsg = 'No results yet.' }) {
  if (!rows?.length) return (
    <div style={{ fontSize: '11px', color: 'var(--text-dim)', padding: '8px 0', fontFamily: 'var(--mono)' }}>{emptyMsg}</div>
  )
  return (
    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '11px', fontFamily: 'var(--mono)' }}>
      <thead>
        <tr>
          {headers.map(h => (
            <th key={h} style={{ textAlign: 'left', padding: '5px 8px', color: 'var(--text-dim)', borderBottom: '1px solid var(--border)', letterSpacing: '0.06em', fontWeight: 'normal' }}>{h}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row, i) => (
          <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
            {row.map((cell, j) => (
              <td key={j} style={{ padding: '5px 8px', color: cell?.color ?? 'var(--text)' }}>
                {cell?.text ?? cell}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  )
}


// ─── Baselines Tab ──────────────────────────────────────────────────────────

function BaselineResults({ data }) {
  const headers = ['Exit %', 'Majority AUC', 'Energy AUC / F1', 'XGBoost AUC / F1']
  const rows = data.map(row => [
    { text: `${row.early_exit_pct}%`, color: 'var(--cyan)' },
    row.majority_class?.error ? { text: 'ERR', color: 'var(--red)' }
      : `${row.majority_class?.auc?.toFixed(3) ?? '—'}`,
    row.energy_threshold?.error ? { text: 'ERR', color: 'var(--red)' }
      : `${row.energy_threshold?.auc?.toFixed(3) ?? '—'} / ${row.energy_threshold?.f1?.toFixed(3) ?? '—'}`,
    row.xgboost?.error ? { text: 'ERR', color: 'var(--red)' }
      : `${row.xgboost?.auc?.toFixed(3) ?? '—'} / ${row.xgboost?.f1?.toFixed(3) ?? '—'}`,
  ])
  return (
    <div className="card">
      <div className="card-hdr">RESULTS — BASELINE COMPARISON</div>
      <ResultsTable headers={headers} rows={rows} />
      <div style={{ marginTop: '8px', fontSize: '10px', color: 'var(--text-dim)', fontFamily: 'var(--mono)' }}>
        Full results → reports/baselines/baseline_results.json
      </div>
    </div>
  )
}

function BaselinesTab() {
  const [params, setParams] = useState({ exitFracs: '0.1 0.2 0.3 0.4 0.6 1.0', seed: '42', downsampleFactor: '15' })
  const [results, setResults] = useState(null)
  const set = k => e => setParams(p => ({ ...p, [k]: e.target.value }))

  const loadResults = () =>
    fetch(`${API}/api/experiments/baselines/results`)
      .then(r => r.ok ? r.json() : null)
      .then(d => d && setResults(d))
      .catch(() => {})

  useEffect(() => { loadResults() }, [])
  const exp = useExperiment('/api/experiments/baselines/start', loadResults)

  return (
    <div className="g-left-right">
      <div className="card">
        <div className="card-hdr">BASELINES <span className="badge">EXP.01</span></div>
        <ApiStatus apiOnline={exp.apiOnline} />
        <Field label="EXIT FRACTIONS"    value={params.exitFracs}        onChange={set('exitFracs')}        disabled={exp.running} />
        <Field label="RANDOM SEED"       value={params.seed}             onChange={set('seed')}             disabled={exp.running} />
        <Field label="DOWNSAMPLE FACTOR" value={params.downsampleFactor} onChange={set('downsampleFactor')} disabled={exp.running} />
        <RunButton running={exp.running} apiOnline={exp.apiOnline}
          onStart={() => exp.start({ exitFracs: params.exitFracs, seed: params.seed, downsampleFactor: params.downsampleFactor })}
          onStop={exp.stop} label="RUN BASELINES"
        />
        <div style={{ marginTop: '14px', borderTop: '1px solid var(--border)', paddingTop: '12px' }}>
          {[['MODELS', 'MajorityClass · EnergyThreshold · XGBoost'], ['FEATURES', '13×6 summary stats + length = 79'], ['OUTPUT', 'reports/baselines/baseline_results.json']].map(([k, v]) => (
            <div key={k} className="stat-row">{k} <span style={{ color: 'var(--text-dim)', fontSize: '10px' }}>{v}</span></div>
          ))}
        </div>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
        <div className="card" style={{ padding: '14px 16px' }}>
          <div className="card-hdr">LIVE OUTPUT</div>
          <Terminal logs={exp.logs} termRef={exp.termRef} />
        </div>
        {results && <BaselineResults data={results} />}
      </div>
    </div>
  )
}


// ─── Multi-Seed Tab ─────────────────────────────────────────────────────────

function MultiSeedResults({ data }) {
  return (
    <div className="card">
      <div className="card-hdr">RESULTS — MEAN ± STD  ({data.n_seeds} seeds)</div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px', marginBottom: '12px' }}>
        {['auc', 'f1', 'acc'].map(metric => (
          <div key={metric} style={{ background: 'var(--bg3)', border: '1px solid var(--border)', padding: '12px 10px' }}>
            <div style={{ fontSize: '10px', color: 'var(--text-dim)', letterSpacing: '0.08em', fontFamily: 'var(--mono)' }}>{metric.toUpperCase()}</div>
            <div style={{ fontSize: '18px', color: 'var(--cyan)', margin: '6px 0 2px', fontFamily: 'var(--mono)', letterSpacing: '0.04em' }}>
              {data[metric].mean.toFixed(4)}
            </div>
            <div style={{ fontSize: '10px', color: 'var(--text-dim)', fontFamily: 'var(--mono)' }}>
              ± {data[metric].std.toFixed(4)}
            </div>
          </div>
        ))}
      </div>
      <div style={{ fontSize: '10px', color: 'var(--text-dim)', fontFamily: 'var(--mono)', lineHeight: 1.8 }}>
        Seeds: {data.seeds?.join(', ')} · exit={Math.round((data.early_exit_frac ?? 0.4) * 100)}%<br />
        AUC values: {data.auc.values.map(v => v.toFixed(4)).join(', ')}
      </div>
    </div>
  )
}

function MultiSeedTab() {
  const [params, setParams] = useState({ seeds: '42 123 456 789 1024', earlyExit: '0.4', epochs: '30', batchSize: '32', lr: '1e-3' })
  const [results, setResults] = useState(null)
  const set = k => e => setParams(p => ({ ...p, [k]: e.target.value }))

  const loadResults = () =>
    fetch(`${API}/api/experiments/multi_seed/results`)
      .then(r => r.ok ? r.json() : null)
      .then(d => d && setResults(d))
      .catch(() => {})

  useEffect(() => { loadResults() }, [])
  const exp = useExperiment('/api/experiments/multi_seed/start', loadResults)

  return (
    <div className="g-left-right">
      <div className="card">
        <div className="card-hdr">MULTI-SEED <span className="badge">EXP.02</span></div>
        <ApiStatus apiOnline={exp.apiOnline} />
        <Field label="SEEDS (space-separated)" value={params.seeds}     onChange={set('seeds')}     disabled={exp.running} />
        <Field label="EARLY EXIT FRACTION"     value={params.earlyExit} onChange={set('earlyExit')} disabled={exp.running} />
        <Field label="EPOCHS"                  value={params.epochs}    onChange={set('epochs')}    disabled={exp.running} />
        <Field label="BATCH SIZE"              value={params.batchSize} onChange={set('batchSize')} disabled={exp.running} />
        <Field label="LEARNING RATE"           value={params.lr}        onChange={set('lr')}        disabled={exp.running} />
        <RunButton running={exp.running} apiOnline={exp.apiOnline}
          onStart={() => exp.start(params)}
          onStop={exp.stop} label="RUN MULTI-SEED"
        />
        <div style={{ marginTop: '14px', borderTop: '1px solid var(--border)', paddingTop: '12px' }}>
          {[['MODEL', 'TrajectoryTransformer (production)'], ['PURPOSE', 'Mean ± std → defensible claims'], ['OUTPUT', 'reports/multi_seed/summary.json']].map(([k, v]) => (
            <div key={k} className="stat-row">{k} <span style={{ color: 'var(--text-dim)', fontSize: '10px' }}>{v}</span></div>
          ))}
        </div>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
        <div className="card" style={{ padding: '14px 16px' }}>
          <div className="card-hdr">LIVE OUTPUT</div>
          <Terminal logs={exp.logs} termRef={exp.termRef} />
        </div>
        {results && <MultiSeedResults data={results} />}
      </div>
    </div>
  )
}


// ─── Arch Ablation Tab ──────────────────────────────────────────────────────

const VARIANT_OPTIONS = [
  { name: 'full_transformer',     label: 'Full Transformer (reference)' },
  { name: 'no_cls_token',         label: 'No CLS Token (mean pool)' },
  { name: 'no_pos_encoding',      label: 'No Positional Encoding' },
  { name: 'no_context_features',  label: 'No Context Features' },
  { name: 'lstm',                 label: 'LSTM' },
]

function ArchAblationResults({ data }) {
  const ref = data.find(r => r.variant === 'full_transformer')
  const headers = ['Variant', 'AUC', 'F1', 'Acc', 'ΔAUC', 'Params']
  const rows = data.map(r => {
    const isRef = r.variant === 'full_transformer'
    const delta = ref && !isRef ? r.test_auc - ref.test_auc : null
    const deltaStr = delta != null ? (delta >= 0 ? `+${delta.toFixed(3)}` : delta.toFixed(3)) : 'ref'
    const deltaColor = delta == null ? 'var(--text-dim)' : delta < 0 ? 'var(--red)' : 'var(--green)'
    return [
      { text: r.label, color: isRef ? 'var(--cyan)' : 'var(--text)' },
      r.test_auc.toFixed(3),
      r.test_f1.toFixed(3),
      `${(r.test_acc * 100).toFixed(1)}%`,
      { text: deltaStr, color: deltaColor },
      r.total_params?.toLocaleString() ?? '—',
    ]
  })
  return (
    <div className="card">
      <div className="card-hdr">RESULTS — COMPONENT ABLATION</div>
      <ResultsTable headers={headers} rows={rows} />
      <div style={{ marginTop: '8px', fontSize: '10px', color: 'var(--text-dim)', fontFamily: 'var(--mono)' }}>
        Full results → reports/arch_ablation/arch_ablation_results.json
      </div>
    </div>
  )
}

function ArchAblationTab() {
  const [selected, setSelected] = useState(VARIANT_OPTIONS.map(v => v.name))
  const [earlyExit, setEarlyExit] = useState('0.4')
  const [epochs, setEpochs]     = useState('30')
  const [seed, setSeed]         = useState('42')
  const [results, setResults]   = useState(null)

  const toggleVariant = name =>
    setSelected(prev => prev.includes(name) ? prev.filter(n => n !== name) : [...prev, name])

  const loadResults = () =>
    fetch(`${API}/api/experiments/arch_ablation/results`)
      .then(r => r.ok ? r.json() : null)
      .then(d => d && setResults(d))
      .catch(() => {})

  useEffect(() => { loadResults() }, [])
  const exp = useExperiment('/api/experiments/arch_ablation/start', loadResults)

  return (
    <div className="g-left-right">
      <div className="card">
        <div className="card-hdr">ARCH ABLATION <span className="badge">EXP.03</span></div>
        <ApiStatus apiOnline={exp.apiOnline} />

        <div style={{ marginBottom: '14px' }}>
          <div className="field-label" style={{ marginBottom: '8px' }}>VARIANTS TO RUN</div>
          {VARIANT_OPTIONS.map(v => (
            <label key={v.name} style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px', cursor: exp.running ? 'not-allowed' : 'pointer', opacity: exp.running ? 0.5 : 1 }}>
              <input
                type="checkbox"
                checked={selected.includes(v.name)}
                onChange={() => !exp.running && toggleVariant(v.name)}
                style={{ accentColor: 'var(--cyan)' }}
              />
              <span style={{ fontSize: '11px', color: 'var(--text-dim)', fontFamily: 'var(--mono)', letterSpacing: '0.04em' }}>{v.label}</span>
            </label>
          ))}
        </div>

        <Field label="EARLY EXIT FRACTION" value={earlyExit} onChange={e => setEarlyExit(e.target.value)} disabled={exp.running} />
        <Field label="EPOCHS"              value={epochs}    onChange={e => setEpochs(e.target.value)}    disabled={exp.running} />
        <Field label="SEED"                value={seed}      onChange={e => setSeed(e.target.value)}      disabled={exp.running} />

        <RunButton running={exp.running} apiOnline={exp.apiOnline}
          onStart={() => exp.start({ variants: selected, earlyExit, epochs, seed })}
          onStop={exp.stop} label="RUN ABLATION"
        />
        <div style={{ marginTop: '14px', borderTop: '1px solid var(--border)', paddingTop: '12px' }}>
          {[['PURPOSE', 'Measure each component\'s contribution'], ['OUTPUT', 'reports/arch_ablation/']].map(([k, v]) => (
            <div key={k} className="stat-row">{k} <span style={{ color: 'var(--text-dim)', fontSize: '10px' }}>{v}</span></div>
          ))}
        </div>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
        <div className="card" style={{ padding: '14px 16px' }}>
          <div className="card-hdr">LIVE OUTPUT</div>
          <Terminal logs={exp.logs} termRef={exp.termRef} />
        </div>
        {results && <ArchAblationResults data={results} />}
      </div>
    </div>
  )
}


// ─── Tables Tab ─────────────────────────────────────────────────────────────

function CopyButton({ text, label }) {
  const [copied, setCopied] = useState(false)
  const copy = () => {
    navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }
  return (
    <button onClick={copy} style={{
      background: 'transparent', border: '1px solid var(--border2)',
      color: copied ? 'var(--green)' : 'var(--cyan)',
      fontFamily: 'var(--mono)', fontSize: '10px', letterSpacing: '0.08em',
      padding: '3px 10px', cursor: 'pointer', transition: 'color 0.2s',
    }}>
      {copied ? 'COPIED ✓' : label}
    </button>
  )
}

function TablesResults({ data }) {
  const TABLE_LABELS = {
    main_comparison: 'Table 1 — Main Comparison (Transformer vs Baselines)',
    arch_ablation:   'Table 2 — Architecture Ablation',
    multi_seed:      'Table 3 — Multi-Seed Robustness',
  }
  const available = Object.entries(data).filter(([, f]) => f.md || f.tex)
  if (!available.length) return (
    <div style={{ fontSize: '11px', color: 'var(--text-dim)', fontFamily: 'var(--mono)' }}>
      No tables generated yet. Run experiments first, then generate tables.
    </div>
  )
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
      {available.map(([name, files]) => (
        <div key={name} className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
            <div style={{ fontSize: '11px', color: 'var(--cyan)', letterSpacing: '0.08em', fontFamily: 'var(--mono)' }}>
              {TABLE_LABELS[name] ?? name.toUpperCase()}
            </div>
            <div style={{ display: 'flex', gap: '6px' }}>
              {files.md  && <CopyButton text={files.md}  label="COPY .MD" />}
              {files.tex && <CopyButton text={files.tex} label="COPY .TEX" />}
            </div>
          </div>
          {files.md && (
            <pre style={{
              background: 'var(--bg3)', border: '1px solid var(--border)',
              padding: '10px 12px', fontSize: '10px', overflowX: 'auto',
              maxHeight: '180px', overflowY: 'auto',
              color: 'var(--text)', fontFamily: 'var(--mono)',
              margin: 0, lineHeight: 1.6,
            }}>
              {files.md}
            </pre>
          )}
        </div>
      ))}
    </div>
  )
}

function TablesTab() {
  const [tables, setTables] = useState(null)
  const exp = useExperiment('/api/experiments/tables/start', loadTables)

  function loadTables() {
    fetch(`${API}/api/experiments/tables`)
      .then(r => r.ok ? r.json() : null)
      .then(d => d && setTables(d))
      .catch(() => {})
  }

  useEffect(() => { loadTables() }, [])

  const prereqs = [
    'reports/ablation/ablation_results.json',
    'reports/baselines/baseline_results.json',
    'reports/arch_ablation/arch_ablation_results.json',
    'reports/multi_seed/summary.json',
  ]

  return (
    <div className="g-left-right">
      <div className="card">
        <div className="card-hdr">TABLES <span className="badge">EXP.04</span></div>
        <ApiStatus apiOnline={exp.apiOnline} />

        <div style={{ marginBottom: '14px', fontSize: '11px', color: 'var(--text-dim)', fontFamily: 'var(--mono)', lineHeight: 1.8 }}>
          Reads all experiment results and generates publication-ready Markdown and LaTeX tables.
        </div>

        <div style={{ marginBottom: '14px' }}>
          <div className="field-label" style={{ marginBottom: '6px' }}>REQUIRED INPUTS</div>
          {prereqs.map(p => (
            <div key={p} style={{ fontSize: '10px', color: 'var(--text-dim)', fontFamily: 'var(--mono)', padding: '2px 0' }}>
              → {p}
            </div>
          ))}
        </div>

        <RunButton running={exp.running} apiOnline={exp.apiOnline}
          onStart={() => exp.start({})}
          onStop={exp.stop} label="GENERATE TABLES"
        />

        <div style={{ marginTop: '14px', borderTop: '1px solid var(--border)', paddingTop: '12px' }}>
          {[['OUTPUT', 'reports/tables/*.{md,tex}'], ['TABLES', 'Main comparison · Arch ablation · Multi-seed']].map(([k, v]) => (
            <div key={k} className="stat-row">{k} <span style={{ color: 'var(--text-dim)', fontSize: '10px' }}>{v}</span></div>
          ))}
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
        <div className="card" style={{ padding: '14px 16px' }}>
          <div className="card-hdr">LIVE OUTPUT</div>
          <Terminal logs={exp.logs} termRef={exp.termRef} />
        </div>
        {tables && <TablesResults data={tables} />}
      </div>
    </div>
  )
}


// ─── Main Panel ─────────────────────────────────────────────────────────────

const EXP_TABS = ['BASELINES', 'MULTI-SEED', 'ARCH ABL', 'TABLES']

export default function Experiments() {
  const [active, setActive] = useState('BASELINES')

  return (
    <div className="panel-content">
      {/* Sub-tab bar */}
      <div style={{
        display: 'flex', borderBottom: '1px solid var(--border)',
        marginBottom: '14px', paddingBottom: '0',
      }}>
        {EXP_TABS.map(tab => (
          <button
            key={tab}
            onClick={() => setActive(tab)}
            style={{
              padding: '10px 20px',
              fontSize: '11px',
              letterSpacing: '0.1em',
              color: active === tab ? 'var(--cyan)' : 'var(--text-dim)',
              background: 'none', border: 'none',
              borderBottom: active === tab ? '2px solid var(--cyan)' : '2px solid transparent',
              cursor: 'pointer',
              fontFamily: 'var(--mono)',
              transition: 'color 0.15s',
            }}
          >
            {tab}
          </button>
        ))}
      </div>

      {active === 'BASELINES'  && <BaselinesTab />}
      {active === 'MULTI-SEED' && <MultiSeedTab />}
      {active === 'ARCH ABL'   && <ArchAblationTab />}
      {active === 'TABLES'     && <TablesTab />}
    </div>
  )
}
