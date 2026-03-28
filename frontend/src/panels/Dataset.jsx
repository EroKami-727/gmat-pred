const FEATURES = [
  { name: 'rel_x / rel_y / rel_z', desc: 'Target-centric synodic frame position (km)', ctx: false },
  { name: 'spec_energy',           desc: 'Specific orbital energy v²/2 − μ/r (J/kg)', ctx: false },
  { name: 'fpa_deg',               desc: 'Flight path angle (degrees)', ctx: false },
  { name: 'norm_target_dist',      desc: 'Distance to target / SOI radius', ctx: false },
  { name: 'radial_vel',            desc: 'dr/dt toward target body (km/s)', ctx: false },
  { name: 'vel_mag',               desc: 'Total spacecraft speed (km/s)', ctx: false },
  { name: 'earth_rmag',            desc: 'Distance from source body (km)', ctx: false },
  { name: 'ecc',                   desc: 'Orbital eccentricity', ctx: false },
  { name: 'mu_ratio',              desc: 'μ_target / μ_sun — enables cross-planet transfer', ctx: true },
  { name: 'soi_ratio',             desc: 'SOI_target / dist_transfer — gravity well focus', ctx: true },
  { name: 'dist_ratio',            desc: 'dist_transfer / 1 AU — spatial scale normalizer', ctx: true },
]

const FAILURES = [
  { id: 0, name: 'success',          cls: 'flyby corridor ✓',    color: 'var(--green)' },
  { id: 1, name: 'surface_impact',   cls: 'hit target body',      color: 'var(--red)' },
  { id: 2, name: 'orbit_too_high',   cls: 'periapsis > 500 km',   color: 'var(--orange)' },
  { id: 3, name: 'missed_target',    cls: 'SOI miss',             color: 'var(--orange)' },
  { id: 4, name: 'source_impact',    cls: 'return to source',     color: 'var(--red)' },
  { id: 5, name: 'hyperbolic',       cls: 'escape trajectory',    color: 'var(--cyan)' },
  { id: 6, name: 'degenerate',       cls: 'numerical failure',    color: 'var(--text-dim)' },
  { id: 7, name: 'unknown',          cls: 'unclassified',         color: 'var(--text-dim)' },
]

const PLANETS = [
  {
    name: 'EARTH → MOON',
    stats: [['TOI burn', '3.240 km/s'], ['Propagation', '6 days'], ['Corridor', '100–500 km alt'], ['Status', 'PRIMARY TRAINED']],
    statusColor: 'var(--green)',
  },
  {
    name: 'EARTH → MARS',
    stats: [['Transfer', 'Hohmann approx'], ['μ_ratio', '0.0000320'], ['SOI', '577,000 km'], ['Status', 'ZERO-SHOT TARGET']],
    statusColor: 'var(--orange)',
  },
  {
    name: 'EARTH → JUPITER',
    stats: [['Transfer', 'Hohmann approx'], ['μ_ratio', '0.000955'], ['SOI', '48.2M km'], ['Status', 'FUTURE RESEARCH']],
    statusColor: 'var(--text-dim)',
  },
]

const ARCHS = [
  {
    name: 'TrajectoryLSTM',
    tag: 'PRIMARY',
    tagColor: 'var(--green)',
    specs: [
      ['Embedding', 'Linear → ReLU → Dropout(0.2)'],
      ['Layers', '2 × stacked LSTM'],
      ['Hidden', '128 units'],
      ['Dropout', '0.2'],
      ['Packing', 'pack_padded_sequence ✓'],
      ['Total params', '~250K'],
      ['Loss', 'BCEWithLogitsLoss + pos_weight'],
    ],
  },
  {
    name: 'TrajectoryTransformer',
    tag: 'RESEARCH',
    tagColor: 'var(--orange)',
    specs: [
      ['d_model', '64'],
      ['Heads', '4 × self-attention'],
      ['Layers', '3 encoder blocks'],
      ['FF dim', '128'],
      ['Pos. enc', 'Sinusoidal (max 2000)'],
      ['Pooling', 'Masked mean over valid steps'],
      ['Loss', 'BCEWithLogitsLoss'],
    ],
  },
]

const PIPELINE = [
  ['# 1. Generate 10K missions (2 × 5K runs)',   'dim'],
  ['python3 -m src.data_collection.build_database \\', 'info'],
  ['  --num-missions 5000 --output-dir data/production', 'dim'],
  ['', 'dim'],
  ['# 2. Merge into master (streaming, 13.44 GB)', 'dim'],
  ['python3 -m src.data_collection.merge_datasets \\', 'info'],
  ['  --base data/production --new data/run2 --out data/merged', 'dim'],
  ['', 'dim'],
  ['# 3. Train LSTM (30–50 epochs)', 'dim'],
  ['python3 -m src.ml.train \\', 'info'],
  ['  --data data/merged/missions.parquet --epochs 50 --model lstm', 'dim'],
  ['', 'dim'],
  ['# 4. Ablation sweep', 'dim'],
  ['python3 -m src.ml.ablation \\', 'info'],
  ['  --data data/merged/missions.parquet --epochs 30', 'dim'],
  ['', 'dim'],
  ['# 5. Launch Streamlit dashboard', 'dim'],
  ['streamlit run src/frontend/app.py', 'info'],
]

export default function Dataset() {
  return (
    <div className="panel-content">
      {/* Top stat tiles */}
      <div className="g4" style={{ marginBottom: '20px' }}>
        {[
          { label: 'TOTAL MISSIONS',  value: '10', unit: 'K',      note: 'Monte Carlo dispersion',     bc: 'var(--cyan)',     nc: 'var(--green)' },
          { label: 'TELEMETRY ROWS',  value: '85', unit: '.3M',    note: '↓ 15× downsampled (15-min)', bc: 'var(--cyan-dim)', nc: 'var(--green)' },
          { label: 'SUCCESS RATE',    value: '35', unit: '.3%',    note: 'Stratified class balance',   bc: 'var(--orange)',   nc: 'var(--orange)' },
          { label: 'STORAGE',         value: '13', unit: '.44 GB', note: 'Zstd + Snappy Parquet',      bc: 'var(--green)',    nc: 'var(--green)' },
        ].map(t => (
          <div key={t.label} className="tile" style={{ borderTopColor: t.bc }}>
            <div className="tile-label">{t.label}</div>
            <div className="tile-value">{t.value}<em>{t.unit}</em></div>
            <div className="tile-sub" style={{ color: t.nc }}>{t.note}</div>
          </div>
        ))}
      </div>

      <div className="g2" style={{ gap: '22px' }}>

        {/* Left */}
        <div>
          <div className="sec">PHYSICS-INVARIANT INPUT FEATURES (13-DIM)</div>
          <div className="feat-grid">
            {FEATURES.map(f => (
              <div key={f.name} className="feat-card" style={f.ctx ? { borderLeftColor: 'var(--green)' } : {}}>
                <div className="feat-name" style={f.ctx ? { color: 'var(--green)' } : {}}>{f.name}</div>
                <div className="feat-desc">{f.desc}{f.ctx && <span style={{ color: 'var(--green)', fontSize: '10px' }}>&nbsp;[cross-planet]</span>}</div>
              </div>
            ))}
          </div>

          <div className="sec">MESH TOPOLOGY — PLANETARY TRANSFERS</div>
          <div className="planet-grid">
            {PLANETS.map(p => (
              <div key={p.name} className="planet-card">
                <div className="planet-name">{p.name}</div>
                {p.stats.map(([k, v], i) => (
                  i < 3
                    ? <div key={k} className="planet-stat">{k}: <b>{v}</b></div>
                    : <div key={k} style={{ marginTop: '8px', fontSize: '11px', color: p.statusColor, letterSpacing: '0.04em' }}>○ {v}</div>
                ))}
              </div>
            ))}
          </div>
        </div>

        {/* Right */}
        <div>
          <div className="sec">FAILURE CLASSIFICATION (8 OUTPUT CLASSES)</div>
          <div className="ft-grid">
            {FAILURES.map(f => (
              <div key={f.id} className="ft-card">
                <div className="ft-id" style={{ color: f.color }}>{f.id}</div>
                <div className="ft-name">{f.name}</div>
                <div className="ft-cls">{f.cls}</div>
              </div>
            ))}
          </div>

          <div className="sec">MODEL ARCHITECTURES</div>
          <div className="arch-grid">
            {ARCHS.map(a => (
              <div key={a.name} className="arch-card">
                <div className="arch-name">
                  {a.name}
                  <span style={{ fontSize: '11px', color: a.tagColor, letterSpacing: '0.04em' }}>[{a.tag}]</span>
                </div>
                {a.specs.map(([k, v]) => (
                  <div key={k} className="stat-row">{k} <span>{v}</span></div>
                ))}
              </div>
            ))}
          </div>

          <div className="sec">PIPELINE QUICK START</div>
          <div className="terminal" style={{ height: 'auto' }}>
            {PIPELINE.map(([line, cls], i) => (
              <div key={i} className={cls}>{line}</div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
