import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis, YAxis,
} from 'recharts'
import { API } from '../lib/api.js'

// ── Constants ─────────────────────────────────────────────────────────────────

// Empty means "use whatever dataset the server is configured for"
// ($ORBITGUARD_DATA, resolved by src/paths.py). This used to hold an absolute
// path from one developer's machine, which got compiled into the bundle and
// sent to the API on every request — so the deployed dashboard asked the
// backend to read a filesystem path that only existed on someone else's box.
// The field below is still editable, as a deliberate override.
const DEFAULT_DATA = ''
const PLANET_LIST  = ['ALL', 'Jupiter', 'Saturn', 'Neptune', 'Uranus', 'Mars', 'Mercury', 'Venus', 'Moon']
const PLANET_COLORS = {
  Jupiter: '#ffaa44', Saturn: '#ddcc88', Neptune: '#5588ff',
  Uranus:  '#55ddcc', Mars:   '#ff6644', Mercury: '#aaaaaa',
  Venus:   '#ffdd66', Moon:   '#cccccc',
}

// Plain-language meaning of each failure mode, for the prune explanation.
const MODE_DESC = {
  surface_impact:  'trajectory intersects the target body — the spacecraft hits the surface',
  orbit_too_high:  'arrival energy too high — the spacecraft is not captured into the intended orbit',
  missed_target:   'closest approach falls outside the sphere of influence — the target is missed entirely',
  source_impact:   'trajectory falls back and re-impacts the departure body',
  hyperbolic_flyby:'excess hyperbolic energy — the spacecraft escapes past the target',
  degenerate_orbit:'resulting orbit is degenerate / unphysical',
  success:         'nominal transfer',
  unknown:         'mode not classified',
}

const GEN_PLANETS = ['mars', 'venus', 'mercury', 'jupiter', 'saturn', 'uranus', 'neptune']

const INIT_STREAM = {
  status:            'idle',  // idle | loading | streaming | paused | canceled | completed | error
  history:           [],      // displayed history (controlled by playback)
  buffer:            [],      // all received steps from SSE
  currentProb:       null,
  currentIdx:        0,
  playbackPos:       0,       // index into buffer for manual playback
  abortIdx:          null,
  abortPct:          null,
  abortProb:         null,
  abortThreshold:    null,
  wasCorrect:        null,
  trueLabel:         null,
  finalProb:         null,
  calThreshold:      null,    // calibrated threshold from backend info header
  regime:            null,    // target body name (per-planet model in use)
  modelAvail:        true,    // false when no trained model exists for the target
  predMode:          null,    // predicted failure mode (how it would fail)
  modeConf:          null,
  actualFailureType: null,
  modeCorrect:       null,
  ood:               false,   // input far outside the training distribution
  oodFraction:       0,
  canceledInBuffer:  false,   // whether the buffer contains a cancel event
  cancelBufIdx:      null,    // buffer index of the cancel event
}

const SPEEDS = [
  { label: '4×',  delay: 12  },
  { label: '2×',  delay: 25  },
  { label: '1×',  delay: 50  },
  { label: '½×',  delay: 100 },
  { label: '¼×',  delay: 200 },
  { label: 'STEP', delay: null },  // frame-by-frame
]

// ── Helpers ───────────────────────────────────────────────────────────────────

function probColor(p) {
  if (p == null) return '#333333'
  if (p < 0.30)  return 'var(--green)'
  if (p < 0.50)  return '#aacc00'
  if (p < 0.65)  return '#ffaa00'
  if (p < 0.80)  return '#ff7700'
  return 'var(--red)'
}

function statusIcon(s) {
  return { idle: '○', selected: '◈', streaming: '●', canceled: '✗', completed: '✓', error: '△' }[s] ?? '○'
}

function statusColor(s) {
  return {
    idle: '#444444', selected: 'var(--cyan)', streaming: 'var(--cyan)',
    canceled: 'var(--red)', completed: 'var(--green)', error: 'var(--orange)',
  }[s] ?? '#444444'
}

function fmtVal(v, dec = 3) {
  if (v == null || isNaN(v)) return '—'
  const n = Number(v)
  if (Math.abs(n) >= 1e9)  return (n / 1e9).toFixed(2) + 'G'
  if (Math.abs(n) >= 1e6)  return (n / 1e6).toFixed(2) + 'M'
  if (Math.abs(n) >= 1e4)  return (n / 1e3).toFixed(1) + 'k'
  return n.toFixed(dec)
}

// ── OrbitalMap ────────────────────────────────────────────────────────────────

function OrbitalMap({ positions, currentIdx, abortIdx, isLoading, targetName, onScrub, operatingFrac = 0.4 }) {
  const W = 300, H = 300
  const svgRef = useRef(null)

  // pts must be computed BEFORE handleMouseMove closes over it
  const { pts, toSVG } = useMemo(() => {
    if (!positions?.length) return { pts: null, toSVG: null }

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity
    for (const p of positions) {
      if (p.rel_x < minX) minX = p.rel_x
      if (p.rel_x > maxX) maxX = p.rel_x
      if (p.rel_y < minY) minY = p.rel_y
      if (p.rel_y > maxY) maxY = p.rel_y
    }

    // Ensure target (0,0) is within the view
    minX = Math.min(minX, 0); maxX = Math.max(maxX, 0)
    minY = Math.min(minY, 0); maxY = Math.max(maxY, 0)

    const spanX  = maxX - minX || 1
    const spanY  = maxY - minY || 1
    const span   = Math.max(spanX, spanY) * 1.28
    const cx0    = (minX + maxX) / 2
    const cy0    = (minY + maxY) / 2

    const toSVG = (rx, ry) => [
      W / 2 + ((rx - cx0) / span) * W,
      H / 2 - ((ry - cy0) / span) * H,  // flip Y for screen coords
    ]

    const pts = positions.map(p => toSVG(p.rel_x, p.rel_y))
    return { pts, toSVG }
  }, [positions])

  // Scrub: find nearest trajectory point when mouse moves over SVG
  const handleMouseMove = useCallback((e) => {
    if (!pts || !onScrub || !svgRef.current) return
    const rect = svgRef.current.getBoundingClientRect()
    const scaleX = W / rect.width
    const scaleY = H / rect.height
    const mx = (e.clientX - rect.left) * scaleX
    const my = (e.clientY - rect.top)  * scaleY
    let best = 0, bestDist = Infinity
    for (let i = 0; i < pts.length; i++) {
      const dx = pts[i][0] - mx, dy = pts[i][1] - my
      const d = dx*dx + dy*dy
      if (d < bestDist) { bestDist = d; best = i }
    }
    onScrub(best)
  }, [pts, onScrub])

  const mkPath = (arr) =>
    arr.map((p, i) => `${i === 0 ? 'M' : 'L'}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ')

  const { fullD, liveD, observedD, futureD, decisionIdx, closestIdx, arrows, scale } = useMemo(() => {
    if (!pts?.length) return { fullD: '', liveD: '', observedD: '', futureD: '', decisionIdx: 0, closestIdx: 0, arrows: [], scale: null }
    const end   = Math.max(1, Math.min(currentIdx + 1, pts.length))
    const dIdx  = Math.max(1, Math.min(Math.round(operatingFrac * (pts.length - 1)), pts.length - 1))

    // Closest approach to the target — the moment that decides the outcome.
    let closest = 0, best = Infinity
    positions.forEach((p, i) => {
      const d = p.rel_x * p.rel_x + p.rel_y * p.rel_y
      if (d < best) { best = d; closest = i }
    })

    // Direction chevrons spaced along the path so travel direction is readable.
    const arr = []
    const stepN = Math.max(6, Math.floor(pts.length / 7))
    for (let i = stepN; i < pts.length - 1; i += stepN) {
      const [x1, y1] = pts[i - 1], [x2, y2] = pts[i]
      const ang = Math.atan2(y2 - y1, x2 - x1) * 180 / Math.PI
      arr.push({ x: x2, y: y2, ang, past: i <= dIdx })
    }

    // Scale bar: a round number of km mapped to screen units.
    let sc = null
    if (positions.length > 1) {
      const [ax, ay] = pts[0]
      const kmPerPx = Math.hypot(positions[0].rel_x - positions[1].rel_x,
                                 positions[0].rel_y - positions[1].rel_y) /
                      (Math.hypot(ax - pts[1][0], ay - pts[1][1]) || 1)
      if (isFinite(kmPerPx) && kmPerPx > 0) {
        const targetPx = W * 0.22
        const raw = kmPerPx * targetPx
        const pow = Math.pow(10, Math.floor(Math.log10(raw)))
        const nice = [1, 2, 5, 10].map(m => m * pow).find(v => v >= raw * 0.6) ?? pow
        sc = { px: nice / kmPerPx, km: nice }
      }
    }

    return {
      fullD: mkPath(pts),
      liveD: mkPath(pts.slice(0, end)),
      observedD: mkPath(pts.slice(0, dIdx + 1)),
      futureD: mkPath(pts.slice(dIdx)),
      decisionIdx: dIdx, closestIdx: closest, arrows: arr, scale: sc,
    }
  }, [pts, currentIdx, positions, operatingFrac])

  if (isLoading) {
    return (
      <div style={mapShell}>
        <span style={mapLabel}>LOADING TRAJECTORY DATA...</span>
      </div>
    )
  }

  if (!pts) {
    return (
      <div style={mapShell}>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '10px' }}>
          <div style={{ fontSize: '9px', color: '#2a2a2a', fontFamily: 'var(--mono)', letterSpacing: '0.12em' }}>[ ORBITAL MAP ]</div>
          <div style={mapLabel}>SELECT A MISSION TO INITIALIZE</div>
        </div>
      </div>
    )
  }

  const [tx, ty]   = toSVG(0, 0)
  const [sx, sy]   = pts[0]
  const safeIdx    = Math.max(0, Math.min(currentIdx, pts.length - 1))
  const [dx, dy]   = pts[safeIdx]
  const abortPt    = abortIdx != null ? pts[Math.min(abortIdx, pts.length - 1)] : null
  const corridorR  = W * 0.052
  const pColor     = PLANET_COLORS[targetName] || 'var(--cyan)'

  return (
    <svg ref={svgRef} width="100%" height="100%" viewBox={`0 0 ${W} ${H}`}
      style={{ display: 'block', background: 'var(--bg2)', border: '1px solid var(--border)', cursor: pts ? 'crosshair' : 'default' }}
      onMouseMove={handleMouseMove}
      onMouseLeave={() => onScrub && onScrub(null)}>

      {/* Background grid lines */}
      <line x1={W / 2} y1={0} x2={W / 2} y2={H} stroke="#161616" strokeWidth={0.5} />
      <line x1={0} y1={H / 2} x2={W} y2={H / 2} stroke="#161616" strokeWidth={0.5} />

      {/* Approach corridor rings */}
      <circle cx={tx} cy={ty} r={corridorR}
        fill={`${pColor}08`} stroke={pColor} strokeWidth={0.5} strokeDasharray="4 5" opacity={0.7} />
      <circle cx={tx} cy={ty} r={corridorR * 2.2}
        fill="none" stroke="#1a1a1a" strokeWidth={0.5} strokeDasharray="2 8" opacity={0.5} />

      {/* Trajectory beyond the decision point — the model never sees this.
          Dashed so it reads as "what would have happened", not as evidence. */}
      <path d={futureD} fill="none" stroke="#2b2b2b" strokeWidth={1.4}
        strokeDasharray="3 4" strokeLinecap="round" />

      {/* Observed window (0 → decision point) */}
      <path d={observedD} fill="none" stroke="#3d3d3d" strokeWidth={2} strokeLinecap="round" />

      {/* Live traversed path */}
      {currentIdx > 0 && (
        <path d={liveD} fill="none" stroke="var(--cyan)" strokeWidth={2.2}
          opacity={0.9} strokeLinecap="round" />
      )}

      {/* Travel-direction chevrons */}
      {arrows.map((a, i) => (
        <path key={i} d="M-3.5,-2.6 L0,0 L-3.5,2.6"
          transform={`translate(${a.x.toFixed(1)},${a.y.toFixed(1)}) rotate(${a.ang.toFixed(1)})`}
          fill="none" stroke={a.past ? '#5a5a5a' : '#333'} strokeWidth={1.1}
          strokeLinecap="round" strokeLinejoin="round" />
      ))}

      {/* Decision point — where the abort call is made */}
      {pts[decisionIdx] && (
        <>
          <line x1={pts[decisionIdx][0]} y1={pts[decisionIdx][1] - 9}
                x2={pts[decisionIdx][0]} y2={pts[decisionIdx][1] + 9}
                stroke="#ffaa00" strokeWidth={1.2} opacity={0.85} />
          <text x={pts[decisionIdx][0] + 5} y={pts[decisionIdx][1] - 11}
            fill="#ffaa00" fontSize="7" fontFamily="Share Tech Mono, monospace"
            letterSpacing="0.06em" opacity={0.9}>
            DECIDE {Math.round(operatingFrac * 100)}%
          </text>
        </>
      )}

      {/* Closest approach — what actually determines the outcome */}
      {pts[closestIdx] && closestIdx !== 0 && (
        <>
          <circle cx={pts[closestIdx][0]} cy={pts[closestIdx][1]} r={3.2}
            fill="none" stroke={pColor} strokeWidth={1} opacity={0.8} />
          <text x={pts[closestIdx][0] + 6} y={pts[closestIdx][1] + 10}
            fill={`${pColor}bb`} fontSize="7" fontFamily="Share Tech Mono, monospace"
            letterSpacing="0.06em">
            CLOSEST
          </text>
        </>
      )}

      {/* Target body at (0,0) — filled disc so it reads as a body, not a ring */}
      <circle cx={tx} cy={ty} r={corridorR} fill="none" stroke={pColor}
        strokeWidth={0.8} strokeDasharray="3 4" opacity={0.55} />
      <circle cx={tx} cy={ty} r={7} fill={pColor} opacity={0.28} />
      <circle cx={tx} cy={ty} r={4.5} fill={pColor} />
      <text x={tx + 12} y={ty - 8} fill={pColor} fontSize="9"
        fontFamily="Share Tech Mono, monospace" letterSpacing="0.1em">
        {(targetName || 'TARGET').toUpperCase()}
      </text>
      <text x={tx + 12} y={ty + 1} fill={`${pColor}77`} fontSize="6.5"
        fontFamily="Share Tech Mono, monospace" letterSpacing="0.06em">
        SOI
      </text>

      {/* Departure body — distinct hue so it is never confused with the s/c */}
      <circle cx={sx} cy={sy} r={5.5} fill="#3d6fd9" opacity={0.30} />
      <circle cx={sx} cy={sy} r={3} fill="#5b87e8" />
      <text x={sx + 9} y={sy - 6} fill="#5b87e8" fontSize="8"
        fontFamily="Share Tech Mono, monospace" letterSpacing="0.08em">
        EARTH
      </text>
      <text x={sx + 9} y={sy + 3} fill="#5b87e877" fontSize="6.5"
        fontFamily="Share Tech Mono, monospace" letterSpacing="0.06em">
        DEPART
      </text>

      {/* Abort marker */}
      {abortPt && (
        <>
          <circle cx={abortPt[0]} cy={abortPt[1]} r={11}
            fill="none" stroke="var(--red)" strokeWidth={2} />
          <line x1={abortPt[0] - 7} y1={abortPt[1] - 7}
                x2={abortPt[0] + 7} y2={abortPt[1] + 7}
                stroke="var(--red)" strokeWidth={1.5} />
          <line x1={abortPt[0] + 7} y1={abortPt[1] - 7}
                x2={abortPt[0] - 7} y2={abortPt[1] + 7}
                stroke="var(--red)" strokeWidth={1.5} />
          <text x={abortPt[0] + 15} y={abortPt[1] + 4}
            fill="var(--red)" fontSize="8" fontFamily="Share Tech Mono, monospace"
            letterSpacing="0.08em">
            ML ABORT
          </text>
        </>
      )}

      {/* Spacecraft — a chevron on the path, not another circle */}
      <circle cx={dx} cy={dy} r={13} fill="none" stroke="#fff" strokeWidth={0.5} opacity={0.18} />
      <circle cx={dx} cy={dy} r={7}  fill="#0a0a0a" stroke="#fff" strokeWidth={1.4} />
      <circle cx={dx} cy={dy} r={2.4} fill="#fff" />
      <text x={dx + 14} y={dy + 3} fill="#fff" fontSize="8"
        fontFamily="Share Tech Mono, monospace" letterSpacing="0.1em" opacity={0.95}>
        S/C
      </text>

      {/* Scale bar */}
      {scale && (
        <g transform={`translate(${W - scale.px - 14}, ${H - 20})`}>
          <line x1={0} y1={0} x2={scale.px} y2={0} stroke="#5a5a5a" strokeWidth={1} />
          <line x1={0} y1={-3} x2={0} y2={3} stroke="#5a5a5a" strokeWidth={1} />
          <line x1={scale.px} y1={-3} x2={scale.px} y2={3} stroke="#5a5a5a" strokeWidth={1} />
          <text x={scale.px / 2} y={-5} fill="#6a6a6a" fontSize="7" textAnchor="middle"
            fontFamily="Share Tech Mono, monospace" letterSpacing="0.06em">
            {scale.km >= 1e6 ? `${(scale.km / 1e6).toFixed(scale.km >= 1e7 ? 0 : 1)}M km`
              : scale.km >= 1e3 ? `${(scale.km / 1e3).toFixed(0)}k km`
              : `${scale.km.toFixed(0)} km`}
          </text>
        </g>
      )}

      {/* Header + legend */}
      <text x={8} y={14} fill="#3a3a3a" fontSize="8"
        fontFamily="Share Tech Mono, monospace" letterSpacing="0.08em">
        [ SYNODIC FRAME — TARGET AT ORIGIN ]
      </text>
      <g transform={`translate(8, ${H - 8})`} fontFamily="Share Tech Mono, monospace" fontSize="7">
        <circle cx={3} cy={-3} r={2.6} fill="#0a0a0a" stroke="#fff" strokeWidth={1} />
        <text x={9} y={-1} fill="#8a8a8a" letterSpacing="0.06em">S/C</text>
        <circle cx={32} cy={-3} r={3} fill={pColor} />
        <text x={38} y={-1} fill="#8a8a8a" letterSpacing="0.06em">TARGET</text>
        <circle cx={78} cy={-3} r={3} fill="#5b87e8" />
        <text x={84} y={-1} fill="#8a8a8a" letterSpacing="0.06em">EARTH</text>
        <line x1={116} y1={-3} x2={128} y2={-3} stroke="var(--cyan)" strokeWidth={2} />
        <text x={132} y={-1} fill="#8a8a8a" letterSpacing="0.06em">FLOWN</text>
        <line x1={168} y1={-3} x2={180} y2={-3} stroke="#2b2b2b" strokeWidth={1.4} strokeDasharray="3 3" />
        <text x={184} y={-1} fill="#8a8a8a" letterSpacing="0.06em">UNSEEN BY ML</text>
      </g>
    </svg>
  )
}

const mapShell = {
  width: '100%', height: '100%', display: 'flex',
  alignItems: 'center', justifyContent: 'center',
  background: 'var(--bg2)', border: '1px solid var(--border)',
}
const mapLabel = {
  fontSize: '11px', color: 'var(--text-dim)',
  fontFamily: 'var(--mono)', letterSpacing: '0.1em',
}

// ── Probability bar ───────────────────────────────────────────────────────────

function ProbBar({ prob, threshold }) {
  const pct = (prob ?? 0) * 100
  const col = probColor(prob)
  return (
    <div>
      <div style={{
        display: 'flex', justifyContent: 'space-between',
        fontSize: '9px', color: '#444', fontFamily: 'var(--mono)',
        marginBottom: '5px', letterSpacing: '0.06em',
      }}>
        <span>0.000</span>
        <span>THRESHOLD {threshold.toFixed(3)}</span>
        <span>1.000</span>
      </div>
      <div style={{ position: 'relative', height: '8px', background: '#0d0d0d', border: '1px solid #1e1e1e', borderRadius: '0' }}>
        <div style={{
          width: `${pct}%`, height: '100%',
          background: col,
          transition: 'width 0.14s linear, background 0.2s ease',
        }} />
        {/* Threshold tick */}
        <div style={{
          position: 'absolute', top: '-3px', bottom: '-3px',
          left: `${threshold * 100}%`, width: '1.5px',
          background: '#666', transform: 'translateX(-50%)',
        }} />
      </div>
    </div>
  )
}

// ── Timeline chart ────────────────────────────────────────────────────────────

const TT = {
  contentStyle: { background: '#0a0a0a', border: '1px solid #222', fontSize: 10, fontFamily: 'Share Tech Mono, monospace' },
  labelStyle:   { color: '#888' },
}

function ProbTimeline({ history, threshold, abortPct }) {
  return (
    <ResponsiveContainer width="100%" height={100}>
      <LineChart data={history} margin={{ left: -26, right: 8, top: 4, bottom: 0 }}>
        <CartesianGrid strokeDasharray="2 6" stroke="#141414" vertical={false} />
        <XAxis dataKey="elapsed_pct"
          tickFormatter={v => `${(v * 100).toFixed(0)}%`}
          tick={{ fontSize: 8, fill: '#444', fontFamily: 'Share Tech Mono, monospace' }} />
        <YAxis domain={[0, 1]}
          tick={{ fontSize: 8, fill: '#444' }} />
        <Tooltip {...TT}
          formatter={v => [v.toFixed(4), 'P(FAIL)']}
          labelFormatter={v => `ELAPSED ${(v * 100).toFixed(1)}%`} />
        <ReferenceLine y={threshold} stroke="#555" strokeDasharray="3 4" />
        {abortPct != null && (
          <ReferenceLine x={abortPct} stroke="var(--red)" strokeDasharray="3 4" />
        )}
        <Line type="monotone" dataKey="probability"
          stroke="var(--cyan)" strokeWidth={1.5} dot={false} isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  )
}

// ── Telemetry row ─────────────────────────────────────────────────────────────

function TelRow({ label, value, unit }) {
  return (
    <div style={{
      display: 'flex', justifyContent: 'space-between', alignItems: 'baseline',
      padding: '4px 0', borderBottom: '1px solid #0e0e0e',
    }}>
      <span style={{ fontSize: '9px', color: '#444', fontFamily: 'var(--mono)', letterSpacing: '0.06em' }}>
        {label}
      </span>
      <span style={{ fontSize: '11px', color: '#999', fontFamily: 'var(--mono)' }}>
        {value}
        {unit && <span style={{ fontSize: '8px', color: '#444', marginLeft: '2px' }}>{unit}</span>}
      </span>
    </div>
  )
}

// ── Mission item ──────────────────────────────────────────────────────────────

function MissionItem({ mission, status, isActive, onClick }) {
  const s   = status || 'idle'
  const ic  = statusIcon(s)
  const col = statusColor(s)
  const pCol = PLANET_COLORS[mission.target] || '#555'

  return (
    <div onClick={onClick} style={{
      padding: '8px 12px', cursor: 'pointer',
      borderLeft: `2px solid ${isActive ? 'var(--cyan)' : 'transparent'}`,
      background: isActive ? 'rgba(226,226,226,0.04)' : 'transparent',
      display: 'flex', alignItems: 'center', gap: '8px',
      transition: 'background 0.15s',
    }}
      onMouseEnter={e => { if (!isActive) e.currentTarget.style.background = 'rgba(255,255,255,0.02)' }}
      onMouseLeave={e => { if (!isActive) e.currentTarget.style.background = isActive ? 'rgba(226,226,226,0.04)' : 'transparent' }}
    >
      <span style={{ color: col, fontSize: '11px', fontFamily: 'var(--mono)', flexShrink: 0, width: '12px' }}>
        {ic}
      </span>
      <div style={{ flex: 1, minWidth: 0, overflow: 'hidden' }}>
        <div style={{ fontSize: '10px', color: '#aaa', fontFamily: 'var(--mono)', letterSpacing: '0.04em', display: 'flex', alignItems: 'center', gap: '5px' }}>
          {mission.generated
            ? <span style={{ fontSize: '8px', color: 'var(--cyan)', border: '1px solid var(--cyan)44', padding: '1px 3px' }}>GEN</span>
            : null}
          {mission.generated
            ? `${mission.target?.toUpperCase()} ${mission.dvSigma != null ? `${mission.dvSigma >= 0 ? '+' : ''}${mission.dvSigma.toFixed(2)}σ` : 'CUSTOM'}`
            : `#${mission.mission_id}`}
        </div>
        <div style={{ fontSize: '9px', color: '#555', fontFamily: 'var(--mono)', letterSpacing: '0.03em', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
          {mission.label === 1 ? 'NOM' : mission.label === 0 ? 'FAIL' : '?'} · {mission.failure_type || '—'}
        </div>
      </div>
      {mission.target && mission.target !== 'unknown' && (
        <span style={{
          fontSize: '8px', color: pCol, fontFamily: 'var(--mono)',
          letterSpacing: '0.06em', flexShrink: 0,
          border: `1px solid ${pCol}44`, padding: '1px 4px',
        }}>
          {mission.target.slice(0, 3).toUpperCase()}
        </span>
      )}
    </div>
  )
}

// ── Verdict banner ────────────────────────────────────────────────────────────

function VerdictBanner({ status, wasCorrect, trueLabel, finalProb, abortPct, abortProb, abortThreshold,
                         predMode, actualFailureType, modeCorrect, ood, oodFraction,
                         targetBody, totalSteps }) {
  if (status !== 'canceled' && status !== 'completed') return null
  const isAbort = status === 'canceled'
  const col = wasCorrect
    ? (isAbort ? 'var(--red)' : 'var(--green)')
    : 'var(--orange)'
  const headline = isAbort
    ? (wasCorrect ? '✗  ABORT — ML CORRECTLY PREDICTED FAILURE' : '✗  ABORT — FALSE POSITIVE (MISSION WAS NOMINAL)')
    : (wasCorrect ? '✓  MISSION GO — PREDICTION CORRECT'        : '△  PASSED — ML MISSED THE FAILURE')

  return (
    <div style={{ marginTop: '8px' }}>
      <div style={{
        padding: '13px 20px',
        background: `${col}0d`,
        border: `1px solid ${col}`,
        borderLeft: `4px solid ${col}`,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      }}>
        <div style={{
          fontSize: '12px', color: col, fontFamily: 'var(--mono)',
          letterSpacing: '0.1em', fontWeight: 700,
        }}>
          {headline}
        </div>
        <div style={{ display: 'flex', gap: '28px' }}>
          {[
            ['FINAL P(FAIL)', finalProb?.toFixed(4) ?? '—', col],
            ['TRUE OUTCOME', trueLabel === 1 ? 'SUCCESS' : trueLabel === 0 ? 'FAILURE' : '—',
              trueLabel === 1 ? 'var(--green)' : trueLabel === 0 ? 'var(--red)' : '#555'],
            ['ML CALL', isAbort ? 'ABORT' : 'GO', isAbort ? 'var(--red)' : 'var(--green)'],
          ].map(([k, v, c]) => (
            <div key={k} style={{ textAlign: 'right' }}>
              <div style={{ fontSize: '9px', color: '#555', fontFamily: 'var(--mono)', letterSpacing: '0.08em' }}>{k}</div>
              <div style={{ fontSize: '14px', color: c, fontFamily: 'var(--mono)' }}>{v}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Abort detail row */}
      {isAbort && (abortPct != null || abortProb != null || abortThreshold != null) && (
        <div style={{
          padding: '7px 20px',
          background: '#0a0a0a',
          borderLeft: `4px solid ${col}`,
          border: `1px solid #1a1a1a`,
          borderTop: 'none',
          display: 'flex', gap: '32px', alignItems: 'center',
        }}>
          <span style={{ fontSize: '9px', color: '#444', fontFamily: 'var(--mono)', letterSpacing: '0.08em' }}>
            ABORT DETAILS
          </span>
          {abortPct != null && (
            <span style={{ fontSize: '9px', fontFamily: 'var(--mono)', color: '#888' }}>
              ELAPSED: <span style={{ color: 'var(--red)' }}>{(abortPct * 100).toFixed(1)}%</span>
            </span>
          )}
          {abortProb != null && (
            <span style={{ fontSize: '9px', fontFamily: 'var(--mono)', color: '#888' }}>
              P(FAIL) AT ABORT: <span style={{ color: 'var(--red)' }}>{abortProb.toFixed(4)}</span>
            </span>
          )}
          {abortThreshold != null && (
            <span style={{ fontSize: '9px', fontFamily: 'var(--mono)', color: '#888' }}>
              CALIBRATED THRESHOLD: <span style={{ color: '#aaa' }}>{abortThreshold.toFixed(4)}</span>
            </span>
          )}
          {abortProb != null && abortThreshold != null && (
            <span style={{ fontSize: '9px', fontFamily: 'var(--mono)', color: '#555' }}>
              EXCESS: <span style={{ color: 'var(--orange)' }}>
                {(abortProb - abortThreshold >= 0 ? '+' : '')}{(abortProb - abortThreshold).toFixed(4)}
              </span>
            </span>
          )}
        </div>
      )}

      {/* Predicted failure mode — what the model expects to go wrong */}
      {predMode && (
        <div style={{
          padding: '7px 20px', background: '#0a0a0a',
          borderLeft: `4px solid ${col}`, border: '1px solid #1a1a1a', borderTop: 'none',
          display: 'flex', gap: '32px', alignItems: 'center', flexWrap: 'wrap',
        }}>
          <span style={{ fontSize: '9px', color: '#444', fontFamily: 'var(--mono)', letterSpacing: '0.08em' }}>
            FAILURE MODE
          </span>
          <span style={{ fontSize: '9px', fontFamily: 'var(--mono)', color: '#888' }}>
            PREDICTED: <span style={{ color: 'var(--orange)' }}>{String(predMode).toUpperCase()}</span>
          </span>
          {actualFailureType && (
            <span style={{ fontSize: '9px', fontFamily: 'var(--mono)', color: '#888' }}>
              ACTUAL: <span style={{ color: trueLabel === 1 ? 'var(--green)' : 'var(--red)' }}>
                {String(actualFailureType).toUpperCase()}
              </span>
            </span>
          )}
          {modeCorrect != null && (
            <span style={{ fontSize: '9px', fontFamily: 'var(--mono)',
                           color: modeCorrect ? 'var(--green)' : 'var(--red)' }}>
              {modeCorrect ? '✓ MODE CORRECT' : '✗ MODE WRONG'}
            </span>
          )}
        </div>
      )}

      {/* Why this mission was pruned — stated in full, not just as numbers */}
      {isAbort && abortPct != null && (
        <div style={{
          padding: '10px 20px 11px', background: '#0d0708',
          borderLeft: `4px solid ${col}`, border: '1px solid #1f1416', borderTop: 'none',
        }}>
          <div style={{ fontSize: '9px', fontFamily: 'var(--mono)',
                        letterSpacing: '0.1em', marginBottom: '6px', color: '#7a5560' }}>
            WHY THIS RUN WAS PRUNED
          </div>
          <div style={{ fontSize: '10.5px', color: '#b9b9b9', fontFamily: 'var(--mono)', lineHeight: 1.75 }}>
            After observing <span style={{ color: 'var(--cyan)' }}>{(abortPct * 100).toFixed(1)}%</span>
            {' '}of the trajectory
            {totalSteps ? <> (<span style={{ color: 'var(--cyan)' }}>
              {Math.round(abortPct * totalSteps)}</span> of {totalSteps} steps)</> : null}
            {targetBody ? <> for the <span style={{ color: '#ddd' }}>{targetBody.toUpperCase()}</span> transfer</> : null},
            {' '}the model put failure probability at{' '}
            <span style={{ color: 'var(--red)' }}>{abortProb?.toFixed(4)}</span>, which is{' '}
            <span style={{ color: 'var(--orange)' }}>
              {abortThreshold != null ? `${((abortProb - abortThreshold) >= 0 ? '+' : '')}${(abortProb - abortThreshold).toFixed(4)}` : '—'}
            </span>{' '}past this planet's calibrated abort threshold of{' '}
            <span style={{ color: '#aaa' }}>{abortThreshold?.toFixed(4)}</span>.
            {predMode && MODE_DESC[predMode] && (
              <> The expected failure is{' '}
                <span style={{ color: 'var(--orange)' }}>{predMode.toUpperCase().replace(/_/g, ' ')}</span>
                {' '}— {MODE_DESC[predMode]}.</>
            )}
          </div>
          <div style={{ marginTop: '7px', fontSize: '10px', fontFamily: 'var(--mono)',
                        color: wasCorrect ? 'var(--green)' : 'var(--red)', lineHeight: 1.6 }}>
            {wasCorrect
              ? <>✓ CORRECT — this mission does fail{actualFailureType && actualFailureType !== 'success'
                  ? ` (${actualFailureType.replace(/_/g, ' ')})` : ''}. Stopping here saves the remaining{' '}
                  <span style={{ color: 'var(--green)' }}>{((1 - abortPct) * 100).toFixed(0)}%</span> of propagation.</>
              : <>✗ FALSE PRUNE — this mission would have SUCCEEDED. A good run was discarded;
                  this is the expensive error class.</>}
          </div>
        </div>
      )}

      {/* Out-of-distribution advisory */}
      {ood && (
        <div style={{
          padding: '7px 20px', background: '#140d00',
          borderLeft: '4px solid var(--orange)', border: '1px solid #2a1d00', borderTop: 'none',
          display: 'flex', gap: '18px', alignItems: 'center', flexWrap: 'wrap',
        }}>
          <span style={{ fontSize: '9px', color: 'var(--orange)', fontFamily: 'var(--mono)', letterSpacing: '0.08em' }}>
            ⚠ OUT OF DISTRIBUTION
          </span>
          <span style={{ fontSize: '9px', fontFamily: 'var(--mono)', color: '#997744' }}>
            {(oodFraction * 100).toFixed(0)}% OF INPUTS BEYOND TRAINING RANGE — TREAT THIS VERDICT AS LOW CONFIDENCE
          </span>
        </div>
      )}
    </div>
  )
}

// ── Create Mission Form ───────────────────────────────────────────────────────

// Missions are defined the way the dataset defines them: a Hohmann nominal
// burn plus an execution error. The error is entered in sigma units of the
// dispersion the dataset actually sampled (Venus dv_V sigma = 0.003 km/s), so
// 0 is a textbook transfer and +-2 sigma is already a failure.
function CreateMissionForm({ onGenerate, onClose, disabled, planetInfo }) {
  const [target,     setTarget]     = useState('mars')
  const [dvSigma,    setDvSigma]    = useState(0)
  const [aopOff,     setAopOff]     = useState(0)
  const [incOff,     setIncOff]     = useState(0)
  const [generating, setGenerating] = useState(false)
  const [genError,   setGenError]   = useState(null)
  const [showAdv,    setShowAdv]    = useState(false)

  const info    = planetInfo?.[target] || {}
  const sigmaDv = info.sigma?.dv_V ?? 0.003
  const nominal = info.nominal?.TOI_V
  const trained = info.model_trained !== false
  const pColor  = PLANET_COLORS[target.charAt(0).toUpperCase() + target.slice(1)] || 'var(--cyan)'

  const dvKms = dvSigma * sigmaDv

  const handleTargetChange = (t) => {
    setTarget(t); setDvSigma(0); setAopOff(0); setIncOff(0); setGenError(null)
  }

  const handleSubmit = async () => {
    setGenerating(true); setGenError(null)
    try {
      const body = {
        target,
        dv_v_offset: dvKms,
        aop_offset:  parseFloat(aopOff) || 0,
        inc_offset:  parseFloat(incOff) || 0,
      }
      const r = await fetch(`${API}/api/simulator/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      const data = await r.json()
      if (!r.ok || data.error) throw new Error(data.error || `HTTP ${r.status}`)
      onGenerate({ ...data, dv_sigma: dvSigma })
    } catch (e) {
      console.error('Generate mission:', e)
      setGenError(String(e.message || e))
    } finally {
      setGenerating(false)
    }
  }

  return (
    <div style={{
      background: 'var(--bg2)', border: '1px solid var(--border)',
      borderLeft: `3px solid ${pColor}`,
      padding: '14px 18px', marginBottom: '0',
    }}>
      <div style={{ display: 'flex', gap: '20px', alignItems: 'flex-end', flexWrap: 'wrap' }}>

        {/* Target */}
        <div>
          <div style={labelStyle}>TARGET BODY</div>
          <select className="field-input"
            style={{ width: '110px', fontSize: '11px', background: 'var(--bg2)', color: pColor, fontFamily: 'var(--mono)', borderColor: `${pColor}88` }}
            value={target} onChange={e => handleTargetChange(e.target.value)}>
            {GEN_PLANETS.map(p => <option key={p} value={p}>{p.toUpperCase()}</option>)}
          </select>
        </div>

        {/* TOI burn execution error */}
        <div style={{ minWidth: '240px', flex: 1 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
            <div style={labelStyle}>TOI BURN ERROR  (σ)</div>
            <div style={{ fontSize: '9px', color: '#555', fontFamily: 'var(--mono)' }}>
              1σ = {sigmaDv.toFixed(4)} km/s
            </div>
          </div>
          <div style={{ position: 'relative' }}>
            <input type="range" min={-6} max={6} step={0.25}
              value={dvSigma} onChange={e => setDvSigma(parseFloat(e.target.value))}
              style={{ width: '100%', accentColor: pColor }} />
            {/* nominal tick at 0 sigma */}
            <div style={{
              position: 'absolute', top: '-2px', bottom: '-2px',
              left: '50%', width: '2px', background: `${pColor}88`,
              pointerEvents: 'none',
            }} />
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '3px' }}>
            <span style={{ fontSize: '9px', color: '#333', fontFamily: 'var(--mono)' }}>−6σ</span>
            <span style={{ fontSize: '11px', color: pColor, fontFamily: 'var(--mono)' }}>
              {dvSigma >= 0 ? '+' : ''}{dvSigma.toFixed(2)}σ = {dvKms >= 0 ? '+' : ''}{dvKms.toFixed(5)} km/s
            </span>
            <span style={{ fontSize: '9px', color: '#333', fontFamily: 'var(--mono)' }}>+6σ</span>
          </div>
        </div>

        {/* Advanced orientation offsets */}
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
            <div style={labelStyle}>ORIENTATION</div>
            <label style={{ display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer' }}>
              <input type="checkbox" checked={showAdv}
                onChange={e => setShowAdv(e.target.checked)}
                style={{ accentColor: pColor }} />
              <span style={{ fontSize: '9px', color: '#555', fontFamily: 'var(--mono)' }}>EDIT</span>
            </label>
          </div>
          <div style={{ display: 'flex', gap: '6px' }}>
            <input className="field-input" style={{ width: '68px' }} title="AOP offset (deg)"
              type="number" step={0.05} disabled={!showAdv}
              value={aopOff} onChange={e => setAopOff(parseFloat(e.target.value) || 0)} />
            <input className="field-input" style={{ width: '68px' }} title="Inclination offset (deg)"
              type="number" step={0.05} disabled={!showAdv}
              value={incOff} onChange={e => setIncOff(parseFloat(e.target.value) || 0)} />
          </div>
        </div>

        {/* Actions */}
        <div style={{ display: 'flex', gap: '8px', marginLeft: 'auto' }}>
          <button className="launch-btn"
            style={{ padding: '7px 18px', fontSize: '10px', borderColor: pColor, color: pColor }}
            onClick={handleSubmit} disabled={generating || disabled}>
            {generating ? 'GENERATING...' : '⊕ GENERATE'}
          </button>
          <button className="launch-btn"
            style={{ padding: '7px 12px', fontSize: '10px' }}
            onClick={onClose} disabled={generating}>
            ✕
          </button>
        </div>
      </div>

      {/* Status */}
      <div style={{ marginTop: '8px', fontSize: '9px', fontFamily: 'var(--mono)', color: '#555', letterSpacing: '0.06em' }}>
        {Math.abs(dvSigma) < 1
          ? <span style={{ color: 'var(--green)' }}>✓ WITHIN NOMINAL DISPERSION — TRANSFER SHOULD SUCCEED</span>
          : Math.abs(dvSigma) <= 5
            ? <span style={{ color: 'var(--red)' }}>⚠ {Math.abs(dvSigma).toFixed(1)}σ BURN ERROR — TRANSFER EXPECTED TO FAIL</span>
            : <span style={{ color: 'var(--amber, #d18616)' }}>⚠ {Math.abs(dvSigma).toFixed(1)}σ — BEYOND SAMPLED DISPERSION, MODEL WILL FLAG OUT-OF-DISTRIBUTION</span>
        }
        {nominal != null && <> {'  |  '}NOMINAL TOI_V {nominal.toFixed(4)} km/s</>}
        {!trained && <span style={{ color: 'var(--red)' }}>{'  |  '}NO TRAINED MODEL FOR THIS TARGET</span>}
      </div>
      {genError && (
        <div style={{ marginTop: '6px', fontSize: '9px', fontFamily: 'var(--mono)', color: 'var(--red)' }}>
          GENERATE FAILED: {genError}
        </div>
      )}
    </div>
  )
}


// ── Main component ────────────────────────────────────────────────────────────

export default function Simulator() {
  const [dataPath, setDataPath]       = useState(DEFAULT_DATA)
  const [n, setN]                     = useState(16)
  const [threshold, setThreshold]     = useState(0.443)  // P(fail) equiv of calibrated 0.557 P(success)
  const [minElapsed, setMinElapsed]   = useState(0.4)
  const [planetFilter, setPlanetFilter] = useState('ALL')
  const [apiOnline, setApiOnline]     = useState(null)
  const [planetInfo, setPlanetInfo]   = useState(null)

  const [missions, setMissions]               = useState([])
  const [missionResults, setMissionResults]   = useState({})
  const [loadingList, setLoadingList]         = useState(false)
  const [showCreateForm, setShowCreateForm]   = useState(false)

  const [activeMissionId, setActiveMissionId] = useState(null)
  const [trajectory, setTrajectory]           = useState(null)
  const [trajLoading, setTrajLoading]         = useState(false)

  const [stream, setStream]           = useState(INIT_STREAM)
  const [speedIdx, setSpeedIdx]       = useState(2)     // index into SPEEDS (default 1×)
  const [scrubIdx, setScrubIdx]       = useState(null)  // hover scrub position on orbital map

  const esRef          = useRef(null)
  const trajectoryRef  = useRef(null)
  const bufferRef      = useRef([])   // raw SSE steps: {elapsed_pct, probability, abortEvent?}
  const playTimerRef   = useRef(null)
  const playPosRef     = useRef(0)    // current display position in buffer
  const streamDoneRef  = useRef(false) // SSE fully received

  // ── API health check
  useEffect(() => {
    fetch(`${API}/api/health`, { signal: AbortSignal.timeout(3000) })
      .then(r => r.json()).then(() => setApiOnline(true)).catch(() => setApiOnline(false))
  }, [])

  // ── Planet nominals + sampled dispersions, for the mission creator sliders
  useEffect(() => {
    fetch(`${API}/api/simulator/planet_info`)
      .then(r => r.json()).then(setPlanetInfo).catch(() => setPlanetInfo(null))
  }, [])

  // ── Cleanup on unmount
  useEffect(() => () => {
    esRef.current?.close()
    if (playTimerRef.current) clearInterval(playTimerRef.current)
  }, [])

  // ── Keep trajectoryRef in sync
  useEffect(() => { trajectoryRef.current = trajectory }, [trajectory])

  // ── Load mission list
  const loadMissions = useCallback(async () => {
    setLoadingList(true)
    esRef.current?.close()
    setStream(INIT_STREAM)
    setActiveMissionId(null)
    setTrajectory(null)
    trajectoryRef.current = null
    setMissionResults({})
    try {
      const params = new URLSearchParams({
        n, seed: String(Date.now() % 99999),
        ...(dataPath ? { data: dataPath } : {}),
        ...(planetFilter !== 'ALL' ? { target: planetFilter } : {}),
      })
      const r = await fetch(`${API}/api/simulator/missions?${params}`)
      if (!r.ok) throw new Error(await r.text())
      const { missions: ms } = await r.json()
      setMissions(ms)
    } catch (e) { console.error('Load missions:', e) }
    finally { setLoadingList(false) }
  }, [n, dataPath, planetFilter])

  // ── Playback engine (stopPlayTimer must be declared before selectMission uses it)
  const stopPlayTimer = useCallback(() => {
    if (playTimerRef.current) { clearInterval(playTimerRef.current); playTimerRef.current = null }
  }, [])

  // ── Select mission → load trajectory
  const selectMission = useCallback(async (mission) => {
    if (['streaming', 'loading', 'paused'].includes(stream.status)) return
    esRef.current?.close()
    stopPlayTimer()
    bufferRef.current = []
    playPosRef.current = 0
    streamDoneRef.current = false
    setStream(INIT_STREAM)
    setActiveMissionId(mission.mission_id)
    setTrajectory(null)
    trajectoryRef.current = null
    setTrajLoading(true)
    try {
      const r = await fetch(
        `${API}/api/simulator/trajectory?mission_id=${mission.mission_id}` +
        (dataPath ? `&data=${encodeURIComponent(dataPath)}` : '')
      )
      if (!r.ok) throw new Error(await r.text())
      const data = await r.json()
      setTrajectory(data)
      trajectoryRef.current = data
    } catch (e) { console.error('Trajectory load:', e) }
    finally { setTrajLoading(false) }
  }, [stream.status, dataPath, stopPlayTimer])

  const advancePlayback = useCallback(() => {
    const buf  = bufferRef.current
    const pos  = playPosRef.current
    if (pos >= buf.length) {
      if (streamDoneRef.current) {
        stopPlayTimer()
        // Finalize completed status if no cancel event was in the buffer
        setStream(prev => {
          if (prev.status === 'streaming' || prev.status === 'paused') {
            return { ...prev, status: 'completed' }
          }
          return prev
        })
      }
      return
    }
    const step = buf[pos]
    playPosRef.current = pos + 1

    const traj = trajectoryRef.current
    const len  = traj?.positions?.length ?? 1
    const idx  = Math.max(0, Math.min(Math.round(step.elapsed_pct * (len - 1)), len - 1))

    if (step.isCancel) {
      setStream(prev => ({
        ...prev,
        currentProb:    step.probability,
        currentIdx:     idx,
        history:        [...prev.history, { elapsed_pct: step.elapsed_pct, probability: step.probability }],
        abortIdx:       idx,
        abortPct:       step.elapsed_pct,
        abortProb:      step.probability,
        abortThreshold: step.threshold_used ?? null,
        predMode:       step.predicted_failure_mode ?? prev.predMode,
        modeConf:       step.mode_confidence ?? prev.modeConf,
        status:         'canceled',
        playbackPos:    pos + 1,
      }))
      stopPlayTimer()
    } else {
      setStream(prev => ({
        ...prev,
        currentProb:  step.probability,
        currentIdx:   idx,
        history:      [...prev.history, { elapsed_pct: step.elapsed_pct, probability: step.probability }],
        playbackPos:  pos + 1,
      }))
    }
  }, [stopPlayTimer])

  const startPlayTimer = useCallback((delayMs) => {
    stopPlayTimer()
    playTimerRef.current = setInterval(advancePlayback, delayMs)
  }, [stopPlayTimer, advancePlayback])

  // ── Launch mission (SSE stream)
  const launchMission = useCallback(() => {
    const isActive = stream.status === 'streaming' || stream.status === 'loading'
    if (!activeMissionId || isActive || !trajectoryRef.current) return

    // Reset buffer and playback
    stopPlayTimer()
    bufferRef.current   = []
    playPosRef.current  = 0
    streamDoneRef.current = false
    esRef.current?.close()

    const speed = SPEEDS[speedIdx]
    setScrubIdx(null)
    setStream({ ...INIT_STREAM, status: 'loading' })
    setMissionResults(prev => ({ ...prev, [activeMissionId]: { status: 'streaming' } }))

    const url = `${API}/api/simulator/stream?` + new URLSearchParams({
      mission_id:      String(activeMissionId),
      ...(dataPath ? { data: dataPath } : {}),
      threshold:       String(threshold),
      min_elapsed_pct: String(minElapsed),
      step_delay_ms:   '10',   // always stream fast; playback speed is frontend-controlled
    })
    const es = new EventSource(url)
    esRef.current = es

    es.onmessage = (e) => {
      let ev
      try { ev = JSON.parse(e.data) } catch { return }

      if (ev.type === 'info') {
        setStream(prev => ({
          ...prev,
          status:       speed.delay === null ? 'paused' : 'streaming',
          calThreshold: ev.calibrated_threshold ?? null,
          regime:       ev.target_body ?? null,
          modelAvail:   ev.model_available !== false,
          trueLabel:    ev.true_label,
        }))
        // Start playback timer (unless step mode)
        if (speed.delay !== null) startPlayTimer(speed.delay)
      }

      if (ev.type === 'step') {
        bufferRef.current.push({ elapsed_pct: ev.elapsed_pct, probability: ev.probability })
      }

      if (ev.type === 'cancel') {
        bufferRef.current.push({
          elapsed_pct: ev.elapsed_pct, probability: ev.probability,
          isCancel: true, threshold_used: ev.threshold_used,
          predicted_failure_mode: ev.predicted_failure_mode,
          mode_confidence: ev.mode_confidence,
        })
      }

      if (ev.type === 'done') {
        es.close()
        streamDoneRef.current = true
        const mid = activeMissionId
        const wasCorrect = ev.was_correct
        const finalStatus = ev.canceled ? 'canceled' : 'completed'
        setStream(prev => {
          // If playback already consumed the cancel event, status is already canceled
          if (prev.status === 'canceled') {
            setMissionResults(p => ({ ...p, [mid]: { status: 'canceled', wasCorrect } }))
            return { ...prev, wasCorrect, trueLabel: ev.true_label, finalProb: ev.final_prob,
                     predMode: ev.predicted_failure_mode ?? prev.predMode,
                     actualFailureType: ev.actual_failure_type ?? null,
                     modeCorrect: ev.mode_correct ?? null,
                     ood: !!ev.out_of_distribution, oodFraction: ev.ood_fraction ?? 0 }
          }
          // Otherwise mark done when playback catches up (handled in advancePlayback)
          return {
            ...prev,
            wasCorrect,
            trueLabel:  ev.true_label,
            finalProb:  ev.final_prob,
            predMode:   ev.predicted_failure_mode ?? prev.predMode,
            actualFailureType: ev.actual_failure_type ?? null,
            modeCorrect: ev.mode_correct ?? null,
            ood: !!ev.out_of_distribution,
            oodFraction: ev.ood_fraction ?? 0,
            _pendingStatus: finalStatus,
            _pendingWasCorrect: wasCorrect,
          }
        })
        setMissionResults(prev => ({ ...prev, [mid]: { status: finalStatus, wasCorrect } }))
      }

      if (ev.type === 'error') {
        es.close()
        stopPlayTimer()
        setStream(prev => ({ ...prev, status: 'error' }))
      }
    }

    es.onerror = () => {
      es.close()
      stopPlayTimer()
      setStream(prev => (prev.status === 'loading' || prev.status === 'streaming' || prev.status === 'paused'
        ? { ...prev, status: 'error' } : prev))
    }
  }, [activeMissionId, dataPath, threshold, minElapsed, stream.status, speedIdx, startPlayTimer, stopPlayTimer])

  // ── Step forward one frame (STEP mode / when paused)
  const stepForward = useCallback(() => {
    advancePlayback()
  }, [advancePlayback])

  // ── Step backward one frame
  const stepBackward = useCallback(() => {
    const pos = Math.max(0, playPosRef.current - 2)
    if (pos < 0) return
    playPosRef.current = pos
    // Rewind history
    setStream(prev => {
      const newHistory = prev.history.slice(0, pos)
      const last = newHistory[newHistory.length - 1]
      return {
        ...prev,
        history:     newHistory,
        currentProb: last?.probability ?? null,
        playbackPos: pos,
      }
    })
  }, [])

  // ── Toggle pause/resume
  const togglePause = useCallback(() => {
    setStream(prev => {
      if (prev.status === 'paused') {
        const speed = SPEEDS[speedIdx]
        if (speed.delay !== null) startPlayTimer(speed.delay)
        return { ...prev, status: 'streaming' }
      }
      if (prev.status === 'streaming') {
        stopPlayTimer()
        return { ...prev, status: 'paused' }
      }
      return prev
    })
  }, [speedIdx, startPlayTimer, stopPlayTimer])

  // ── Stop stream manually
  const stopStream = useCallback(() => {
    esRef.current?.close()
    stopPlayTimer()
    setStream(prev => (['streaming', 'loading', 'paused'].includes(prev.status))
      ? { ...prev, status: 'idle' } : prev)
  }, [stopPlayTimer])

  // ── Reset current mission
  const resetMission = useCallback(() => {
    esRef.current?.close()
    stopPlayTimer()
    bufferRef.current = []
    playPosRef.current = 0
    streamDoneRef.current = false
    setStream(INIT_STREAM)
    if (activeMissionId) {
      setMissionResults(prev => { const n = { ...prev }; delete n[activeMissionId]; return n })
    }
  }, [activeMissionId, stopPlayTimer])

  // ── Handle generated mission from CreateMissionForm
  const handleGenerated = useCallback((data) => {
    setShowCreateForm(false)
    const genMission = {
      mission_id:   data.mission_id,
      label:        data.label,
      failure_type: data.failure_type,
      target:       data.target_body,
      generated:    true,
      dvSigma:      data.dv_sigma ?? null,
    }
    setMissions(prev => [genMission, ...prev])
    // Auto-select it
    esRef.current?.close()
    setStream(INIT_STREAM)
    setActiveMissionId(data.mission_id)
    setTrajectory({
      mission_id:   data.mission_id,
      total_steps:  data.total_steps,
      label:        data.label,
      failure_type: data.failure_type,
      target_body:  data.target_body,
      positions:    data.positions,
      telemetry:    data.telemetry,
    })
    trajectoryRef.current = {
      mission_id:  data.mission_id,
      total_steps: data.total_steps,
      positions:   data.positions,
      telemetry:   data.telemetry,
    }
  }, [])

  // ── Derived state
  const isActive    = ['streaming', 'loading', 'paused'].includes(stream.status)
  const isStreaming  = stream.status === 'streaming'
  const isPaused     = stream.status === 'paused'
  const isDone       = stream.status === 'canceled' || stream.status === 'completed'
  const isStepMode   = SPEEDS[speedIdx].delay === null
  const canLaunch    = !!activeMissionId && !!trajectory && !isActive && !trajLoading
  const activeMission = missions.find(m => m.mission_id === activeMissionId)

  // Telemetry: show hovered position when scrubbing, otherwise playback position
  const displayIdx   = scrubIdx ?? stream.currentIdx
  const telStep      = trajectory?.telemetry?.[displayIdx] ?? null
  const isScrubbing  = scrubIdx !== null

  // Summary stats
  const doneResults = Object.values(missionResults).filter(r => r.status !== 'streaming')
  const correctCount = doneResults.filter(r => r.wasCorrect).length

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="panel-content" style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>

      {/* ── Control bar ───────────────────────────────────────────────────── */}
      <div className="card" style={{ padding: '10px 16px', flexShrink: 0 }}>
        <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-end', flexWrap: 'wrap' }}>

          <div>
            <div style={labelStyle}>DATASET PATH — OPTIONAL OVERRIDE</div>
            <input className="field-input" style={{ width: '300px', fontSize: '10px' }}
              value={dataPath} onChange={e => setDataPath(e.target.value)}
              placeholder="server default ($ORBITGUARD_DATA)"
              disabled={isActive || loadingList} />
          </div>

          <div>
            <div style={labelStyle}>PLANET FILTER</div>
            <select className="field-input" style={{ width: '106px', fontSize: '10px', background: 'var(--bg2)', color: 'var(--text-dim)', fontFamily: 'var(--mono)' }}
              value={planetFilter} onChange={e => setPlanetFilter(e.target.value)}
              disabled={isActive}>
              {PLANET_LIST.map(p => <option key={p} value={p}>{p}</option>)}
            </select>
          </div>

          <div>
            <div style={labelStyle}>SAMPLE N</div>
            <input className="field-input" style={{ width: '52px' }}
              type="number" min={4} max={40}
              value={n} onChange={e => setN(parseInt(e.target.value) || 16)}
              disabled={isActive || loadingList} />
          </div>

          <div>
            <div style={labelStyle}>ABORT THRESHOLD</div>
            <input className="field-input" style={{ width: '64px' }}
              type="number" min={0.05} max={0.99} step={0.01}
              value={threshold} onChange={e => setThreshold(parseFloat(e.target.value) || 0.443)}
              disabled={isActive} />
          </div>

          <div>
            <div style={labelStyle}>MIN ELAPSED</div>
            <input className="field-input" style={{ width: '58px' }}
              type="number" min={0.05} max={1.0} step={0.05}
              value={minElapsed} onChange={e => setMinElapsed(parseFloat(e.target.value) || 0.4)}
              disabled={isActive} />
          </div>

          <div style={{ marginLeft: 'auto', display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' }}>
            {/* Speed selector */}
            <div style={{ display: 'flex', gap: '2px', border: '1px solid var(--border)', padding: '2px' }}>
              {SPEEDS.map((s, i) => (
                <button key={s.label}
                  onClick={() => {
                    setSpeedIdx(i)
                    // If currently streaming and switching to a different speed, restart timer
                    if (isStreaming && s.delay !== null) startPlayTimer(s.delay)
                    if (isStreaming && s.delay === null) { stopPlayTimer(); setStream(prev => ({ ...prev, status: 'paused' })) }
                  }}
                  style={{
                    padding: '4px 7px', fontSize: '9px', fontFamily: 'var(--mono)',
                    letterSpacing: '0.05em', cursor: 'pointer', border: 'none',
                    background: speedIdx === i ? 'var(--cyan)22' : 'transparent',
                    color: speedIdx === i ? 'var(--cyan)' : '#444',
                    outline: speedIdx === i ? '1px solid var(--cyan)44' : 'none',
                  }}>
                  {s.label}
                </button>
              ))}
            </div>

            {/* Pause/resume (during active stream only) */}
            {isActive && (isStreaming || isPaused) && (
              <button className="launch-btn"
                style={{
                  padding: '6px 12px', fontSize: '10px',
                  borderColor: isPaused ? 'var(--cyan)' : 'var(--border)',
                  color: isPaused ? 'var(--cyan)' : 'var(--text-dim)',
                }}
                onClick={togglePause}>
                {isPaused ? '▶ RESUME' : '⏸ PAUSE'}
              </button>
            )}

            {/* Step controls (only after sim completes — scrub mode) */}
            {isDone && (
              <div style={{ display: 'flex', gap: '4px' }}>
                <button className="launch-btn"
                  style={{ padding: '6px 10px', fontSize: '10px', minWidth: 0 }}
                  onClick={stepBackward} title="Step back">
                  ◀
                </button>
                <button className="launch-btn"
                  style={{ padding: '6px 10px', fontSize: '10px', minWidth: 0 }}
                  onClick={stepForward} title="Step forward">
                  ▶
                </button>
              </div>
            )}

            <div style={{ fontSize: '10px', fontFamily: 'var(--mono)', letterSpacing: '0.06em',
              color: apiOnline ? 'var(--green)' : apiOnline === false ? 'var(--red)' : '#444' }}>
              ● API {apiOnline == null ? '—' : apiOnline ? 'ONLINE' : 'OFFLINE'}
            </div>

            <button className="launch-btn" style={{ padding: '7px 14px', fontSize: '10px' }}
              onClick={loadMissions}
              disabled={loadingList || isActive || apiOnline === false}>
              {loadingList ? 'LOADING...' : '↺ LOAD'}
            </button>

            <button className="launch-btn" style={{
              padding: '7px 14px', fontSize: '10px',
              borderColor: showCreateForm ? 'var(--cyan)' : 'var(--border)',
              color: showCreateForm ? 'var(--cyan)' : 'var(--text-dim)',
            }}
              onClick={() => setShowCreateForm(v => !v)}
              disabled={isActive}>
              ⊕ NEW MISSION
            </button>

            <button className="launch-btn" style={{
              padding: '7px 20px', fontSize: '10px',
              background: canLaunch ? 'var(--cyan)12' : 'transparent',
              borderColor: canLaunch ? 'var(--cyan)' : 'var(--border)',
              color: canLaunch ? 'var(--cyan)' : 'var(--text-dim)',
            }}
              onClick={launchMission} disabled={!canLaunch}>
              ▶ LAUNCH
            </button>

            {isActive && (
              <button className="launch-btn"
                style={{ padding: '7px 14px', fontSize: '10px', borderColor: 'var(--red)', color: 'var(--red)' }}
                onClick={stopStream}>
                ■ STOP
              </button>
            )}

            {isDone && (
              <button className="launch-btn" style={{ padding: '7px 14px', fontSize: '10px' }}
                onClick={resetMission}>
                ↺ RESET
              </button>
            )}
          </div>
        </div>
      </div>

      {/* ── Mission creator ───────────────────────────────────────────────── */}
      {showCreateForm && (
        <CreateMissionForm
          onGenerate={handleGenerated}
          onClose={() => setShowCreateForm(false)}
          planetInfo={planetInfo}
          disabled={isStreaming} />
      )}

      {/* ── Main 3-column body ────────────────────────────────────────────── */}
      <div style={{ display: 'grid', gridTemplateColumns: '210px 1fr 250px', gap: '10px', flex: 1, minHeight: '430px' }}>

        {/* LEFT — Mission Queue */}
        <div style={{ display: 'flex', flexDirection: 'column', border: '1px solid var(--border)', background: 'var(--bg2)', overflow: 'hidden' }}>
          <div style={{ padding: '8px 12px', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between' }}>
            <span style={{ fontSize: '9px', color: 'var(--text-dim)', fontFamily: 'var(--mono)', letterSpacing: '0.1em' }}>
              [ MISSION QUEUE ]
            </span>
            <span style={{ fontSize: '9px', color: '#444', fontFamily: 'var(--mono)' }}>
              {missions.length}
            </span>
          </div>

          <div style={{ flex: 1, overflowY: 'auto' }}>
            {missions.length === 0 ? (
              <div style={{ padding: '30px 12px', textAlign: 'center', fontSize: '10px', color: '#333', fontFamily: 'var(--mono)', lineHeight: 2 }}>
                LOAD MISSIONS<br />TO POPULATE
              </div>
            ) : (
              missions.map(m => {
                const res = missionResults[m.mission_id]
                const s = m.mission_id === activeMissionId
                  ? (isStreaming ? 'streaming' : isDone ? stream.status : 'selected')
                  : (res?.status || 'idle')
                return (
                  <MissionItem key={m.mission_id} mission={m} status={s}
                    isActive={m.mission_id === activeMissionId}
                    onClick={() => selectMission(m)} />
                )
              })
            )}
          </div>

          {/* Queue score */}
          {doneResults.length > 0 && (
            <div style={{ borderTop: '1px solid var(--border)', padding: '8px 12px', flexShrink: 0 }}>
              <div style={{ fontSize: '9px', color: 'var(--text-dim)', fontFamily: 'var(--mono)', letterSpacing: '0.06em', marginBottom: '6px' }}>
                SESSION SCORE
              </div>
              <div style={{ fontSize: '18px', color: 'var(--cyan)', fontFamily: 'var(--mono)' }}>
                {correctCount}/{doneResults.length}
              </div>
              <div style={{ fontSize: '9px', color: '#555', fontFamily: 'var(--mono)', marginTop: '2px' }}>
                CORRECT CALLS
              </div>
            </div>
          )}
        </div>

        {/* CENTER — Orbital map + timeline */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', minHeight: 0 }}>
          <div style={{ flex: 1, minHeight: '320px' }}>
            <OrbitalMap
              positions={trajectory?.positions}
              currentIdx={displayIdx}
              abortIdx={stream.abortIdx}
              isLoading={trajLoading}
              targetName={trajectory?.target_body ?? activeMission?.target}
              onScrub={isDone ? setScrubIdx : null}
              operatingFrac={minElapsed}
            />
          </div>

          {/* Probability timeline */}
          <div style={{ background: 'var(--bg2)', border: '1px solid var(--border)', padding: '10px 12px 6px', flexShrink: 0 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
              <span style={{ fontSize: '9px', color: 'var(--text-dim)', fontFamily: 'var(--mono)', letterSpacing: '0.1em' }}>
                P(FAIL) OVER TRAJECTORY
              </span>
              {stream.abortPct != null && (
                <span style={{ fontSize: '9px', color: 'var(--red)', fontFamily: 'var(--mono)', letterSpacing: '0.06em' }}>
                  ABORT @ {(stream.abortPct * 100).toFixed(1)}%
                </span>
              )}
              {stream.history.length > 0 && stream.abortPct == null && (
                <span style={{ fontSize: '9px', color: '#444', fontFamily: 'var(--mono)' }}>
                  {stream.history.length} STEPS
                </span>
              )}
            </div>
            {stream.history.length === 0 ? (
              <div style={{ height: '100px', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '10px', color: '#2a2a2a', fontFamily: 'var(--mono)', letterSpacing: '0.08em' }}>
                AWAITING STREAM
              </div>
            ) : (
              <ProbTimeline history={stream.history} threshold={threshold} abortPct={stream.abortPct} />
            )}
          </div>
        </div>

        {/* RIGHT — P(fail) gauge + mission info + telemetry */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>

          {/* Big probability readout */}
          <div style={{ background: 'var(--bg2)', border: '1px solid var(--border)', padding: '16px 18px', flexShrink: 0 }}>
            <div style={{ fontSize: '9px', color: 'var(--text-dim)', fontFamily: 'var(--mono)', letterSpacing: '0.1em', marginBottom: '12px' }}>
              FAILURE PROBABILITY
            </div>
            <div style={{
              fontSize: '44px', fontFamily: 'var(--mono)', letterSpacing: '-0.02em', lineHeight: 1,
              marginBottom: '12px',
              color: probColor(stream.currentProb),
              textShadow: `0 0 22px ${probColor(stream.currentProb)}55`,
              transition: 'color 0.18s ease, text-shadow 0.18s ease',
            }}>
              {stream.currentProb != null ? stream.currentProb.toFixed(3) : '—'}
            </div>

            <ProbBar prob={stream.currentProb} threshold={threshold} />

            <div style={{ marginTop: '10px', fontSize: '10px', fontFamily: 'var(--mono)', letterSpacing: '0.07em', minHeight: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span>
                {stream.status === 'idle'      && <span style={{ color: '#444' }}>AWAITING LAUNCH</span>}
                {stream.status === 'loading'   && <span style={{ color: '#666' }}>◌ CONNECTING...</span>}
                {stream.status === 'streaming' && <span style={{ color: 'var(--cyan)' }}>● STREAMING</span>}
                {stream.status === 'paused'    && <span style={{ color: '#aacc00' }}>⏸ PAUSED</span>}
                {stream.status === 'canceled'  && <span style={{ color: 'var(--red)' }}>■ ML ABORTED</span>}
                {stream.status === 'completed' && <span style={{ color: 'var(--green)' }}>✓ COMPLETE</span>}
                {stream.status === 'error'     && <span style={{ color: 'var(--orange)' }}>△ STREAM ERROR</span>}
              </span>
              {stream.calThreshold != null && (
                <span style={{ fontSize: '9px', color: '#444' }}>
                  CAL THR {stream.calThreshold.toFixed(4)}
                </span>
              )}
            </div>
            {stream.regime && (
              <div style={{ marginTop: '4px', fontSize: '9px', color: '#333', fontFamily: 'var(--mono)', letterSpacing: '0.06em' }}>
                MODEL: {stream.regime.toUpperCase()} {stream.modelAvail === false ? '— NOT TRAINED' : ''}
              </div>
            )}
          </div>

          {/* Mission info card */}
          {activeMission ? (
            <div style={{ background: 'var(--bg2)', border: '1px solid var(--border)', padding: '12px 16px', flexShrink: 0 }}>
              <div style={{ fontSize: '9px', color: 'var(--text-dim)', fontFamily: 'var(--mono)', letterSpacing: '0.1em', marginBottom: '8px' }}>
                ACTIVE MISSION
              </div>
              <div style={{ fontSize: '12px', color: 'var(--cyan)', fontFamily: 'var(--mono)', marginBottom: '8px', letterSpacing: '0.04em' }}>
                MID #{activeMission.mission_id}
              </div>
              {[
                ['TARGET',       activeMission.target || '—'],
                ['TRUE OUTCOME', activeMission.label === 1 ? 'NOMINAL' : activeMission.label === 0 ? 'FAILURE' : '?'],
                ['FAILURE TYPE', activeMission.failure_type || '—'],
                ['TRAJECTORY',   trajectory ? `${trajectory.total_steps} STEPS` : trajLoading ? 'LOADING...' : 'NOT LOADED'],
              ].map(([k, v]) => (
                <div key={k} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', fontFamily: 'var(--mono)', padding: '2px 0', borderBottom: '1px solid #0e0e0e' }}>
                  <span style={{ color: '#444', letterSpacing: '0.05em' }}>{k}</span>
                  <span style={{ color: '#999' }}>{v}</span>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ background: 'var(--bg2)', border: '1px solid var(--border)', padding: '20px 16px', flexShrink: 0, textAlign: 'center' }}>
              <div style={{ fontSize: '10px', color: '#333', fontFamily: 'var(--mono)', letterSpacing: '0.08em' }}>
                SELECT A MISSION<br />FROM THE QUEUE
              </div>
            </div>
          )}

          {/* Live telemetry */}
          <div style={{ background: 'var(--bg2)', border: `1px solid ${isScrubbing ? 'var(--cyan)44' : 'var(--border)'}`, padding: '12px 16px', flex: 1 }}>
            <div style={{ fontSize: '9px', color: isScrubbing ? 'var(--cyan)' : 'var(--text-dim)', fontFamily: 'var(--mono)', letterSpacing: '0.1em', marginBottom: '8px', display: 'flex', justifyContent: 'space-between' }}>
              <span>{isScrubbing ? '⊹ SCRUB TELEMETRY' : 'LIVE TELEMETRY'}</span>
              {isScrubbing && telStep && (
                <span style={{ color: '#555' }}>{(telStep.elapsed_pct * 100).toFixed(1)}%</span>
              )}
            </div>
            <TelRow label="VEL_MAG"          value={fmtVal(telStep?.vel_mag,           2)} unit="km/s" />
            <TelRow label="ECC"              value={fmtVal(telStep?.ecc,               5)} unit=""     />
            <TelRow label="SPEC_ENERGY"      value={fmtVal(telStep?.spec_energy,       1)} unit="J/kg" />
            <TelRow label="EARTH_RMAG"       value={fmtVal(telStep?.earth_rmag,        0)} unit="km"   />
            <TelRow label="FPA"              value={fmtVal(telStep?.fpa_deg,           4)} unit="deg"  />
            <TelRow label="NORM_TARGET_DIST" value={fmtVal(telStep?.norm_target_dist,  5)} unit=""     />
            {!telStep && (
              <div style={{ fontSize: '9px', color: '#2a2a2a', fontFamily: 'var(--mono)', textAlign: 'center', marginTop: '12px', letterSpacing: '0.08em' }}>
                STREAM TO POPULATE
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ── Verdict banner ────────────────────────────────────────────────── */}
      <VerdictBanner
        status={stream.status}
        wasCorrect={stream.wasCorrect}
        trueLabel={stream.trueLabel}
        finalProb={stream.finalProb}
        abortPct={stream.abortPct}
        abortProb={stream.abortProb}
        predMode={stream.predMode}
        actualFailureType={stream.actualFailureType}
        modeCorrect={stream.modeCorrect}
        ood={stream.ood}
        oodFraction={stream.oodFraction}
        targetBody={stream.regime}
        totalSteps={stream.buffer?.length || null}
        abortThreshold={stream.abortThreshold}
      />
    </div>
  )
}

const labelStyle = {
  fontSize: '9px', color: '#555', fontFamily: 'var(--mono)',
  letterSpacing: '0.08em', marginBottom: '4px',
}
