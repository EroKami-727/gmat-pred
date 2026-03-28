import React from 'react'

export function Globe({ size = 440 }) {
  // Scale animation and shadow values relative to original 250px design
  const scale = size / 250
  const scrollPx = Math.round(400 * scale)

  const styleId = 'orbitguard-globe-tex-styles'
  React.useEffect(() => {
    if (typeof document === 'undefined') return
    if (document.getElementById(styleId)) return
    const style = document.createElement('style')
    style.id = styleId
    style.innerHTML = `
      @keyframes earthRotateTex {
        0%   { background-position: 0 0; }
        100% { background-position: ${scrollPx}px 0; }
      }
      @keyframes starTwinkle      { 0%,100% { opacity:0.15; } 50% { opacity:0.9; } }
      @keyframes starTwinkleSlow  { 0%,100% { opacity:0.1;  } 50% { opacity:0.8; } }
      @keyframes starTwinkleFast  { 0%,100% { opacity:0.1;  } 50% { opacity:1;   } }
    `
    document.head.appendChild(style)
  }, [scrollPx])

  // Atmosphere box-shadow — scale the large offset values with size
  const boxShadow = [
    `0 0 ${Math.round(20 * scale)}px rgba(255,255,255,0.2)`,
    `-${Math.round(5 * scale)}px 0 ${Math.round(8 * scale)}px #c3f4ff inset`,
    `${Math.round(15 * scale)}px ${Math.round(2 * scale)}px ${Math.round(25 * scale)}px #000 inset`,
    `-${Math.round(24 * scale)}px -${Math.round(2 * scale)}px ${Math.round(34 * scale)}px #c3f4ff99 inset`,
    `${size}px 0 ${Math.round(44 * scale)}px #00000066 inset`,
    `${Math.round(150 * scale)}px 0 ${Math.round(38 * scale)}px #000000aa inset`,
  ].join(', ')

  const s = size

  return (
    <div style={{ position: 'relative', width: s, height: s, flexShrink: 0 }}>
      {/* Outer atmospheric glow ring */}
      <div style={{
        position: 'absolute',
        inset: -Math.round(12 * scale),
        borderRadius: '50%',
        background: 'radial-gradient(circle, rgba(0,212,255,0.07) 0%, transparent 70%)',
        pointerEvents: 'none',
      }} />

      {/* Globe sphere with real Earth texture */}
      <div
        style={{
          width: s,
          height: s,
          borderRadius: '50%',
          overflow: 'hidden',
          backgroundImage: "url('https://pub-940ccf6255b54fa799a9b01050e6c227.r2.dev/globe.jpeg')",
          backgroundSize: 'cover',
          backgroundPosition: 'left',
          animation: `earthRotateTex 30s linear infinite`,
          boxShadow,
          position: 'relative',
        }}
      >
        {/* Twinkling star dots (positioned relative to sphere) */}
        {[
          { left: -20,        top: s * 0.1,  dur: '3s',   anim: 'starTwinkle' },
          { left: -40,        top: s * 0.22, dur: '2s',   anim: 'starTwinkleSlow' },
          { left: s * 0.8,   top: s * 0.36, dur: '4s',   anim: 'starTwinkle' },
          { left: s * 0.45,  top: s * 1.05, dur: '3s',   anim: 'starTwinkle' },
          { left: s * 0.2,   top: s * 0.95, dur: '1.5s', anim: 'starTwinkleFast' },
          { left: s * 0.95,  top: -s * 0.1, dur: '4s',   anim: 'starTwinkle' },
          { left: s * 1.08,  top: s * 0.24, dur: '2s',   anim: 'starTwinkleSlow' },
        ].map((star, i) => (
          <div
            key={i}
            style={{
              position: 'absolute',
              left: star.left,
              top: star.top,
              width: Math.round(4 * scale),
              height: Math.round(4 * scale),
              background: 'white',
              borderRadius: '50%',
              animation: `${star.anim} ${star.dur} infinite`,
            }}
          />
        ))}
      </div>

      {/* Orbit ring */}
      <div style={{
        position: 'absolute',
        top: '50%', left: '50%',
        width: s * 1.35, height: s * 1.35,
        borderRadius: '50%',
        border: '1px solid rgba(0,212,255,0.18)',
        transform: 'translate(-50%,-50%) rotateX(72deg)',
        pointerEvents: 'none',
      }} />
    </div>
  )
}

export default Globe
