import { useEffect, useRef } from 'react'

export function FlowFieldBackground({
  color = '#4de8e8',
  trailOpacity = 0.18,
  particleCount = 120,
  speed = 1.2,
  className = '',
}) {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')

    let width, height, animId
    const mouse = { x: -9999, y: -9999 }

    const resize = () => {
      width = canvas.width = canvas.offsetWidth
      height = canvas.height = canvas.offsetHeight
    }
    resize()
    window.addEventListener('resize', resize)

    const onMouseMove = e => {
      const rect = canvas.getBoundingClientRect()
      mouse.x = e.clientX - rect.left
      mouse.y = e.clientY - rect.top
    }
    window.addEventListener('mousemove', onMouseMove)

    // Parse hex color to rgb
    const hexToRgb = hex => {
      const r = parseInt(hex.slice(1, 3), 16)
      const g = parseInt(hex.slice(3, 5), 16)
      const b = parseInt(hex.slice(5, 7), 16)
      return { r, g, b }
    }
    const { r, g, b } = hexToRgb(color)

    // Flow field resolution
    const CELL = 28

    // Particles
    const particles = Array.from({ length: particleCount }, () => ({
      x: Math.random() * (width || 1200),
      y: Math.random() * (height || 800),
      vx: 0,
      vy: 0,
      life: Math.random(),
      maxLife: 0.6 + Math.random() * 0.4,
      alpha: 0,
    }))

    let t = 0

    const getAngle = (x, y, time) => {
      const nx = x / (CELL * 8)
      const ny = y / (CELL * 8)
      return (
        Math.cos(nx + time * 0.3) * Math.PI +
        Math.sin(ny * 1.4 + time * 0.2) * Math.PI +
        Math.cos((nx + ny) * 0.7 + time * 0.15) * Math.PI * 0.5
      )
    }

    const draw = () => {
      t += 0.004 * speed

      // Trail
      ctx.fillStyle = `rgba(9,9,9,${trailOpacity})`
      ctx.fillRect(0, 0, width, height)

      for (const p of particles) {
        // Flow field angle at particle position
        const angle = getAngle(p.x, p.y, t)
        const flowX = Math.cos(angle) * speed
        const flowY = Math.sin(angle) * speed

        // Mouse repulsion
        const dx = p.x - mouse.x
        const dy = p.y - mouse.y
        const dist = Math.sqrt(dx * dx + dy * dy)
        const repelRadius = 120
        let repelX = 0, repelY = 0
        if (dist < repelRadius && dist > 0) {
          const force = (1 - dist / repelRadius) * 2.5
          repelX = (dx / dist) * force
          repelY = (dy / dist) * force
        }

        p.vx = p.vx * 0.85 + (flowX + repelX) * 0.15
        p.vy = p.vy * 0.85 + (flowY + repelY) * 0.15

        p.x += p.vx
        p.y += p.vy
        p.life += 0.004 * speed

        // Reset when out of bounds or life exceeded
        if (
          p.x < -10 || p.x > width + 10 ||
          p.y < -10 || p.y > height + 10 ||
          p.life > p.maxLife
        ) {
          p.x = Math.random() * width
          p.y = Math.random() * height
          p.vx = 0
          p.vy = 0
          p.life = 0
          p.maxLife = 0.6 + Math.random() * 0.4
        }

        // Alpha envelope: fade in then fade out
        const lifeRatio = p.life / p.maxLife
        p.alpha = lifeRatio < 0.2
          ? lifeRatio / 0.2
          : lifeRatio > 0.8
            ? 1 - (lifeRatio - 0.8) / 0.2
            : 1

        const a = p.alpha * 0.55
        ctx.beginPath()
        ctx.arc(p.x, p.y, 1.4, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(${r},${g},${b},${a})`
        ctx.fill()
      }

      // Subtle connecting lines between nearby particles
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x
          const dy = particles[i].y - particles[j].y
          const d = Math.sqrt(dx * dx + dy * dy)
          if (d < 80) {
            const a = (1 - d / 80) * 0.08 * particles[i].alpha * particles[j].alpha
            ctx.beginPath()
            ctx.moveTo(particles[i].x, particles[i].y)
            ctx.lineTo(particles[j].x, particles[j].y)
            ctx.strokeStyle = `rgba(${r},${g},${b},${a})`
            ctx.lineWidth = 0.6
            ctx.stroke()
          }
        }
      }

      animId = requestAnimationFrame(draw)
    }

    draw()

    return () => {
      cancelAnimationFrame(animId)
      window.removeEventListener('resize', resize)
      window.removeEventListener('mousemove', onMouseMove)
    }
  }, [color, trailOpacity, particleCount, speed])

  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        zIndex: 0,
      }}
    />
  )
}

export default FlowFieldBackground
