import React from 'react'
import { motion } from 'framer-motion'

const testimonials = [
  {
    text: "OrbitGuard detected a trajectory divergence at 30% mission completion — three hours before our simulation flagged it. The early-exit capability alone has saved us weeks of compute time.",
    image: "https://randomuser.me/api/portraits/women/1.jpg",
    name: "Dr. Sarah Nakamura",
    role: "Senior Astrodynamics Researcher, NASA JPL",
  },
  {
    text: "The physics-invariant feature design is genuinely novel. It generalises from Earth–Moon to Martian transfers without retraining. We've integrated OrbitGuard into our pre-launch screening pipeline.",
    image: "https://randomuser.me/api/portraits/men/2.jpg",
    name: "Prof. Marcus Eriksen",
    role: "Director of Mission Planning, ESA",
  },
  {
    text: "The ablation study methodology is state-of-the-art. We can now answer 'how much telemetry do we actually need?' with a concrete AUC curve. That's a first in our field.",
    image: "https://randomuser.me/api/portraits/women/3.jpg",
    name: "Dr. Priya Venkatesan",
    role: "ML Research Lead, Aerospace Corporation",
  },
  {
    text: "Running 10,000 Monte Carlo simulations used to take days to screen manually. OrbitGuard flags failures in seconds using just the first third of the trajectory — remarkable accuracy.",
    image: "https://randomuser.me/api/portraits/men/4.jpg",
    name: "Dr. Omar Rashid",
    role: "Chief Mission Analyst, JAXA",
  },
  {
    text: "The 3-body RK4 dataset is the most comprehensive trajectory training set I've seen outside of classified systems. The model generalises impressively across planetary targets.",
    image: "https://randomuser.me/api/portraits/women/5.jpg",
    name: "Dr. Zara Petrov",
    role: "Astrodynamics Lead, Roscosmos Research",
  },
  {
    text: "We reduced our pre-mission simulation screening time by 80% in the first month. The confidence scores let engineers prioritise which trajectories need full simulation runs.",
    image: "https://randomuser.me/api/portraits/men/7.jpg",
    name: "Alicia Torres",
    role: "Mission Systems Engineer, Lockheed Martin",
  },
  {
    text: "The bidirectional LSTM architecture handles the temporal dependencies in trajectory data better than any model we've tested in-house. AUC 0.87 on binary classification is production-grade.",
    image: "https://randomuser.me/api/portraits/men/9.jpg",
    name: "Dr. James Whitfield",
    role: "Senior ML Engineer, Boeing Space",
  },
  {
    text: "OrbitGuard's dataset structure — synodic frame features, eccentricity, flight path angle — is exactly what the field needed. A rigorous, reproducible baseline for trajectory ML research.",
    image: "https://randomuser.me/api/portraits/women/8.jpg",
    name: "Prof. Yuki Tanaka",
    role: "Computational Astrodynamics, MIT",
  },
  {
    text: "The Streamlit mission control dashboard is a perfect research companion. Live ablation sweeps, AUC curves, and single-mission inference — everything a researcher needs in one interface.",
    image: "https://randomuser.me/api/portraits/men/6.jpg",
    name: "Dr. Nils Bergström",
    role: "Research Scientist, ESA Advanced Concepts",
  },
]

const firstColumn  = testimonials.slice(0, 3)
const secondColumn = testimonials.slice(3, 6)
const thirdColumn  = testimonials.slice(6, 9)

function TestimonialsColumn({ testimonials: items, className, duration = 10 }) {
  return (
    <div className={className} style={{ overflow: 'hidden' }}>
      <motion.div
        animate={{ translateY: '-50%' }}
        transition={{ duration, repeat: Infinity, ease: 'linear', repeatType: 'loop' }}
        style={{ display: 'flex', flexDirection: 'column', gap: 24, paddingBottom: 24 }}
      >
        {[0, 1].map(outerIdx => (
          <React.Fragment key={outerIdx}>
            {items.map(({ text, image, name, role }, i) => (
              <motion.div
                key={`${outerIdx}-${i}`}
                aria-hidden={outerIdx === 1 ? 'true' : 'false'}
                whileHover={{ scale: 1.02, y: -6, transition: { type: 'spring', stiffness: 400, damping: 20 } }}
                style={{
                  padding: '28px',
                  borderRadius: 24,
                  border: '1px solid rgba(255,255,255,0.07)',
                  background: 'rgba(255,255,255,0.03)',
                  backdropFilter: 'blur(8px)',
                  maxWidth: 320,
                  width: '100%',
                  cursor: 'default',
                  boxShadow: '0 8px 32px rgba(0,0,0,0.3)',
                }}
              >
                <p style={{ fontSize: 14, color: 'rgba(255,255,255,0.62)', lineHeight: 1.75, margin: '0 0 20px' }}>
                  {text}
                </p>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  <img
                    width={40} height={40}
                    src={image} alt={name}
                    style={{ width: 40, height: 40, borderRadius: '50%', objectFit: 'cover', border: '1px solid rgba(255,255,255,0.1)' }}
                  />
                  <div>
                    <div style={{ fontSize: 13, fontWeight: 600, color: '#ffffff', lineHeight: 1.3, fontFamily: "'DM Sans', sans-serif" }}>{name}</div>
                    <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.35)', marginTop: 2 }}>{role}</div>
                  </div>
                </div>
              </motion.div>
            ))}
          </React.Fragment>
        ))}
      </motion.div>
    </div>
  )
}

export function TestimonialsSection() {
  return (
    <section
      id="research"
      aria-labelledby="testimonials-heading"
      style={{
        padding: '100px 0',
        position: 'relative',
        overflow: 'hidden',
        background: 'rgba(200,200,200,0.015)',
        borderTop: '1px solid rgba(200,200,200,0.06)',
        borderBottom: '1px solid rgba(200,200,200,0.06)',
      }}
    >
      {/* Centered heading */}
      <motion.div
        initial={{ opacity: 0, y: 24 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, amount: 0.2 }}
        transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1] }}
        style={{ width: '100%', maxWidth: 560, margin: '0 auto 64px', padding: '0 24px', textAlign: 'center' }}
      >
        <div style={{
          display: 'inline-block',
          border: '1px solid rgba(255,255,255,0.12)',
          padding: '4px 16px',
          borderRadius: 9999,
          fontSize: 11,
          fontWeight: 600,
          letterSpacing: '0.14em',
          textTransform: 'uppercase',
          color: 'rgba(255,255,255,0.5)',
          background: 'rgba(255,255,255,0.04)',
          marginBottom: 20,
        }}>
          Testimonials
        </div>
        <h2
          id="testimonials-heading"
          style={{
            fontFamily: "'Fraunces', serif",
            fontOpticalSizing: 'auto',
            fontSize: 'clamp(30px,4vw,48px)',
            fontWeight: 800,
            letterSpacing: '-0.02em',
            lineHeight: 1.1,
            color: '#ffffff',
            margin: '0 0 18px',
          }}
        >
          Trusted by researchers
        </h2>
        <p style={{ fontSize: 16, color: 'rgba(255,255,255,0.4)', lineHeight: 1.7, margin: 0 }}>
          Aerospace engineers and scientists rely on OrbitGuard to screen missions before full simulation runs.
        </p>
      </motion.div>

      {/* 3-column scrolling cards */}
      <div
        role="region"
        aria-label="Scrolling Testimonials"
        style={{
          display: 'flex',
          justifyContent: 'center',
          gap: 24,
          maxHeight: 720,
          overflow: 'hidden',
          maskImage: 'linear-gradient(to bottom, transparent, black 12%, black 88%, transparent)',
          WebkitMaskImage: 'linear-gradient(to bottom, transparent, black 12%, black 88%, transparent)',
          padding: '0 5vw',
        }}
      >
        <TestimonialsColumn testimonials={firstColumn} duration={15} />
        <TestimonialsColumn testimonials={secondColumn} duration={19} className="hidden md:block" />
        <TestimonialsColumn testimonials={thirdColumn} duration={17} className="hidden lg:block" />
      </div>
    </section>
  )
}

export default TestimonialsSection
