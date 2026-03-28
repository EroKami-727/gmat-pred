import * as React from 'react'
import { HelpCircle, MessageCircle, ChevronDown } from 'lucide-react'
import * as AccordionPrimitive from '@radix-ui/react-accordion'

// ── Minimal accordion primitives ────────────────────────────────────────────
const CustomAccordion = AccordionPrimitive.Root

const CustomAccordionItem = React.forwardRef(({ style, ...props }, ref) => (
  <AccordionPrimitive.Item ref={ref} style={style} {...props} />
))
CustomAccordionItem.displayName = 'CustomAccordionItem'

const CustomAccordionTrigger = React.forwardRef(({ children, style, ...props }, ref) => (
  <AccordionPrimitive.Header style={{ margin: 0 }}>
    <AccordionPrimitive.Trigger
      ref={ref}
      style={{
        width: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: 16,
        padding: '20px 24px',
        borderRadius: 16,
        background: 'rgba(255,255,255,0.03)',
        border: '1px solid rgba(226,226,226,0.12)',
        cursor: 'pointer',
        textAlign: 'left',
        transition: 'background 0.2s, border-color 0.2s, box-shadow 0.2s',
        ...style,
      }}
      {...props}
    >
      {children}
    </AccordionPrimitive.Trigger>
  </AccordionPrimitive.Header>
))
CustomAccordionTrigger.displayName = 'CustomAccordionTrigger'

const CustomAccordionContent = React.forwardRef(({ children, style, ...props }, ref) => (
  <AccordionPrimitive.Content
    ref={ref}
    style={{ overflow: 'hidden', ...style }}
    {...props}
  >
    {children}
  </AccordionPrimitive.Content>
))
CustomAccordionContent.displayName = 'CustomAccordionContent'

// ── OrbitGuard FAQ data ──────────────────────────────────────────────────────
const faqs = [
  {
    question: 'How early in a trajectory can OrbitGuard predict mission failure?',
    answer:
      'OrbitGuard can classify mission success or failure using as little as 20% of the total trajectory telemetry. The model was trained with ablation sweeps from 10% to 100% trajectory completeness, allowing operators to trigger an early exit decision before committing the full compute budget to simulation.',
  },
  {
    question: 'What planetary systems does the model generalise to?',
    answer:
      'The model uses 13 physics-invariant features computed in the synodic reference frame — including specific orbital energy, eccentricity, and flight path angle — making it inherently transferable across Earth–Moon, Mars transfer, and Jupiter flyby scenarios without retraining.',
  },
  {
    question: 'What simulation data was the model trained on?',
    answer:
      'Training data was generated using NASA GMAT with a full 3-body RK4 integrator across 10,000+ Monte Carlo missions. The resulting dataset contains 85.3M trajectory data points stored in compressed Parquet format, covering diverse initial conditions and planetary targets.',
  },
  {
    question: 'What model architecture does OrbitGuard use?',
    answer:
      'OrbitGuard uses a bidirectional LSTM as its primary architecture, which captures both forward and backward temporal dependencies in trajectory sequences. We also benchmarked a Transformer-based model. The BiLSTM achieves AUC 0.87+ on binary classification at 60% trajectory completeness.',
  },
  {
    question: 'How does OrbitGuard integrate with NASA GMAT?',
    answer:
      'OrbitGuard reads GMAT output files directly. You export trajectory telemetry from GMAT as a report file, upload the CSV to the Mission Control dashboard, select the exit fraction, and receive an instant P(success) confidence score — no additional setup or SDK required.',
  },
  {
    question: 'Can I run OrbitGuard on my own trajectory data?',
    answer:
      'Yes. The Mission Control dashboard accepts any CSV with the required 13 physics columns. A feature extraction script is provided to compute synodic-frame coordinates, orbital energy, and eccentricity from raw Cartesian state vectors if your data is not already in the expected format.',
  },
]

// ── Main FAQ Section ─────────────────────────────────────────────────────────
export function FaqSection() {
  const [openItem, setOpenItem] = React.useState('item-0')

  return (
    <section
      id="faq"
      style={{
        padding: '100px 5vw',
        background: 'linear-gradient(180deg, transparent, rgba(226,226,226,0.025) 50%, transparent)',
        borderTop: '1px solid rgba(226,226,226,0.07)',
      }}
    >
      <style>{`
        .faq-trigger:hover {
          background: rgba(226,226,226,0.06) !important;
          border-color: rgba(226,226,226,0.3) !important;
          box-shadow: 0 4px 24px rgba(226,226,226,0.08) !important;
        }
        .faq-trigger[data-state="open"] {
          background: rgba(226,226,226,0.07) !important;
          border-color: rgba(226,226,226,0.35) !important;
          box-shadow: 0 0 32px rgba(226,226,226,0.1) !important;
          border-bottom-left-radius: 0 !important;
          border-bottom-right-radius: 0 !important;
        }
        .faq-chevron { transition: transform 0.3s ease; }
        .faq-trigger[data-state="open"] .faq-chevron { transform: rotate(180deg); }
        @keyframes accordion-down {
          from { height: 0; opacity: 0; }
          to { height: var(--radix-accordion-content-height); opacity: 1; }
        }
        @keyframes accordion-up {
          from { height: var(--radix-accordion-content-height); opacity: 1; }
          to { height: 0; opacity: 0; }
        }
        .faq-content[data-state="open"]  { animation: accordion-down 0.28s ease forwards; }
        .faq-content[data-state="closed"] { animation: accordion-up 0.22s ease forwards; }
      `}</style>

      <div style={{ maxWidth: 860, margin: '0 auto' }}>
        {/* Heading */}
        <div style={{ textAlign: 'center', marginBottom: 64 }}>
          <div style={{
            display: 'inline-block',
            border: '1px solid rgba(226,226,226,0.18)',
            padding: '4px 18px',
            borderRadius: 9999,
            fontSize: 11,
            fontWeight: 600,
            letterSpacing: '0.16em',
            textTransform: 'uppercase',
            color: '#e2e2e2',
            background: 'rgba(226,226,226,0.06)',
            marginBottom: 20,
            fontFamily: "'Space Grotesk', sans-serif",
          }}>
            FAQ
          </div>
          <h2 style={{
            fontFamily: "'Fraunces', serif",
            fontSize: 'clamp(32px,4vw,52px)',
            fontWeight: 800,
            letterSpacing: '-0.02em',
            lineHeight: 1.1,
            color: '#ffffff',
            margin: '0 0 16px',
          }}>
            Mission briefing
          </h2>
          <p style={{ fontSize: 13, color: 'rgba(255,255,255,0.35)', lineHeight: 1.85, margin: 0, fontFamily: "'Space Mono',monospace" }}>
            Everything you need to know before your first prediction run.
          </p>
        </div>

        {/* Accordion */}
        <CustomAccordion
          type="single"
          collapsible
          value={openItem}
          onValueChange={setOpenItem}
          style={{ display: 'flex', flexDirection: 'column', gap: 10 }}
        >
          {faqs.map((faq, index) => (
            <CustomAccordionItem
              key={index}
              value={`item-${index}`}
              style={{ borderRadius: 16, overflow: 'hidden' }}
            >
              <CustomAccordionTrigger className="faq-trigger">
                <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
                  <div style={{
                    width: 32, height: 32, borderRadius: '50%',
                    background: 'rgba(226,226,226,0.1)',
                    border: '1px solid rgba(226,226,226,0.2)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    flexShrink: 0,
                  }}>
                    <HelpCircle size={15} color="#e2e2e2" />
                  </div>
                  <span style={{
                    fontFamily: "'DM Sans', sans-serif",
                    fontSize: 15,
                    fontWeight: 600,
                    color: 'rgba(255,255,255,0.88)',
                    letterSpacing: '-0.01em',
                    lineHeight: 1.4,
                  }}>
                    {faq.question}
                  </span>
                </div>
                <div className="faq-chevron" style={{
                  width: 28, height: 28, borderRadius: '50%',
                  background: 'rgba(226,226,226,0.08)',
                  border: '1px solid rgba(226,226,226,0.18)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  flexShrink: 0,
                }}>
                  <ChevronDown size={15} color="#e2e2e2" />
                </div>
              </CustomAccordionTrigger>

              <CustomAccordionContent className="faq-content">
                <div style={{
                  padding: '20px 24px 24px',
                  background: 'rgba(226,226,226,0.04)',
                  borderLeft: '1px solid rgba(226,226,226,0.25)',
                  borderRight: '1px solid rgba(226,226,226,0.25)',
                  borderBottom: '1px solid rgba(226,226,226,0.25)',
                  borderBottomLeftRadius: 16,
                  borderBottomRightRadius: 16,
                }}>
                  <div style={{ display: 'flex', gap: 14, alignItems: 'flex-start', paddingLeft: 46 }}>
                    <p style={{
                      fontFamily: "'Space Mono', monospace",
                      fontSize: 12,
                      color: 'rgba(255,255,255,0.5)',
                      lineHeight: 1.9,
                      margin: 0,
                      flex: 1,
                    }}>
                      {faq.answer}
                    </p>
                    <div style={{
                      width: 30, height: 30, borderRadius: '50%',
                      background: 'rgba(170,170,170,0.08)',
                      border: '1px solid rgba(170,170,170,0.2)',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      flexShrink: 0, marginTop: 2,
                    }}>
                      <MessageCircle size={14} color="#aaaaaa" />
                    </div>
                  </div>
                </div>
              </CustomAccordionContent>
            </CustomAccordionItem>
          ))}
        </CustomAccordion>
      </div>
    </section>
  )
}

export default FaqSection
