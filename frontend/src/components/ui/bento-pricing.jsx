import React from 'react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { CheckIcon, SparklesIcon } from 'lucide-react'

function FilledCheck() {
  return (
    <div className="rounded-full p-0.5" style={{ background: '#00d4ff' }}>
      <CheckIcon className="size-3" strokeWidth={3} style={{ color: '#050509' }} />
    </div>
  )
}

function PricingCard({ titleBadge, priceLabel, priceSuffix = '/month', features, cta = 'Get Started', className }) {
  return (
    <div className={cn(
      'relative overflow-hidden rounded-xl border',
      'border-white/10 bg-white/5 backdrop-blur-sm',
      className,
    )}>
      <div className="flex items-center gap-3 p-4">
        <Badge variant="secondary" className="bg-white/10 text-white/80 border-white/15">{titleBadge}</Badge>
        <div className="ml-auto">
          <Button variant="outline" className="border-white/20 bg-white/5 text-white/80 hover:bg-white/10 hover:text-white text-xs">{cta}</Button>
        </div>
      </div>
      <div className="flex items-end gap-2 px-4 py-2">
        <span className="font-mono text-5xl font-semibold tracking-tight text-white">{priceLabel}</span>
        {priceLabel.toLowerCase() !== 'free' && priceLabel.toLowerCase() !== 'custom' && (
          <span className="text-white/40 text-sm">{priceSuffix}</span>
        )}
      </div>
      <ul className="grid gap-4 p-4 text-sm text-white/60">
        {features.map((f, i) => (
          <li key={i} className="flex items-center gap-3">
            <FilledCheck />
            <span>{f}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

export function BentoPricing({ onEnterDashboard }) {
  return (
    <div className="grid grid-cols-1 gap-2 md:grid-cols-2 lg:grid-cols-8">
      {/* Featured — Deep Space */}
      <div className={cn(
        'relative w-full overflow-hidden rounded-xl border border-cyan-400/30 bg-cyan-400/5 backdrop-blur-sm',
        'lg:col-span-5',
      )}>
        <div className="pointer-events-none absolute top-0 left-1/2 -mt-2 -ml-20 h-full w-full"
          style={{ maskImage: 'linear-gradient(white, transparent)' }}>
          <div className="absolute inset-0 bg-gradient-to-r from-cyan-400/5 to-transparent"
            style={{ maskImage: 'radial-gradient(farthest-side at top, white, transparent)' }}>
            <div aria-hidden className="absolute inset-0 size-full mix-blend-overlay"
              style={{ backgroundImage: 'linear-gradient(to right, rgba(0,212,255,0.08) 1px, transparent 1px)', backgroundSize: '24px' }} />
          </div>
        </div>

        <div className="flex items-center gap-3 p-4">
          <Badge variant="secondary" className="bg-cyan-400/15 text-cyan-300 border-cyan-400/25">DEEP SPACE</Badge>
          <Badge variant="outline" className="hidden lg:flex border-cyan-400/30 text-cyan-300/80 bg-cyan-400/5 text-xs">
            <SparklesIcon className="me-1 size-3" /> Most Popular
          </Badge>
          <div className="ml-auto">
            <Button className="bg-cyan-400 hover:bg-cyan-300 text-slate-900 font-semibold text-xs">Get Started</Button>
          </div>
        </div>

        <div className="flex flex-col p-4 lg:flex-row">
          <div className="pb-4 lg:w-[30%]">
            <span className="font-mono text-5xl font-semibold tracking-tight text-white">$299</span>
            <span className="text-white/40 text-sm">/month</span>
          </div>
          <ul className="grid gap-4 text-sm text-white/60 lg:w-[70%]">
            {[
              'Unlimited mission predictions',
              'LSTM + Transformer models',
              'All planetary systems (Moon/Mars/Jupiter)',
              'Ablation sweep tooling + AUC curves',
            ].map((f, i) => (
              <li key={i} className="flex items-center gap-3">
                <FilledCheck />
                <span className="leading-relaxed">{f}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* Orbit — Free */}
      <PricingCard
        titleBadge="ORBIT"
        priceLabel="Free"
        features={[
          'For academic research',
          '500 predictions / month',
          'Binary LSTM classifier',
          'CSV trajectory upload',
        ]}
        cta="Start Free"
        className="lg:col-span-3"
      />

      {/* Pro */}
      <PricingCard
        titleBadge="PROFESSIONAL"
        priceLabel="$699"
        features={[
          'Everything in Deep Space',
          'REST API access (10K calls/mo)',
          'Custom early-exit thresholds',
          'Priority support + SLA',
        ]}
        cta="Upgrade"
        className="lg:col-span-4"
      />

      {/* Enterprise */}
      <PricingCard
        titleBadge="MISSION CRITICAL"
        priceLabel="Custom"
        features={[
          'On-premise GMAT integration',
          'Dedicated compute cluster',
          'Custom model fine-tuning',
          '24/7 dedicated engineer',
        ]}
        cta="Contact Sales"
        className="lg:col-span-4"
      />
    </div>
  )
}

export default BentoPricing
