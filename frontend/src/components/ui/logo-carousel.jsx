import { InfiniteSlider } from '@/components/ui/infinite-slider'
import { cn } from '@/lib/utils'

const logos = [
  { src: 'https://svgl.app/library/nvidia-wordmark-light.svg',         alt: 'NVIDIA'    },
  { src: 'https://svgl.app/library/openai_wordmark_light.svg',         alt: 'OpenAI'    },
  { src: 'https://svgl.app/library/github_wordmark_light.svg',         alt: 'GitHub'    },
  { src: 'https://svgl.app/library/vercel_wordmark.svg',               alt: 'Vercel'    },
  { src: 'https://svgl.app/library/claude-ai-wordmark-icon_light.svg', alt: 'Claude'    },
  { src: 'https://svgl.app/library/tensorflow.svg',                    alt: 'TensorFlow'},
  { src: 'https://svgl.app/library/python.svg',                        alt: 'Python'    },
  { src: 'https://svgl.app/library/jupyter.svg',                       alt: 'Jupyter'   },
  { src: 'https://svgl.app/library/pytorch.svg',                       alt: 'PyTorch'   },
  { src: 'https://svgl.app/library/aws_wordmark_light.svg',            alt: 'AWS'       },
]

export function LogoCloud({ className, logos: customLogos }) {
  const items = customLogos ?? logos
  return (
    <div
      className={cn(
        'overflow-hidden py-4',
        className
      )}
      style={{
        maxWidth: 1100,
        margin: '0 auto',
        maskImage: 'linear-gradient(to right, transparent, black 14%, black 86%, transparent)',
        WebkitMaskImage: 'linear-gradient(to right, transparent, black 14%, black 86%, transparent)',
      }}
    >
      <InfiniteSlider gap={72} duration={45} durationOnHover={90} reverse>
        {items.map(logo => (
          <img
            key={logo.alt}
            src={logo.src}
            alt={logo.alt}
            loading="lazy"
            style={{
              height: 24,
              width: 'auto',
              pointerEvents: 'none',
              userSelect: 'none',
              filter: 'brightness(0) invert(1)',
              opacity: 0.45,
              transition: 'opacity 0.3s',
            }}
            onMouseEnter={e => { e.currentTarget.style.opacity = '0.9' }}
            onMouseLeave={e => { e.currentTarget.style.opacity = '0.35' }}
            onError={e => { e.currentTarget.style.display = 'none' }}
          />
        ))}
      </InfiniteSlider>
    </div>
  )
}

// Keep original name for backwards compatibility
export function LogoCarousel() {
  return <LogoCloud />
}

export default LogoCarousel
