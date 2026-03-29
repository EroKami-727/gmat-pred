# OrbitGuard — AI Context File
> Read this first. Everything you need to understand the project and codebase without wasting tokens.

---

## What is OrbitGuard?

An AI-powered spacecraft mission failure prediction system. The project has:
1. **Landing Page** — marketing/hero page (`LandingPage.jsx`)
2. **Mission Control Dashboard** — data visualization app (`App.jsx` + panels)
3. **ML Backend** — LSTM model trained on GMAT orbital simulation data

The user is building this as a portfolio/demo project. Aesthetic and professional presentation matter a lot.

---

## Tech Stack

| Layer | Tech |
|---|---|
| Frontend | React + Vite (JavaScript, NO TypeScript) |
| Styling | Tailwind CSS v4 via `@tailwindcss/vite` — **UNRELIABLE**, prefer inline styles |
| Charts | Recharts (`AreaChart`, `ReferenceLine`, etc.) |
| Accordion | `@radix-ui/react-accordion` |
| Smooth Scroll | Lenis |
| Icons | `lucide-react` |
| Fonts | Fraunces (serif) + Space Mono (mono) + DM Sans + Space Grotesk |
| ML | LSTM model, Python, TensorFlow/PyTorch, GMAT simulation data |

---

## Critical Gotchas

### 1. Tailwind v4 is unreliable
Many utility classes don't work. **Always use inline styles** for layout/spacing/color-critical things. Only use Tailwind for rough structure if needed.

### 2. Gradient text MUST be a CSS class
`WebkitBackgroundClip: text` + `WebkitTextFillColor: transparent` as React inline styles = renders as a solid grey rectangle.
**Fix:** Use a CSS class in a `<style>` tag:
```css
.clip-grad {
  background: linear-gradient(135deg, #e2e2e2 0%, #888888 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
```

### 3. Dashboard uses CSS custom properties
All dashboard colors come from CSS variables in `index.css`. Do NOT hardcode teal/cyan hex values anywhere — they've all been replaced with B&W.

### 4. Landing page uses `body.landing` class
Landing page styles are isolated under `body.landing` to avoid bleeding into dashboard styles.

---

## Color Palette

### Dashboard (B&W theme)
```
--bg: #090909          (main background)
--bg1: #0f0f0f         (card background)
--bg2: #141414         (elevated surfaces)
--bg3: #060606         (deep background)
--cyan: #e2e2e2        (primary accent — was teal, now light grey)
--cyan-glow: rgba(226,226,226,0.15)
--cyan-dim: #2a2a2a
--green: #aaaaaa       (secondary accent)
--green-glow: rgba(170,170,170,0.12)
--red: #ff5555         (alerts)
--orange: #ffaa00      (warnings)
--text: #aaaaaa        (body text)
--text-dim: #666666    (muted text)
--text-bright: #eeeeee (headings)
--border: #1c1c1c      (subtle borders)
--border2: #282828     (visible borders)
```

### Landing Page (inline styles)
- Background: `#050505` / `#0a0a0a`
- Primary text: `#e2e2e2`
- Muted text: `#888888` / `#666666`
- Accent gradient: `linear-gradient(135deg, #e2e2e2 0%, #888888 100%)`

---

## File Map

```
frontend/src/
├── main.jsx                          # React entry point + Lenis smooth scroll init
├── App.jsx                           # Dashboard shell: Header, tab navigation, panel routing
├── LandingPage.jsx                   # Full landing page (hero, features, workflow, logos, FAQ, CTA)
├── index.css                         # ALL dashboard CSS variables + class definitions
├── smoothScroll.js                   # Lenis init (imported by main.jsx)
├── panels/
│   ├── Overview.jsx                  # Mission overview: gauges, terminal, stat tiles
│   ├── Ablation.jsx                  # Ablation study: bar charts + area chart
│   ├── Training.jsx                  # Training curves: loss + AUC area charts
│   └── Dataset.jsx                   # Dataset explorer: planet cards, feature importance
└── components/ui/
    ├── flow-field-background.jsx     # Canvas particle animation for hero (trail color = rgba(9,9,9,...))
    ├── logo-carousel.jsx             # Infinite SVG logo slider (svgl.app URLs, brightness invert)
    ├── faq-accordion.jsx             # Radix accordion for FAQ section
    ├── infinite-slider.jsx           # Generic infinite slider primitive
    ├── globe.jsx                     # 3D globe component
    ├── progressive-blur.jsx          # Edge blur effect
    └── ...other UI primitives
```

---

## Landing Page Sections (top → bottom)

1. **Nav** — ORBITGUARD wordmark (`.clip-grad`), links, CTA button
2. **Hero** — Flow field canvas bg, large headline, subtext, two CTA buttons
3. **LogoCloud** — Infinite scrolling tech logos (NVIDIA, OpenAI, GitHub, Vercel, Claude, TF, Python, Jupyter, PyTorch, AWS)
4. **Features** — 6 feature cards with icons (neutral bg `rgba(255,255,255,0.04)`, colored borders)
5. **Workflow** — Interactive 4-step workflow (`WorkflowSection` component, see below)
6. **Testimonials** — Social proof cards
7. **FAQ** — Radix accordion
8. **CTA** — Final call-to-action section
9. **Footer** — Links + ORBITGUARD wordmark

---

## WorkflowSection Component

Located in `LandingPage.jsx` above the default export. Fully stateful interactive component.

```
States: active (0-3), hovered (null or index)
Steps: [Run GMAT Sim, Extract Features, LSTM Inference, Go/No-Go]
Each step has: step number, Icon, color, title, desc, detail (long), tags[]
UI: step nodes → animated progress bar (width transition) → detail panel (wfSlide animation) → tag chips → Prev/Next buttons
Animation: @keyframes wfSlide defined in a <style> tag in the component
```

---

## Dashboard Navigation

- App starts on LandingPage (`showDashboard = false` state in `main.jsx` or `App.jsx`)
- "Enter Mission Control" button on landing → `setShowDashboard(true)`
- `← Landing` button in dashboard header → `setShowDashboard(false)` via `onBack` prop

---

## Dashboard Panels

| Panel | Tab | Key components |
|---|---|---|
| Overview | OVERVIEW | Gauges, live terminal log, stat tiles, mission status |
| Ablation | ABLATION | Bar chart comparison, area chart, data table |
| Training | TRAINING | Loss curves (train/val), AUC curve, epoch stats |
| Dataset | DATASET | Planet cards (Mercury/Venus/Earth/Mars/Jupiter), feature importance, architecture tags |

---

## Logo Carousel

File: `frontend/src/components/ui/logo-carousel.jsx`

- Uses `LogoCloud` component (ibelick pattern)
- SVG sources from `svgl.app` — use `-wordmark-light` suffix for light variants
- All images: `filter: brightness(0) invert(1)` (white on black)
- `onError`: `e.currentTarget.style.display = 'none'` (silently hide broken)
- CSS `maskImage` for left/right edge fade on scroll container
- Exports: `LogoCloud` (main) + `LogoCarousel` (alias for backward compat)

---

## ML / Data Context

- Simulation: GMAT (General Mission Analysis Tool), RK4 integrator, 3-body physics
- Training data: ~2000 simulations, state vectors, CSV export
- Model: LSTM for time-series trajectory data
- Output: Go / No-Go failure prediction with confidence score
- Planets covered: Mercury, Venus, Earth, Mars, Jupiter

---

## What NOT to change (fragile things)

- Flow field trail color MUST stay `rgba(9,9,9,...)` — any teal value causes hero section tint
- `.clip-grad` class MUST stay in the `<style>` tag — inline styles break gradient text
- CSS variables MUST stay B&W — dashboard was fully rethemed from teal/cyan
- Logo carousel `onError` handler — some svgl.app URLs are fragile, the fallback hides failures silently

---

## Pending / Future Work

- [ ] Actually verify Lenis smooth scroll is working correctly on the landing page
- [ ] Mobile responsiveness audit (not done yet)
- [ ] Connect ML model API to dashboard (currently static/mock data)
- [ ] Add real mission simulation data to panels
