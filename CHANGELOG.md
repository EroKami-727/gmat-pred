# OrbitGuard — Changelog

---

## Session 1 (2026-03-29) — Landing Page + Dashboard Overhaul

### Landing Page (`frontend/src/LandingPage.jsx`)

**Gradient text fix (grey rectangle blurs)**
- Problem: `WebkitBackgroundClip: text` + `WebkitTextFillColor: transparent` as React inline styles renders as solid grey blocks
- Fix: Added `.clip-grad` CSS class inside the `FontLink` `<style>` tag. Applied to: nav ORBITGUARD, features heading, workflow heading, CTA heading, footer ORBITGUARD

**Feature card icon fix (teal highlights)**
- Problem: Icon backgrounds were computing from old teal palette values
- Fix: Changed icon box to `background: rgba(255,255,255,0.04)` with `border: 1px solid ${f.color}33`

**Interactive WorkflowSection**
- Replaced static workflow HTML with a fully stateful `WorkflowSection` React component
- Features: active step state, hover state, animated progress bar (`width` CSS transition), per-step detail panel with `@keyframes wfSlide` slide-up animation, tag chips, Prev/Next navigation buttons
- Steps: Run GMAT Sim → Extract Features → LSTM Inference → Go/No-Go

**Logo carousel rewrite (`frontend/src/components/ui/logo-carousel.jsx`)**
- Replaced broken text-only carousel with proper `LogoCloud` SVG image carousel
- Uses confirmed svgl.app light wordmark URLs (NVIDIA, OpenAI, GitHub, Vercel, Claude, TensorFlow, Python, Jupyter, PyTorch, AWS)
- `filter: brightness(0) invert(1)` for white-on-black
- `onError` fallback: `e.currentTarget.style.display = 'none'` to silently hide broken images
- CSS `maskImage` for left/right edge fade
- Exports both `LogoCloud` (new) and `LogoCarousel` (backward compat alias)

---

### Dashboard Theme (`frontend/src/index.css`)

**Full B&W retheme — CSS variables**
```
--bg: #090909        --bg1: #0f0f0f       --bg2: #141414       --bg3: #060606
--cyan: #e2e2e2      --cyan-glow: rgba(226,226,226,0.15)        --cyan-dim: #2a2a2a
--green: #aaaaaa     --green-glow: rgba(170,170,170,0.12)
--red: #ff5555       --orange: #ffaa00
--text: #aaaaaa      --text-dim: #666666  --text-bright: #eeeeee
--border: #1c1c1c    --border2: #282828
```

**Fixed hardcoded teal in CSS classes**
- `.face`: background changed from `rgba(0,212,255,0.02)` → `rgba(226,226,226,0.02)`
- `.abl-table tr.hl td`: `rgba(0,212,255,0.06)` → `rgba(226,226,226,0.06)`

**Font size overhaul (all bumped for readability)**
- Base: `13px` → `14px`, line-height `1.5` → `1.6`
- `.card-hdr`: `9px` → `11px`
- `.badge` / `.badge-live`: `8px` → `10px`
- `.tile-label`: `8px` → `10px`; `.tile-value em`: `13px` → `16px`; `.tile-sub`: `10px` → `12px`
- `.sec`: `8px` → `11px`
- `.terminal`: `11px` → `13px`
- `.stat-row`: `10px` → `12px`
- `.chart-label` / `.legend-item`: `8px` → `11px`
- `.abl-table th`: `8px` → `11px`; `td`: `12px` → `13px`; `.bar-val`: `10px` → `12px`
- `.feat-name`: `11px` → `13px`; `.feat-desc`: `8px` → `11px`
- `.planet-name`: `12px` → `13px`; `.planet-stat`: `10px` → `12px`
- `.ft-name`: `9px` → `12px`; `.ft-cls`: `8px` → `11px`
- `.arch-name`: `11px` → `13px`
- `.finding-title`: `13px` → `15px`; `.finding-tag`: `8px` → `10px`; `.finding-body`: `11px` → `13px`; `.finding-action`: `9px` → `11px`
- `.field-label`: `8px` → `11px`; `.launch-btn`: `11px` → `13px`
- `.gauge-text`: `13px` → `14px`; `small`: `8px` → `10px`
- `.face` font-size: `9px` → `11px`

---

### Dashboard App Shell (`frontend/src/App.jsx`)

**B&W color fixes**
- Text-shadow colors: `rgba(0,212,255,...)` → `rgba(226,226,226,...)`
- Header accent bar gradient: teal → `linear-gradient(180deg, #ffffff, #555555)`

**Back navigation button**
- Added `onBack` prop to `Header` component
- Added `← Landing` button in header (monospace font, border hover effect, wired to `setShowDashboard(false)`)

**Font size bumps in App.jsx**
- Header subtitle: `8px` → `11px`; version: `10px` → `12px`; status labels: `9px` → `11px`; status values: `10px` → `12px`
- Tab bar buttons: `10px` → `12px`, padding `13px 22px` → `14px 24px`

---

### Flow Field Background (`frontend/src/components/ui/flow-field-background.jsx`)

- Line 73: Trail color `rgba(10,26,32,...)` (dark teal `#0a1a20`) → `rgba(9,9,9,...)` (pure black)
- This was causing the entire hero section to appear teal-tinted

---

### Ablation Panel (`frontend/src/panels/Ablation.jsx`)

- Bar colors: `#00d4ff`/`#00ffa3` → `#e2e2e2`/`#cccccc`/`#aaaaaa`/`#888888`
- Chart gradient stop: `#00ffa3` → `#e2e2e2`
- Area stroke: `#00ffa3` → `#aaaaaa`; reference line: `#00d4ff44` → `#e2e2e233`
- Grid stroke: `#111e2c` → `#1a1a1a`; axis/tick: `#3a5a7a`/`#334455` → `#333333`/`#555555`
- Tooltip: bg `#0c1118` → `#0f0f0f`; border `#1a2e40` → `#2a2a2a`
- Inline font sizes: `8px`/`9px` → `11px`

---

### Training Panel (`frontend/src/panels/Training.jsx`)

- Grid stroke: `#111e2c` → `#1a1a1a`; axis: `#3a5a7a`/`#334455` → `#333333`/`#555555`
- Tooltip bg/border → B&W values
- Train area stroke: `#00ffa3` → `#aaaaaa`; Val stroke: `#00d4ff` → `#666666`
- AUC gradient/stroke: `#00ffa3` → `#e2e2e2`
- Reference line: `#223344` → `#333333`
- Legend dots: → `#aaaaaa`/`#666666`
- Inline font sizes: `8px`/`9px` → `11px`

---

### Overview Panel (`frontend/src/panels/Overview.jsx`)

- Gauge labels: `8px` → `11px`
- Checksum line: `8px` → `11px`
- Terminal header: `9px` → `11px`

---

### Dataset Panel (`frontend/src/panels/Dataset.jsx`)

- Planet status lines: `8px` → `11px`
- Arch tag labels: `8px` → `11px`
- Cross-planet annotation: `7px` → `10px`

---

### Smooth Scroll (`frontend/src/smoothScroll.js` + `frontend/src/main.jsx`)

- Installed `lenis` npm package
- Created `smoothScroll.js` with Lenis init (duration 1.4, exponential easing, smoothWheel: true)
- Imported in `main.jsx` before React render
