import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

export default defineConfig({
  plugins: [tailwindcss(), react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  build: {
    // recharts alone is ~529 kB minified and cannot be split further without
    // giving up the chart components the panels use. Everything we can control
    // is already split, so raise the ceiling just above it rather than leaving
    // a permanent warning that no longer points at anything actionable.
    chunkSizeWarningLimit: 600,
    rollupOptions: {
      output: {
        // The dashboard shipped as one 912 kB chunk, so a visitor landing on the
        // marketing page downloaded Recharts, framer-motion and all seven panels
        // before seeing anything. Split along the lines the app actually loads:
        // the landing page renders first and the dashboard is behind a click,
        // and the charting stack is only fetched once a panel that draws mounts.
        //
        // react/react-dom are deliberately not listed — pinning them to their own
        // chunk produced a 0.06 kB file, because rollup had already hoisted them
        // into the shared entry that every lazy panel depends on.
        manualChunks: {
          charts: ['recharts'],
          motion: ['framer-motion', 'lenis'],
        },
      },
    },
  },
})
