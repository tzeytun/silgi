import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { writeFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { defineConfig } from 'vitest/config'

// The production build is served by the Python package from src/silgi/web.
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    {
      // the output folder must exist in a fresh clone (hatch force-includes it), so
      // keep the tracked placeholder that emptyOutDir would delete
      name: 'keep-web-placeholder',
      closeBundle() {
        writeFileSync(resolve(__dirname, '../src/silgi/web/.gitkeep'), '')
      },
    },
  ],
  build: { outDir: '../src/silgi/web', emptyOutDir: true, chunkSizeWarningLimit: 900 },
  server: {
    proxy: { '/api': { target: 'http://127.0.0.1:8765', ws: true } },
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test-setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
  },
})
