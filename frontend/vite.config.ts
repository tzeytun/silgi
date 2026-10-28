import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'

// The production build is served by the Python package from src/silgi/web.
export default defineConfig({
  plugins: [react(), tailwindcss()],
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
