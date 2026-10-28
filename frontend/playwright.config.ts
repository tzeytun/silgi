import { defineConfig } from '@playwright/test'

// The E2E suite drives the real app: the Python server serving the built frontend.
export default defineConfig({
  testDir: './e2e',
  timeout: 90_000,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? 'github' : 'list',
  use: {
    baseURL: 'http://127.0.0.1:8799',
    viewport: { width: 1440, height: 900 },
    locale: 'en-US',
    trace: 'retain-on-failure',
  },
  webServer: {
    command: 'python -m silgi --no-browser --port 8799',
    url: 'http://127.0.0.1:8799/api/health',
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
})
