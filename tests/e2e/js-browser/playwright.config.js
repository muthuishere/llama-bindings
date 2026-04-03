// tests/e2e/js-browser/playwright.config.js
//
// Playwright configuration for browser e2e tests of the llama-bindings
// Browser JS / WASM binding.
//
// The tests open a real Chromium browser, navigate to a local web app that
// imports the JS binding, and verify results via page.evaluate().
//
// Prerequisites:
//   npx playwright install chromium
//   task e2e-js   (or: cd tests/e2e/js-browser && npm test)

import { defineConfig, devices } from '@playwright/test';
import { fileURLToPath } from 'url';
import path from 'path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
// Serve from the project root so /bindings/js-browser/src/ paths resolve.
const ROOT_DIR = path.resolve(__dirname, '..', '..', '..');

export default defineConfig({
  testDir: './specs',
  timeout: 60_000,
  retries: 0,
  reporter: [['list'], ['html', { open: 'never', outputFolder: 'playwright-report' }]],

  use: {
    baseURL: 'http://localhost:4321',
    headless: true,
    screenshot: 'only-on-failure',
    video: 'off',
  },

  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],

  // Spin up a static server that serves from the project root.
  // The test app at /tests/e2e/js-browser/app/index.html imports from
  // /bindings/js-browser/src/chat.js etc. via absolute paths.
  webServer: {
    command: `npx serve -l 4321 --cors ${ROOT_DIR}`,
    url: 'http://localhost:4321',
    reuseExistingServer: !process.env.CI,
    timeout: 30_000,
  },
});
