// tests/e2e/js-browser/specs/embed.spec.js
//
// Playwright end-to-end tests for LlamaEmbed (browser JS binding).

import { test, expect } from '@playwright/test';

async function openApp(page) {
  await page.goto('/tests/e2e/js-browser/app/index.html');
  await page.waitForFunction(() => window.__llama_ready === true, { timeout: 60_000 });

  const appError = await page.evaluate(() => window.__llama_error);
  if (appError) {
    throw new Error(`Browser app error: ${appError}`);
  }
}

// ---------------------------------------------------------------------------
// Vector output
// ---------------------------------------------------------------------------

test('LlamaEmbed returns a non-empty float vector', async ({ page }) => {
  await openApp(page);

  const length = await page.evaluate(() => window.__llama_results.embedLength);

  expect(typeof length).toBe('number');
  expect(length).toBeGreaterThan(0);
});

test('LlamaEmbed vector is an array of numbers', async ({ page }) => {
  await openApp(page);

  const vector = await page.evaluate(() => window.__llama_results.embedVector);

  expect(Array.isArray(vector)).toBe(true);
  expect(vector.length).toBeGreaterThan(0);
  for (const v of vector) {
    expect(typeof v).toBe('number');
    expect(isFinite(v)).toBe(true);
  }
});

// ---------------------------------------------------------------------------
// Lifecycle — close after use
// ---------------------------------------------------------------------------

test('LlamaEmbed closes cleanly without errors', async ({ page }) => {
  await openApp(page);
  const appError = await page.evaluate(() => window.__llama_error);
  expect(appError).toBeNull();
});

// ---------------------------------------------------------------------------
// Lifecycle — error on closed engine
// ---------------------------------------------------------------------------

test('LlamaEmbed throws ENGINE_CLOSED after close()', async ({ page }) => {
  await page.goto('/tests/e2e/js-browser/app/index.html');
  await page.waitForFunction(() => window.__llama_ready === true, { timeout: 60_000 });

  const errCode = await page.evaluate(async () => {
    const { LlamaEmbed } = await import('/bindings/js-browser/src/embed.js');
    const embed = await LlamaEmbed.load('dummy-embed.gguf');
    embed.close();
    try {
      await embed.embed('hello');
      return null;
    } catch (e) {
      return e.code;
    }
  });

  expect(errCode).toBe('ENGINE_CLOSED');
});

// ---------------------------------------------------------------------------
// Validation — empty input
// ---------------------------------------------------------------------------

test('LlamaEmbed throws INVALID_REQUEST for empty input', async ({ page }) => {
  await page.goto('/tests/e2e/js-browser/app/index.html');
  await page.waitForFunction(() => window.__llama_ready === true, { timeout: 60_000 });

  const errCode = await page.evaluate(async () => {
    const { LlamaEmbed } = await import('/bindings/js-browser/src/embed.js');
    const embed = await LlamaEmbed.load('dummy-embed.gguf');
    try {
      await embed.embed('');
      return null;
    } catch (e) {
      return e.code;
    } finally {
      embed.close();
    }
  });

  expect(errCode).toBe('INVALID_REQUEST');
});

// ---------------------------------------------------------------------------
// Observability — events field exists
// ---------------------------------------------------------------------------

test('LlamaEmbed emits observability events array', async ({ page }) => {
  await openApp(page);

  const events = await page.evaluate(() => window.__llama_results.embedEvents);

  expect(Array.isArray(events)).toBe(true);
});
