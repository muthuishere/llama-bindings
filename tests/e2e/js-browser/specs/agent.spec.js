// tests/e2e/js-browser/specs/agent.spec.js
//
// Playwright end-to-end tests for Agent (browser JS binding).
//
// Tests open Chromium, navigate to the test app which runs Agent scenarios
// and exposes results on window.__llama_results.
//
// Without the WASM build the stub module provides deterministic responses.

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
// Agent — basic chat
// ---------------------------------------------------------------------------

test('Agent chat returns a non-empty string', async ({ page }) => {
  await openApp(page);

  const reply = await page.evaluate(() => window.__llama_results.agentChat);

  expect(typeof reply).toBe('string');
  expect(reply.trim().length).toBeGreaterThan(0);
});

// ---------------------------------------------------------------------------
// Agent — knowledge (addDocument → query)
// ---------------------------------------------------------------------------

test('Agent with document returns non-empty reply', async ({ page }) => {
  await openApp(page);

  const reply = await page.evaluate(() => window.__llama_results.agentWithDoc);

  expect(typeof reply).toBe('string');
  expect(reply.trim().length).toBeGreaterThan(0);
});

// ---------------------------------------------------------------------------
// Agent — tool dispatch
// ---------------------------------------------------------------------------

test('Agent with tool runs without crashing', async ({ page }) => {
  await openApp(page);

  const reply = await page.evaluate(() => window.__llama_results.agentWithTool);

  // Bridge stub always returns tool_call → agent hits iteration limit.
  // Accept either a real reply or the known stub-loop marker.
  expect(typeof reply).toBe('string');
  expect(reply.length).toBeGreaterThan(0);
});

// ---------------------------------------------------------------------------
// Agent — export → import round-trip
// ---------------------------------------------------------------------------

test('Agent export → import returns non-empty reply', async ({ page }) => {
  // Export/ImportFrom not yet implemented in JS binding — tracked in docs/export-import.md
  test.skip();
});

// ---------------------------------------------------------------------------
// Agent — lifecycle: close is idempotent
// ---------------------------------------------------------------------------

test('Agent close is idempotent in browser', async ({ page }) => {
  await page.goto('/tests/e2e/js-browser/app/index.html');
  await page.waitForFunction(() => window.__llama_ready === true, { timeout: 60_000 });

  const threw = await page.evaluate(async () => {
    const { Agent } = await import('/bindings/js-browser/src/agent/agent.js');
    const agent = await Agent.create('dummy.gguf', 'dummy-embed.gguf', ':memory:');
    agent.close();
    try { agent.close(); return false; } catch { return true; }
  });

  expect(threw).toBe(false);
});

// ---------------------------------------------------------------------------
// Agent — chat after close throws
// ---------------------------------------------------------------------------

test('Agent chat after close throws', async ({ page }) => {
  await page.goto('/tests/e2e/js-browser/app/index.html');
  await page.waitForFunction(() => window.__llama_ready === true, { timeout: 60_000 });

  const threw = await page.evaluate(async () => {
    const { Agent } = await import('/bindings/js-browser/src/agent/agent.js');
    const agent = await Agent.create('dummy.gguf', 'dummy-embed.gguf', ':memory:');
    agent.close();
    try {
      await agent.chat('s', 'hi');
      return false;
    } catch {
      return true;
    }
  });

  expect(threw).toBe(true);
});
