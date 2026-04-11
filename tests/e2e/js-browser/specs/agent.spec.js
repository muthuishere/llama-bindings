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
// Helper: open the Agent Team example page and wait for "Agent ready"
// ---------------------------------------------------------------------------
const EXAMPLE_URL = '/examples/agent-team/js-browser/';

async function openExample(page) {
  await page.goto(EXAMPLE_URL);
  await page.waitForSelector('#status.ready', { timeout: 15_000 });
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

// ===========================================================================
// Agent Team Example — interactive UI tests
// ===========================================================================

// ---------------------------------------------------------------------------
// page_load — example page loads and shows "Agent ready"
// ---------------------------------------------------------------------------

test('page_load: example page loads and shows Agent ready', async ({ page }) => {
  await openExample(page);

  const statusText = await page.locator('#status').textContent();
  expect(statusText).toContain('Agent ready');
});

// ---------------------------------------------------------------------------
// chat — click suggestion chip and get a response
// ---------------------------------------------------------------------------

test('chat: clicking suggestion chip produces agent response', async ({ page }) => {
  await openExample(page);

  // Click the "What is llama-bindings?" suggestion chip
  await page.locator('.suggestion', { hasText: 'What is llama-bindings?' }).click();

  // Wait for an agent response bubble to appear
  const agentMsg = page.locator('.msg.agent').first();
  await agentMsg.waitFor({ state: 'visible', timeout: 10_000 });

  const text = await agentMsg.textContent();
  expect(text.trim().length).toBeGreaterThan(0);
});

// ---------------------------------------------------------------------------
// chat_tool — click tool-related suggestion chip
// ---------------------------------------------------------------------------

test('chat_tool: clicking calculator chip produces response', async ({ page }) => {
  await openExample(page);

  // Click the "What is the square root of 144?" suggestion chip
  await page.locator('.suggestion', { hasText: 'What is the square root of 144?' }).click();

  // Wait for an agent response bubble to appear
  const agentMsg = page.locator('.msg.agent').first();
  await agentMsg.waitFor({ state: 'visible', timeout: 10_000 });

  const text = await agentMsg.textContent();
  expect(text.trim().length).toBeGreaterThan(0);
});

// ---------------------------------------------------------------------------
// multi_turn — two sequential messages produce two agent responses
// ---------------------------------------------------------------------------

test('multi_turn: two messages produce at least two agent responses', async ({ page }) => {
  await openExample(page);

  // First message
  await page.fill('#input', 'Hello');
  await page.click('#send');

  // Wait for first agent response
  const firstMsg = page.locator('.msg.agent').first();
  await firstMsg.waitFor({ state: 'visible', timeout: 10_000 });

  // Wait for input to be re-enabled (agent finished responding)
  await page.waitForSelector('#input:not([disabled])', { timeout: 10_000 });

  // Second message
  await page.fill('#input', 'How do I build everything?');
  await page.click('#send');

  // Wait for at least two agent responses
  await page.waitForFunction(
    () => document.querySelectorAll('.msg.agent').length >= 2,
    { timeout: 10_000 },
  );

  const agentMessages = await page.locator('.msg.agent').count();
  expect(agentMessages).toBeGreaterThanOrEqual(2);
});
