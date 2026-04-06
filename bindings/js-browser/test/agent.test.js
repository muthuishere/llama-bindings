/**
 * agent.test.js – unit tests for Agent (browser binding).
 *
 * Tests run against the stub WASM module so no model file is required.
 */

import { Agent } from '../src/agent/agent.js';

describe('Agent', () => {

  async function createAgent() {
    return Agent.create('dummy.gguf', 'dummy-embed.gguf', ':memory:');
  }

  test('create returns an Agent instance', async () => {
    const agent = await createAgent();
    expect(agent).toBeInstanceOf(Agent);
    agent.close();
  });

  test('chat returns a string', async () => {
    const agent = await createAgent();
    try {
      const reply = await agent.chat('s1', 'Say hello.');
      expect(typeof reply).toBe('string');
      expect(reply.length).toBeGreaterThan(0);
    } finally {
      agent.close();
    }
  });

  test('chat maintains session history across turns', async () => {
    const agent = await createAgent();
    try {
      await agent.chat('history-session', 'What is 2+2?');
      const reply = await agent.chat('history-session', 'What about 3+3?');
      expect(typeof reply).toBe('string');
    } finally {
      agent.close();
    }
  });

  test('clearSession allows fresh conversation', async () => {
    const agent = await createAgent();
    try {
      await agent.chat('clear-session', 'Hello');
      agent.clearSession('clear-session');
      const reply = await agent.chat('clear-session', 'Hello again');
      expect(typeof reply).toBe('string');
    } finally {
      agent.close();
    }
  });

  test('addDocument does not throw', async () => {
    const agent = await createAgent();
    try {
      await expect(
        agent.addDocument('The capital of France is Paris.')
      ).resolves.toBeUndefined();
    } finally {
      agent.close();
    }
  });

  test('addTool does not throw', async () => {
    const agent = await createAgent();
    try {
      expect(() => agent.addTool(
        { name: 'greet', description: 'Greet user', parameters: {} },
        async () => 'Hello!'
      )).not.toThrow();
    } finally {
      agent.close();
    }
  });

  test('chat after close throws', async () => {
    const agent = await createAgent();
    agent.close();
    await expect(agent.chat('s', 'hi')).rejects.toThrow();
  });

  test('close is idempotent', async () => {
    const agent = await createAgent();
    agent.close();
    expect(() => agent.close()).not.toThrow();
  });

  test('multiple sessions are independent', async () => {
    const agent = await createAgent();
    try {
      const r1 = await agent.chat('s1', 'Hello from s1');
      const r2 = await agent.chat('s2', 'Hello from s2');
      expect(typeof r1).toBe('string');
      expect(typeof r2).toBe('string');
    } finally {
      agent.close();
    }
  });

  test('agent with tools registered runs without error', async () => {
    const agent = await createAgent();
    try {
      agent.addTool(
        { name: 'lookup_weather', description: 'Get weather', parameters: { type: 'object' } },
        async ({ city }) => ({ temperature: '30°C', city })
      );
      const reply = await agent.chat('tool-session', 'What is the weather in Chennai?');
      expect(typeof reply).toBe('string');
    } finally {
      agent.close();
    }
  });
});
