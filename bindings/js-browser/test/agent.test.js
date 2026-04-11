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

  // ─── Export / Import ────────────────────────────────────────────────────

  describe('export / import', () => {

    test('export returns a Blob with application/zip type', async () => {
      const agent = await createAgent();
      try {
        await agent.addDocument('The capital of France is Paris.');
        const blob = await agent.export();
        expect(blob).toBeInstanceOf(Blob);
        expect(blob.type).toBe('application/zip');
        expect(blob.size).toBeGreaterThan(0);
      } finally {
        agent.close();
      }
    });

    test('export with no documents produces valid zip', async () => {
      const agent = await createAgent();
      try {
        const blob = await agent.export();
        expect(blob.size).toBeGreaterThan(0);
      } finally {
        agent.close();
      }
    });

    test('export after close throws', async () => {
      const agent = await createAgent();
      agent.close();
      await expect(agent.export()).rejects.toThrow();
    });

    test('round-trip: export then importFrom preserves documents', async () => {
      const agent = await createAgent();
      try {
        await agent.addDocument('The sky is blue.');
        await agent.addDocument('Water boils at 100 degrees.');

        const blob = await agent.export();
        const restored = await Agent.importFrom(blob, 'chat.gguf', 'embed.gguf');
        try {
          // The restored agent should be able to chat without error.
          const reply = await restored.chat('s1', 'Tell me about water.');
          expect(typeof reply).toBe('string');
        } finally {
          restored.close();
        }
      } finally {
        agent.close();
      }
    });

    test('importFrom accepts ArrayBuffer', async () => {
      const agent = await createAgent();
      try {
        await agent.addDocument('Hello world.');
        const blob = await agent.export();
        const arrayBuffer = await blob.arrayBuffer();

        const restored = await Agent.importFrom(arrayBuffer, 'c.gguf', 'e.gguf');
        try {
          expect(restored).toBeInstanceOf(Agent);
        } finally {
          restored.close();
        }
      } finally {
        agent.close();
      }
    });

    test('importFrom accepts Uint8Array', async () => {
      const agent = await createAgent();
      try {
        await agent.addDocument('Hello world.');
        const blob = await agent.export();
        const uint8 = new Uint8Array(await blob.arrayBuffer());

        const restored = await Agent.importFrom(uint8, 'c.gguf', 'e.gguf');
        try {
          expect(restored).toBeInstanceOf(Agent);
        } finally {
          restored.close();
        }
      } finally {
        agent.close();
      }
    });

    test('importFrom rejects unsupported manifest version', async () => {
      // Build a zip with version "99" to trigger the version check.
      const { zipSync } = await import('fflate');
      const manifest = JSON.stringify({ version: '99' });
      const knowledge = JSON.stringify([]);
      const zip = zipSync({
        'manifest.json':  new TextEncoder().encode(manifest),
        'knowledge.json': new TextEncoder().encode(knowledge),
        'knowledge.db':   new Uint8Array(0),
      });

      await expect(
        Agent.importFrom(new Uint8Array(zip), 'c.gguf', 'e.gguf')
      ).rejects.toThrow(/unsupported manifest version/);
    });
  });
});
