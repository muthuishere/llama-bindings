/**
 * embed.test.js – unit tests for LlamaEmbed (browser binding).
 *
 * Tests run against the stub WASM module so no model file is required.
 */

import { LlamaEmbed } from '../src/embed.js';
import { LlamaError }  from '../src/errors.js';

describe('LlamaEmbed', () => {

  test('load returns a LlamaEmbed instance', async () => {
    const embed = await LlamaEmbed.load('dummy-embed.gguf');
    expect(embed).toBeInstanceOf(LlamaEmbed);
    embed.close();
  });

  test('load with empty path throws MODEL_LOAD_FAILED', async () => {
    await expect(LlamaEmbed.load('')).rejects.toThrow(LlamaError);
  });

  test('embed returns a Float32Array with length > 0', async () => {
    const embed = await LlamaEmbed.load('dummy-embed.gguf');
    try {
      const vec = await embed.embed('semantic search example');
      expect(vec).toBeInstanceOf(Float32Array);
      expect(vec.length).toBeGreaterThan(0);
    } finally {
      embed.close();
    }
  });

  test('embed empty string throws INVALID_REQUEST', async () => {
    const embed = await LlamaEmbed.load('dummy-embed.gguf');
    try {
      await expect(embed.embed('')).rejects.toMatchObject({ code: 'INVALID_REQUEST' });
    } finally {
      embed.close();
    }
  });

  test('repeated embed calls are stable', async () => {
    const embed = await LlamaEmbed.load('dummy-embed.gguf');
    try {
      for (let i = 0; i < 3; i++) {
        const vec = await embed.embed('repeated input');
        expect(vec.length).toBeGreaterThan(0);
      }
    } finally {
      embed.close();
    }
  });

  test('embed after close throws ENGINE_CLOSED', async () => {
    const embed = await LlamaEmbed.load('dummy-embed.gguf');
    embed.close();
    await expect(embed.embed('hello')).rejects.toMatchObject({ code: 'ENGINE_CLOSED' });
  });

  test('close is idempotent', async () => {
    const embed = await LlamaEmbed.load('dummy-embed.gguf');
    embed.close();
    expect(() => embed.close()).not.toThrow();
  });

  test('onEvent callback wiring does not throw', async () => {
    const events = [];
    const embed = await LlamaEmbed.load('dummy-embed.gguf', {
      onEvent: (e) => events.push(e),
    });
    embed.close();
    expect(Array.isArray(events)).toBe(true);
  });
});
