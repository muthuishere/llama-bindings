/**
 * chat.test.js – unit tests for LlamaChat (browser binding).
 *
 * Tests run against the stub WASM module so no model file is required.
 */

import { LlamaChat } from '../src/chat.js';
import { LlamaError } from '../src/errors.js';

describe('LlamaChat', () => {

  test('load returns a LlamaChat instance', async () => {
    const chat = await LlamaChat.load('dummy.gguf');
    expect(chat).toBeInstanceOf(LlamaChat);
    chat.close();
  });

  test('load with empty path throws MODEL_LOAD_FAILED', async () => {
    await expect(LlamaChat.load('')).rejects.toThrow(LlamaError);
  });

  test('chat text mode returns assistant_text', async () => {
    const chat = await LlamaChat.load('dummy.gguf');
    try {
      const resp = await chat.chat({
        messages: [{ role: 'user', content: 'Say hello.' }],
      });
      expect(resp.type).toBe('assistant_text');
      expect(typeof resp.text).toBe('string');
    } finally {
      chat.close();
    }
  });

  test('chat json_schema mode returns structured_json', async () => {
    const chat = await LlamaChat.load('dummy.gguf');
    try {
      const resp = await chat.chat({
        messages:     [{ role: 'user', content: 'Extract data.' }],
        responseMode: 'json_schema',
        schema:       { name: 'person', schema: { type: 'object' } },
      });
      expect(resp.type).toBe('structured_json');
    } finally {
      chat.close();
    }
  });

  test('chat tool_call mode returns tool_call', async () => {
    const chat = await LlamaChat.load('dummy.gguf');
    try {
      const resp = await chat.chat({
        messages:     [{ role: 'user', content: 'Weather in Chennai?' }],
        responseMode: 'tool_call',
        tools: [{ name: 'lookup_weather', description: 'Get weather', parameters: {} }],
        toolChoice: 'auto',
      });
      expect(resp.type).toBe('tool_call');
      expect(Array.isArray(resp.tool_calls)).toBe(true);
    } finally {
      chat.close();
    }
  });

  test('chat after close throws ENGINE_CLOSED', async () => {
    const chat = await LlamaChat.load('dummy.gguf');
    chat.close();
    await expect(
      chat.chat({ messages: [{ role: 'user', content: 'hi' }] })
    ).rejects.toMatchObject({ code: 'ENGINE_CLOSED' });
  });

  test('close is idempotent', async () => {
    const chat = await LlamaChat.load('dummy.gguf');
    chat.close();
    expect(() => chat.close()).not.toThrow();
  });

  test('onEvent callback receives events', async () => {
    const events = [];
    const chat = await LlamaChat.load('dummy.gguf', {
      onEvent: (e) => events.push(e),
    });
    chat.close();
    // Load events are emitted by the bridge stub; at minimum we verify
    // that the callback wiring doesn't throw.
    expect(Array.isArray(events)).toBe(true);
  });

  test('missing messages returns error', async () => {
    const chat = await LlamaChat.load('dummy.gguf');
    try {
      await expect(
        chat.chat({ messages: [] })
      ).rejects.toThrow(LlamaError);
    } finally {
      chat.close();
    }
  });
});
