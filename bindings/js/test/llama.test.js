'use strict';

/**
 * llama.test.js — tests for the JS binding.
 *
 * Unit tests run without a real model and validate error-handling logic.
 * Integration tests are skipped when LLAMA_TEST_MODEL is not set.
 *
 * Run with: node --test test/llama.test.js
 */

const assert = require('node:assert/strict');
const test   = require('node:test');

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function modelPath() {
    return process.env.LLAMA_TEST_MODEL || null;
}

// ---------------------------------------------------------------------------
// Unit tests — no native library required
// ---------------------------------------------------------------------------

test('load() throws on empty string', () => {
    const Llama = require('../src/llama');
    assert.throws(() => Llama.load(''),   /modelPath must not be empty/i);
});

test('load() throws on null', () => {
    const Llama = require('../src/llama');
    assert.throws(() => Llama.load(null), /modelPath must not be empty/i);
});

test('engine.complete() throws after close()', () => {
    const { Engine } = require('../src/llama');
    const e = new Engine(null, null);
    assert.throws(() => e.complete('hi'), /closed/i);
});

test('engine.chat() throws after close()', () => {
    const { Engine } = require('../src/llama');
    const e = new Engine(null, null);
    assert.throws(() => e.chat('sys', 'user'), /closed/i);
});

test('engine.chatWithMessages() throws on empty array', () => {
    const { Engine } = require('../src/llama');
    const e = new Engine(null, null);
    // closed check fires first
    assert.throws(() => e.chatWithMessages([]), /closed/i);
});

test('engine.chatSession() throws after close()', () => {
    const { Engine } = require('../src/llama');
    const e = new Engine(null, null);
    assert.throws(() => e.chatSession('s1', 'hi'), /closed/i);
});

test('engine.chatWithTools() throws after close()', () => {
    const { Engine } = require('../src/llama');
    const e = new Engine(null, null);
    assert.throws(() => e.chatWithTools([{ role: 'user', content: 'hi' }], '[]'), /closed/i);
});

test('engine.close() is idempotent with null handle', () => {
    const { Engine } = require('../src/llama');
    const e = new Engine(null, null);
    assert.doesNotThrow(() => e.close());
    assert.doesNotThrow(() => e.close());
});

// ---------------------------------------------------------------------------
// Integration tests — require LLAMA_TEST_MODEL
// ---------------------------------------------------------------------------

test('smoke: load → complete → close', { skip: !modelPath() }, () => {
    const Llama  = require('../src/llama');
    const engine = Llama.load(modelPath());
    try {
        const out = engine.complete('Say hello in one short sentence.');
        assert.ok(typeof out === 'string', 'result should be a string');
        assert.ok(out.trim().length > 0,   'result should be non-empty');
    } finally {
        engine.close();
    }
});

test('factual: completion is non-empty', { skip: !modelPath() }, () => {
    const Llama  = require('../src/llama');
    const engine = Llama.load(modelPath());
    try {
        const out = engine.complete('Complete this: The capital of France is');
        assert.ok(out.trim().length > 0, 'factual completion must be non-empty');
    } finally {
        engine.close();
    }
});

test('empty prompt does not crash', { skip: !modelPath() }, () => {
    const Llama  = require('../src/llama');
    const engine = Llama.load(modelPath());
    try {
        engine.complete('');
    } finally {
        engine.close();
    }
});

test('repeated completions do not corrupt state', { skip: !modelPath() }, () => {
    const Llama  = require('../src/llama');
    const engine = Llama.load(modelPath());
    try {
        for (let i = 0; i < 5; i++) {
            const out = engine.complete('Say hello.');
            assert.ok(out.trim().length > 0, `completion ${i} must be non-empty`);
        }
    } finally {
        engine.close();
    }
});

test('invalid model path throws', { skip: !modelPath() }, () => {
    const Llama = require('../src/llama');
    assert.throws(() => Llama.load('/nonexistent/model.gguf'), /failed to load model/i);
});

test('create/destroy cycle does not crash', { skip: !modelPath() }, () => {
    const Llama = require('../src/llama');
    for (let i = 0; i < 3; i++) {
        const engine = Llama.load(modelPath());
        engine.close();
    }
});

// ---------------------------------------------------------------------------
// Chat integration tests
// ---------------------------------------------------------------------------

test('chat() with system message returns non-empty', { skip: !modelPath() }, () => {
    const Llama  = require('../src/llama');
    const engine = Llama.load(modelPath());
    try {
        const out = engine.chat('You are a helpful assistant.', 'Say hello.');
        assert.ok(out.trim().length > 0, 'chat response must be non-empty');
    } finally {
        engine.close();
    }
});

test('chat() without system message returns non-empty', { skip: !modelPath() }, () => {
    const Llama  = require('../src/llama');
    const engine = Llama.load(modelPath());
    try {
        const out = engine.chat('', 'Say hello.');
        assert.ok(out.trim().length > 0, 'chat (no system) response must be non-empty');
    } finally {
        engine.close();
    }
});

test('chatWithMessages() returns non-empty', { skip: !modelPath() }, () => {
    const Llama  = require('../src/llama');
    const engine = Llama.load(modelPath());
    try {
        const out = engine.chatWithMessages([
            { role: 'system', content: 'You are a helpful assistant.' },
            { role: 'user',   content: 'Say hello in one sentence.' },
        ]);
        assert.ok(out.trim().length > 0, 'chatWithMessages response must be non-empty');
    } finally {
        engine.close();
    }
});

test('chatWithMessages() throws on empty array', { skip: !modelPath() }, () => {
    const Llama  = require('../src/llama');
    const engine = Llama.load(modelPath());
    try {
        assert.throws(() => engine.chatWithMessages([]), /non-empty/i);
    } finally {
        engine.close();
    }
});

test('chatSession() multi-turn does not crash', { skip: !modelPath() }, () => {
    const Llama  = require('../src/llama');
    const engine = Llama.load(modelPath());
    try {
        const t1 = engine.chatSession('sid-js-1', 'Say hello.');
        assert.ok(t1.trim().length > 0, 'turn 1 must be non-empty');
        const t2 = engine.chatSession('sid-js-1', 'What did you just say?');
        assert.ok(t2.trim().length > 0, 'turn 2 must be non-empty');
    } finally {
        engine.close();
    }
});

test('chatSessionSetSystem() then chatSession() works', { skip: !modelPath() }, () => {
    const Llama  = require('../src/llama');
    const engine = Llama.load(modelPath());
    try {
        engine.chatSessionSetSystem('sid-js-sys', 'You are a helpful assistant.');
        const out = engine.chatSession('sid-js-sys', 'Say hello.');
        assert.ok(out.trim().length > 0, 'session with system must return non-empty');
    } finally {
        engine.close();
    }
});

test('chatSessionClear() then chatSession() works', { skip: !modelPath() }, () => {
    const Llama  = require('../src/llama');
    const engine = Llama.load(modelPath());
    try {
        engine.chatSession('sid-js-clear', 'Say hello.');
        engine.chatSessionClear('sid-js-clear');
        const out = engine.chatSession('sid-js-clear', 'Say hello again.');
        assert.ok(out.trim().length > 0, 'response after clear must be non-empty');
    } finally {
        engine.close();
    }
});

test('chatWithTools() returns raw non-empty output', { skip: !modelPath() }, () => {
    const Llama  = require('../src/llama');
    const engine = Llama.load(modelPath());
    try {
        const tools = '[{"name":"get_weather","description":"Get weather","parameters":{"location":{"type":"string"}}}]';
        const out   = engine.chatWithTools(
            [{ role: 'user', content: 'What is the weather in Paris?' }],
            tools
        );
        assert.ok(out.trim().length > 0, 'chatWithTools response must be non-empty');
    } finally {
        engine.close();
    }
});

test('chatWithTools() throws on empty messages', { skip: !modelPath() }, () => {
    const Llama  = require('../src/llama');
    const engine = Llama.load(modelPath());
    try {
        assert.throws(() => engine.chatWithTools([], '[]'), /non-empty/i);
    } finally {
        engine.close();
    }
});
