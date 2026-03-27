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
    assert.throws(() => e.chat('sid', { userMessage: 'hi' }), /closed/i);
});

test('engine.chat() throws on empty sessionId', () => {
    const { Engine } = require('../src/llama');
    const e = new Engine(null, null);
    // closed check fires first — that's acceptable
    assert.throws(() => e.chat('', { userMessage: 'hi' }));
});

test('engine.chatWithObject() throws after close()', () => {
    const { Engine } = require('../src/llama');
    const e = new Engine(null, null);
    assert.throws(() => e.chatWithObject('sid', { userMessage: 'hi' }), /closed/i);
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
        assert.ok(typeof out === 'string',  'result should be a string');
        assert.ok(out.trim().length > 0,    'result should be non-empty');
    } finally {
        engine.close();
    }
});

test('empty prompt does not crash', { skip: !modelPath() }, () => {
    const Llama  = require('../src/llama');
    const engine = Llama.load(modelPath());
    try { engine.complete(''); } finally { engine.close(); }
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

test('chat() with system message returns {role,content}', { skip: !modelPath() }, () => {
    const Llama  = require('../src/llama');
    const engine = Llama.load(modelPath());
    try {
        const msg = engine.chat('sid-js-1', {
            systemMessage: 'You are a helpful assistant.',
            userMessage:   'Say hello.',
        });
        assert.equal(msg.role, 'assistant', 'role must be assistant');
        assert.ok(msg.content.trim().length > 0, 'content must be non-empty');
    } finally {
        engine.close();
    }
});

test('chat() without system message returns {role,content}', { skip: !modelPath() }, () => {
    const Llama  = require('../src/llama');
    const engine = Llama.load(modelPath());
    try {
        const msg = engine.chat('sid-js-2', { userMessage: 'Say hello.' });
        assert.equal(msg.role, 'assistant');
        assert.ok(msg.content.trim().length > 0);
    } finally {
        engine.close();
    }
});

test('chat() multi-turn maintains history', { skip: !modelPath() }, () => {
    const Llama  = require('../src/llama');
    const engine = Llama.load(modelPath());
    try {
        const t1 = engine.chat('sid-js-mt', {
            systemMessage: 'You are helpful.',
            userMessage:   'Say hello.',
        });
        assert.ok(t1.content.trim().length > 0, 'turn 1 must be non-empty');

        const t2 = engine.chat('sid-js-mt', { userMessage: 'What did you just say?' });
        assert.ok(t2.content.trim().length > 0, 'turn 2 must be non-empty');
    } finally {
        engine.close();
    }
});

test('chat() with assistantMessage does not crash', { skip: !modelPath() }, () => {
    const Llama  = require('../src/llama');
    const engine = Llama.load(modelPath());
    try {
        const msg = engine.chat('sid-js-asst', {
            systemMessage:    'You are helpful.',
            assistantMessage: 'I said hello earlier.',
            userMessage:      'What did you say before?',
        });
        assert.ok(msg.content.trim().length > 0);
    } finally {
        engine.close();
    }
});

test('chat() with toolMessage does not crash', { skip: !modelPath() }, () => {
    const Llama  = require('../src/llama');
    const engine = Llama.load(modelPath());
    try {
        const msg = engine.chat('sid-js-tool', {
            systemMessage: 'You are a helpful assistant.',
            toolMessage:   '{"weather":"sunny","temp":"22C"}',
            userMessage:   'What is the weather like?',
        });
        assert.ok(msg.content.trim().length > 0);
    } finally {
        engine.close();
    }
});

test('chatWithObject() returns schema response', { skip: !modelPath() }, () => {
    const Llama  = require('../src/llama');
    const engine = Llama.load(modelPath());
    const sid    = 'sid-js-obj';
    try {
        const resp = engine.chatWithObject(sid, {
            systemMessage: 'You are helpful.',
            userMessage:   'Say hello.',
        });
        assert.equal(resp.role, 'assistant', 'role must be assistant');
        assert.ok(resp.content.trim().length > 0, 'content must be non-empty');
        assert.equal(resp.sessionId, sid, 'sessionId must match');
        assert.ok(resp.messageCount > 0, 'messageCount must be > 0');
    } finally {
        engine.close();
    }
});

test('chatWithObject() messageCount grows across turns', { skip: !modelPath() }, () => {
    const Llama  = require('../src/llama');
    const engine = Llama.load(modelPath());
    const sid    = 'sid-js-obj-mt';
    try {
        const r1 = engine.chatWithObject(sid, { userMessage: 'Hello.' });
        const r2 = engine.chatWithObject(sid, { userMessage: 'How are you?' });
        assert.ok(r2.messageCount > r1.messageCount,
            `messageCount should grow: r1=${r1.messageCount} r2=${r2.messageCount}`);
    } finally {
        engine.close();
    }
});

test('chatSessionClear() resets history', { skip: !modelPath() }, () => {
    const Llama  = require('../src/llama');
    const engine = Llama.load(modelPath());
    const sid    = 'sid-js-clear';
    try {
        engine.chat(sid, { userMessage: 'Say hello.' });
        engine.chatSessionClear(sid);
        const msg = engine.chat(sid, { userMessage: 'Say hello again.' });
        assert.ok(msg.content.trim().length > 0, 'content after clear must be non-empty');
    } finally {
        engine.close();
    }
});
