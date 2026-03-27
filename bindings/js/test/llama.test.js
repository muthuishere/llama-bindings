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
const path   = require('path');

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Returns the model path from LLAMA_TEST_MODEL, or null if not set.
 */
function modelPath() {
    return process.env.LLAMA_TEST_MODEL || null;
}

// ---------------------------------------------------------------------------
// Unit tests — no native library required
// ---------------------------------------------------------------------------

test('load() throws on empty string', () => {
    // We import lazily so this test can run even if the shared library is absent
    // when LLAMA_TEST_MODEL is not set.
    const Llama = require('../src/llama');
    assert.throws(() => Llama.load(''),  /modelPath must not be empty/i);
});

test('load() throws on null', () => {
    const Llama = require('../src/llama');
    assert.throws(() => Llama.load(null), /modelPath must not be empty/i);
});

test('engine.complete() throws after close()', () => {
    // We cannot call load() without a native library, so we test the Engine
    // class directly by constructing it with a null handle.
    const { Engine } = require('../src/llama');
    const fakeEngine = new Engine(null, null);
    assert.throws(() => fakeEngine.complete('hi'), /closed/i);
});

test('engine.close() is idempotent with null handle', () => {
    const { Engine } = require('../src/llama');
    const fakeEngine = new Engine(null, null);
    assert.doesNotThrow(() => fakeEngine.close());
    assert.doesNotThrow(() => fakeEngine.close());
});

// ---------------------------------------------------------------------------
// Integration tests — require LLAMA_TEST_MODEL
// ---------------------------------------------------------------------------

test('smoke: load → complete → close', { skip: !modelPath() }, () => {
    const Llama = require('../src/llama');
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
        // Must not throw; result may be empty.
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
