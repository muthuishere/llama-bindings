'use strict';

/**
 * llama.js — thin Node.js binding to libllama_bridge.
 *
 * Usage:
 *
 *   const Llama = require('./llama');
 *
 *   const engine = Llama.load('model.gguf');
 *
 *   // Raw completion (no chat template)
 *   const out = engine.complete('Say hello.');
 *
 *   // Session-based chat — pass an array of {role, content} objects,
 *   // returns {role, content}.
 *   const msg = engine.chat('sid-1', [
 *     { role: 'system', content: 'You are helpful.' },
 *     { role: 'user',   content: 'What is 2+2?' },
 *   ]);
 *   console.log(msg.content);
 *
 *   // Multi-turn: only pass the new messages each turn.
 *   const msg2 = engine.chat('sid-1', [
 *     { role: 'user', content: 'Tell me more.' },
 *   ]);
 *
 *   // Inject a tool response before the user message.
 *   engine.chat('sid-1', [
 *     { role: 'tool', content: '{"result": 42}' },
 *     { role: 'user', content: 'What was the result?' },
 *   ]);
 *
 *   // Richer schema response — returns plain JS object with
 *   // role, content, sessionId, messageCount.
 *   const resp = engine.chatWithObject('sid-1', [
 *     { role: 'user', content: 'Hello.' },
 *   ]);
 *   console.log(resp.role, resp.content, resp.sessionId, resp.messageCount);
 *
 *   // Clear session history
 *   engine.chatSessionClear('sid-1');
 *
 *   engine.close();
 *
 * Environment variable LLAMA_BRIDGE_LIB_DIR can be used to point to the
 * directory that contains libllama_bridge.so (or .dylib on macOS).
 */

const ffi  = require('ffi-napi');
const ref  = require('ref-napi');
const path = require('path');

// ---------------------------------------------------------------------------
// Locate the shared library
// ---------------------------------------------------------------------------

const libDir = process.env.LLAMA_BRIDGE_LIB_DIR
    || path.resolve(__dirname, '../../../build');

const libName = (() => {
    if (process.platform === 'win32')  return path.join(libDir, 'llama_bridge.dll');
    if (process.platform === 'darwin') return path.join(libDir, 'libllama_bridge.dylib');
    return path.join(libDir, 'libllama_bridge.so');
})();

// ---------------------------------------------------------------------------
// FFI bindings to the C bridge
// ---------------------------------------------------------------------------

const voidPtr = ref.refType(ref.types.void);

let _lib;

function getLib() {
    if (!_lib) {
        _lib = ffi.Library(libName, {
            /* core */
            llama_engine_create:                        [voidPtr, ['string']],
            llama_engine_complete:                      [voidPtr, [voidPtr, 'string']],
            llama_engine_free_string:                   ['void',  [voidPtr]],
            llama_engine_destroy:                       ['void',  [voidPtr]],
            /* chat — array-based (primary interface) */
            llama_engine_chat_messages:                 [voidPtr, [voidPtr, 'string', 'string']],
            llama_engine_chat_with_object_messages:     [voidPtr, [voidPtr, 'string', 'string']],
            llama_engine_chat_session_clear:            ['void',  [voidPtr, 'string']],
        });
    }
    return _lib;
}

// ---------------------------------------------------------------------------
// Helper: read result pointer as string and free it
// ---------------------------------------------------------------------------

function _readAndFree(lib, resultPtr) {
    if (ref.isNull(resultPtr)) return null;
    try {
        return ref.readCString(resultPtr, 0);
    } finally {
        lib.llama_engine_free_string(resultPtr);
    }
}

// ---------------------------------------------------------------------------
// Engine class
// ---------------------------------------------------------------------------

/**
 * Engine wraps an opaque native engine handle.
 * Create with {@link load}; release with {@link Engine#close}.
 */
class Engine {
    /**
     * @param {Buffer} handle — opaque native pointer
     * @param {object} lib    — ffi library instance
     */
    constructor(handle, lib) {
        this._handle = handle;
        this._lib    = lib;
    }

    _ensureOpen() {
        if (!this._handle) throw new Error('llama: engine has been closed');
    }

    // -----------------------------------------------------------------------
    // Raw completion (no chat template)
    // -----------------------------------------------------------------------

    /**
     * Run raw completion for the given prompt (no chat template applied).
     *
     * @param {string} prompt
     * @returns {string} completion text
     */
    complete(prompt) {
        this._ensureOpen();
        if (prompt == null) prompt = '';
        const resultPtr = this._lib.llama_engine_complete(this._handle, prompt);
        if (ref.isNull(resultPtr)) throw new Error('llama: completion returned null');
        return _readAndFree(this._lib, resultPtr);
    }

    // -----------------------------------------------------------------------
    // Chat API
    // -----------------------------------------------------------------------

    /**
     * Session-based chat turn.  Returns {role, content}.
     *
     * The engine maintains conversation history keyed by sessionId.
     * Pass an ordered array of {role, content} messages to append to
     * the session for this turn.  Supported roles: "system", "user",
     * "assistant", "tool".  A "system" entry sets or replaces the session
     * system prompt.
     *
     * @param {string} sessionId  conversation identifier (auto-created on first call)
     * @param {Array<{role:string, content:string}>} messages  messages to append
     * @returns {{role: string, content: string}}
     */
    chat(sessionId, messages) {
        this._ensureOpen();
        if (!sessionId) throw new Error('llama: sessionId must not be empty');
        const msgs = messages || [];
        const resultPtr = this._lib.llama_engine_chat_messages(
            this._handle,
            sessionId,
            JSON.stringify(msgs)
        );
        if (ref.isNull(resultPtr)) throw new Error('llama: chat returned null');
        const json = _readAndFree(this._lib, resultPtr);
        return JSON.parse(json);
    }

    /**
     * Session-based chat turn that returns a richer plain JS object.
     *
     * Same as {@link chat} but the returned object also contains
     * {@code sessionId} and {@code messageCount}.
     *
     * @param {string} sessionId  conversation identifier
     * @param {Array<{role:string, content:string}>} messages  messages to append
     * @returns {{role: string, content: string, sessionId: string, messageCount: number}}
     */
    chatWithObject(sessionId, messages) {
        this._ensureOpen();
        if (!sessionId) throw new Error('llama: sessionId must not be empty');
        const msgs = messages || [];
        const resultPtr = this._lib.llama_engine_chat_with_object_messages(
            this._handle,
            sessionId,
            JSON.stringify(msgs)
        );
        if (ref.isNull(resultPtr)) throw new Error('llama: chatWithObject returned null');
        const json = _readAndFree(this._lib, resultPtr);
        return JSON.parse(json);
    }

    /**
     * Clear all history for the named session (including system message).
     * The session slot is released for reuse.
     *
     * @param {string} sessionId  session identifier
     */
    chatSessionClear(sessionId) {
        this._ensureOpen();
        if (sessionId) {
            this._lib.llama_engine_chat_session_clear(this._handle, sessionId);
        }
    }

    // -----------------------------------------------------------------------

    /**
     * Close the engine and free all native resources.
     * Safe to call multiple times.
     */
    close() {
        if (this._handle) {
            this._lib.llama_engine_destroy(this._handle);
            this._handle = null;
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Load a GGUF model and return a new {@link Engine}.
 *
 * @param {string} modelPath — path to the GGUF model file
 * @returns {Engine}
 * @throws {Error} if modelPath is empty or the model cannot be loaded
 */
function load(modelPath) {
    if (!modelPath || modelPath.trim() === '') {
        throw new Error('llama: modelPath must not be empty');
    }
    const lib    = getLib();
    const handle = lib.llama_engine_create(modelPath);
    if (ref.isNull(handle)) {
        throw new Error(`llama: failed to load model: ${modelPath}`);
    }
    return new Engine(handle, lib);
}

module.exports = { load, Engine };
