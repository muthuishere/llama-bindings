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
 *   // Simple completion (no chat template)
 *   const out = engine.complete('Say hello.');
 *
 *   // One-shot chat
 *   const reply = engine.chat('You are helpful.', 'What is 2+2?');
 *
 *   // Chat with message objects (chatWithObject)
 *   const reply2 = engine.chatWithMessages([
 *     { role: 'system', content: 'You are helpful.' },
 *     { role: 'user',   content: 'What is 2+2?' },
 *   ]);
 *
 *   // Session-based multi-turn chat
 *   const t1 = engine.chatSession('sid-1', 'Hello!');
 *   const t2 = engine.chatSession('sid-1', 'What did I just say?');
 *
 *   // Chat with tool definitions (raw output)
 *   const tools = '[{"name":"add","description":"Add numbers","parameters":{}}]';
 *   const raw = engine.chatWithTools([{ role: 'user', content: 'Add 3 and 4' }], tools);
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
            /* v1: simple completion */
            llama_engine_create:                  [voidPtr, ['string']],
            llama_engine_complete:                [voidPtr, [voidPtr, 'string']],
            llama_engine_free_string:             ['void',  [voidPtr]],
            llama_engine_destroy:                 ['void',  [voidPtr]],
            /* v2: chat */
            llama_engine_chat:                    [voidPtr, [voidPtr, 'string', 'string']],
            llama_engine_chat_with_messages:      [voidPtr, [voidPtr, 'pointer', 'pointer', 'int']],
            llama_engine_chat_session:            [voidPtr, [voidPtr, 'string', 'string']],
            llama_engine_chat_session_set_system: ['void',  [voidPtr, 'string', 'string']],
            llama_engine_chat_session_clear:      ['void',  [voidPtr, 'string']],
            llama_engine_chat_with_tools:         [voidPtr, [voidPtr, 'pointer', 'pointer', 'int', 'string']],
        });
    }
    return _lib;
}

// ---------------------------------------------------------------------------
// Helper: build a native char** from a JS string array
//
// Returns { ptrBuf, cBufs } where:
//   ptrBuf  — a Buffer holding n native pointers (pass to C as char**)
//   cBufs   — array of Buffers holding the null-terminated C strings
//             (kept alive to prevent GC during the native call)
// ---------------------------------------------------------------------------

function _makeStringArray(strings) {
    const ptrSize = ref.sizeof.pointer;
    const cBufs   = strings.map(s => Buffer.from((s || '') + '\0', 'utf8'));
    const ptrBuf  = Buffer.alloc(ptrSize * strings.length);
    cBufs.forEach((cb, i) => ref.writePointer(ptrBuf, i * ptrSize, cb));
    return { ptrBuf, cBufs };
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
    // v1: simple completion (no chat template)
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
    // v2: chat
    // -----------------------------------------------------------------------

    /**
     * One-shot chat with an optional system message and a user message.
     * The model's built-in chat template is applied automatically.
     *
     * @param {string} systemMsg  system prompt (null or '' to omit)
     * @param {string} userMsg    user message
     * @returns {string} assistant response
     */
    chat(systemMsg, userMsg) {
        this._ensureOpen();
        const resultPtr = this._lib.llama_engine_chat(
            this._handle,
            systemMsg || '',
            userMsg   || ''
        );
        if (ref.isNull(resultPtr)) throw new Error('llama: chat returned null');
        return _readAndFree(this._lib, resultPtr);
    }

    /**
     * Chat with an explicit array of message objects (chatWithObject equivalent).
     *
     * Each message must have { role: string, content: string }.
     * Role must be 'system', 'user', or 'assistant'.
     *
     * @param {Array<{role:string,content:string}>} messages
     * @returns {string} assistant response
     */
    chatWithMessages(messages) {
        this._ensureOpen();
        if (!Array.isArray(messages) || messages.length === 0) {
            throw new Error('llama: messages must be a non-empty array');
        }
        const roles    = messages.map(m => m.role    || 'user');
        const contents = messages.map(m => m.content || '');
        const { ptrBuf: rolesBuf,    cBufs: rolesC    } = _makeStringArray(roles);
        const { ptrBuf: contentsBuf, cBufs: contentsC } = _makeStringArray(contents);

        // rolesC and contentsC must remain referenced until the native call
        // returns to prevent the GC from collecting the underlying Buffers.
        // The explicit reference below satisfies the linter while keeping
        // them alive across the synchronous native call.
        const _keepAlive1 = rolesC.length + contentsC.length;

        const resultPtr = this._lib.llama_engine_chat_with_messages(
            this._handle, rolesBuf, contentsBuf, messages.length);
        if (ref.isNull(resultPtr)) throw new Error('llama: chatWithMessages returned null');
        return _readAndFree(this._lib, resultPtr);
    }

    /**
     * Session-based multi-turn chat.
     *
     * Conversation history is maintained inside the native engine, keyed by
     * sessionId.  Each call appends userMsg to the session and returns the
     * next assistant turn.
     *
     * Call {@link chatSessionSetSystem} before the first turn to set a system
     * prompt.
     *
     * @param {string} sessionId  unique session identifier
     * @param {string} userMsg    the user's message
     * @returns {string} assistant response
     */
    chatSession(sessionId, userMsg) {
        this._ensureOpen();
        if (!sessionId) throw new Error('llama: sessionId must not be empty');
        const resultPtr = this._lib.llama_engine_chat_session(
            this._handle, sessionId, userMsg || '');
        if (ref.isNull(resultPtr)) throw new Error('llama: chatSession returned null');
        return _readAndFree(this._lib, resultPtr);
    }

    /**
     * Set (or replace) the system message for a named session.
     * Call before the first {@link chatSession} turn if a system prompt is
     * required.
     *
     * @param {string} sessionId  session identifier
     * @param {string} systemMsg  system prompt (null or '' to clear)
     */
    chatSessionSetSystem(sessionId, systemMsg) {
        this._ensureOpen();
        if (!sessionId) throw new Error('llama: sessionId must not be empty');
        this._lib.llama_engine_chat_session_set_system(
            this._handle, sessionId, systemMsg || '');
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

    /**
     * Chat with tool definitions.
     *
     * The tool definitions are injected into the system message so that any
     * model can reason about available tools.  The <em>raw</em> model output
     * is returned — the caller is responsible for parsing and executing any
     * tool calls the model emits.
     *
     * @param {Array<{role:string,content:string}>} messages  non-empty message array
     * @param {string} toolsJson  JSON array of tool definitions
     *   (OpenAI-compatible format, e.g.
     *   '[{"name":"…","description":"…","parameters":{…}}]')
     * @returns {string} raw assistant response (may contain a tool-call JSON object)
     */
    chatWithTools(messages, toolsJson) {
        this._ensureOpen();
        if (!Array.isArray(messages) || messages.length === 0) {
            throw new Error('llama: messages must be a non-empty array');
        }
        const roles    = messages.map(m => m.role    || 'user');
        const contents = messages.map(m => m.content || '');
        const { ptrBuf: rolesBuf,    cBufs: rolesC    } = _makeStringArray(roles);
        const { ptrBuf: contentsBuf, cBufs: contentsC } = _makeStringArray(contents);

        // rolesC and contentsC must remain referenced until the native call
        // returns to prevent the GC from collecting the underlying Buffers.
        const _keepAlive2 = rolesC.length + contentsC.length;

        const resultPtr = this._lib.llama_engine_chat_with_tools(
            this._handle, rolesBuf, contentsBuf, messages.length,
            toolsJson || '[]');
        if (ref.isNull(resultPtr)) throw new Error('llama: chatWithTools returned null');
        return _readAndFree(this._lib, resultPtr);
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

