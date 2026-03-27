'use strict';

/**
 * llama.js — thin Node.js binding to libllama_bridge.
 *
 * Usage:
 *
 *   const Llama = require('./llama');
 *
 *   const engine = Llama.load('model.gguf');
 *   const out    = engine.complete('Say hello.');
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

const voidPtr  = ref.refType(ref.types.void);

let _lib;

function getLib() {
    if (!_lib) {
        _lib = ffi.Library(libName, {
            llama_engine_create:      [voidPtr,  ['string']],
            llama_engine_complete:    [voidPtr,  [voidPtr, 'string']],
            llama_engine_free_string: ['void',   [voidPtr]],
            llama_engine_destroy:     ['void',   [voidPtr]],
        });
    }
    return _lib;
}

// ---------------------------------------------------------------------------
// Engine class
// ---------------------------------------------------------------------------

/**
 * Engine wraps an opaque native engine handle.
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

    /**
     * Run completion for the given prompt.
     *
     * @param {string} prompt
     * @returns {string} completion text
     * @throws {Error} if inference fails or engine is closed
     */
    complete(prompt) {
        if (!this._handle) {
            throw new Error('llama: engine has been closed');
        }
        if (prompt == null) {
            prompt = '';
        }

        const resultPtr = this._lib.llama_engine_complete(this._handle, prompt);
        if (ref.isNull(resultPtr)) {
            throw new Error('llama: completion returned null');
        }

        try {
            // Read the null-terminated UTF-8 string from the pointer.
            const str = ref.readCString(resultPtr, 0);
            return str;
        } finally {
            this._lib.llama_engine_free_string(resultPtr);
        }
    }

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
