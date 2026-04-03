/**
 * embed.js – LlamaEmbed: browser-side embedding engine backed by the WASM bridge.
 *
 * This module is browser-only.  It does not assume Node.js native addons.
 *
 * Usage:
 *   const embed = await LlamaEmbed.load('embed-model.gguf', { onEvent });
 *   try {
 *     const vec = await embed.embed('hello world', { onEvent });
 *   } finally {
 *     embed.close();
 *   }
 */

import { LlamaError }      from './errors.js';
import { makeWasmEventCb } from './observability.js';

/**
 * Lazily-loaded WASM module singleton (shared with chat.js via module scope).
 */
let wasmModulePromise = null;

async function getWasmModule() {
  if (!wasmModulePromise) {
    // TODO(integration): import the compiled WASM module.
    wasmModulePromise = Promise.resolve(_stubWasmModule());
  }
  return wasmModulePromise;
}

function _stubWasmModule() {
  return {
    llama_embed_create(modelPath, cbPtr, userDataPtr) {
      return modelPath ? 0x2 : 0x0;
    },
    llama_embed_infer(_handle, inputText) {
      if (!inputText) return null;
      // Stub: return a 4-element vector based on input length.
      const ilen = inputText.length;
      return new Float32Array([
        ((ilen + 1) % 100) / 100,
        ((ilen + 2) % 100) / 100,
        ((ilen + 3) % 100) / 100,
        ((ilen + 4) % 100) / 100,
      ]);
    },
    llama_embed_destroy(_handle) {},
  };
}

export class LlamaEmbed {
  /** @type {object} */  #wasm;
  /** @type {number} */  #handle;
  /** @type {boolean} */ #closed = false;

  /**
   * @private – use {@link LlamaEmbed.load} instead.
   */
  constructor(wasm, handle) {
    this.#wasm   = wasm;
    this.#handle = handle;
  }

  /**
   * Load an embedding engine from a model file (or URL in the browser).
   *
   * @param {string}   modelPath  path or URL to the GGUF embedding model
   * @param {object}   [opts]
   * @param {Function} [opts.onEvent]  optional event callback
   * @returns {Promise<LlamaEmbed>}
   */
  static async load(modelPath, { onEvent } = {}) {
    const wasm = await getWasmModule();
    const cb   = makeWasmEventCb(onEvent);
    const handle = wasm.llama_embed_create(modelPath, cb, null);
    if (!handle) {
      throw new LlamaError('MODEL_LOAD_FAILED',
        `Failed to load embed model: ${modelPath}`);
    }
    return new LlamaEmbed(wasm, handle);
  }

  /**
   * Generate a float vector for the given text.
   *
   * @param {string}   text  input text (must not be empty)
   * @param {object}   [opts]
   * @param {Function} [opts.onEvent]  optional per-call event callback
   * @returns {Promise<Float32Array>}  embedding vector
   */
  async embed(text, opts = {}) {
    if (this.#closed) {
      throw new LlamaError('ENGINE_CLOSED', 'LlamaEmbed engine is closed');
    }
    if (!text || text.length === 0) {
      throw new LlamaError('INVALID_REQUEST', 'input text must not be empty');
    }

    const result = this.#wasm.llama_embed_infer(this.#handle, text);
    if (!result || result.length === 0) {
      throw new LlamaError('INFERENCE_FAILED', 'WASM bridge returned empty vector');
    }

    return result instanceof Float32Array ? result : new Float32Array(result);
  }

  /**
   * Release the native engine resources.
   * Safe to call multiple times.
   */
  close() {
    if (this.#closed) return;
    this.#closed = true;
    this.#wasm.llama_embed_destroy(this.#handle);
  }
}
