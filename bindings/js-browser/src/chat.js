/**
 * chat.js – LlamaChat: browser-side chat engine backed by the WASM bridge.
 *
 * This module is browser-only.  It does not assume Node.js native addons.
 *
 * Usage:
 *   const chat = await LlamaChat.load('chat-model.gguf', { onEvent });
 *   try {
 *     const resp = await chat.chat({ messages, responseMode, ... }, { temperature, ... });
 *   } finally {
 *     chat.close();
 *   }
 */

import { LlamaError }        from './errors.js';
import { makeWasmEventCb }   from './observability.js';
import { buildChatRequest, ResponseMode } from './types.js';

/**
 * Lazily-loaded WASM module singleton.
 * Replaced by the actual Emscripten-generated module once llama.cpp is
 * compiled to WASM (see wasm/ directory).
 *
 * @type {Promise<object>|null}
 */
let wasmModulePromise = null;

async function getWasmModule() {
  if (!wasmModulePromise) {
    // TODO(integration): import the compiled WASM module.
    // wasmModulePromise = import('../../wasm/build/llama_bridge.js').then(m => m.default());
    wasmModulePromise = Promise.resolve(_stubWasmModule());
  }
  return wasmModulePromise;
}

/**
 * Stub WASM module used until the real Emscripten build is available.
 * Every exported function follows the same signature as the real bridge.
 */
function _stubWasmModule() {
  return {
    llama_chat_create(modelPath, cbPtr, userDataPtr) {
      return modelPath ? 0x1 : 0x0; // non-zero handle
    },
    llama_chat_infer_json(handle, requestJson) {
      const req = JSON.parse(requestJson);
      if (!req.messages || req.messages.length === 0) {
        return JSON.stringify({ type: 'error', error: { code: 'INVALID_REQUEST', message: 'messages required' } });
      }
      if (req.response_mode === 'json_schema') {
        return JSON.stringify({ type: 'structured_json', json: {}, finish_reason: 'stop', usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 } });
      }
      if (req.response_mode === 'tool_call') {
        return JSON.stringify({ type: 'tool_call', tool_calls: [{ id: 'call_1', name: 'stub_tool', arguments: {} }], finish_reason: 'tool_call', usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 } });
      }
      return JSON.stringify({ type: 'assistant_text', text: 'stub response', finish_reason: 'stop', usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 } });
    },
    llama_chat_destroy(_handle) {},
  };
}

export class LlamaChat {
  /** @type {object} */  #wasm;
  /** @type {number} */  #handle;
  /** @type {boolean} */ #closed = false;

  /**
   * @private – use {@link LlamaChat.load} instead.
   */
  constructor(wasm, handle) {
    this.#wasm   = wasm;
    this.#handle = handle;
  }

  /**
   * Load a chat engine from a model file (or URL in the browser).
   *
   * @param {string}   modelPath  path or URL to the GGUF model
   * @param {object}   [opts]
   * @param {Function} [opts.onEvent]  optional event callback
   * @returns {Promise<LlamaChat>}
   */
  static async load(modelPath, { onEvent } = {}) {
    const wasm = await getWasmModule();
    const cb   = makeWasmEventCb(onEvent);
    const handle = wasm.llama_chat_create(modelPath, cb, null);
    if (!handle) {
      throw new LlamaError('MODEL_LOAD_FAILED',
        `Failed to load chat model: ${modelPath}`);
    }
    return new LlamaChat(wasm, handle);
  }

  /**
   * Run a chat inference.
   *
   * @param {object}   request
   * @param {Array}    request.messages
   * @param {string}   [request.responseMode]
   * @param {object}   [request.schema]
   * @param {Array}    [request.tools]
   * @param {string}   [request.toolChoice]
   * @param {object}   [opts]
   * @param {number}   [opts.temperature]
   * @param {number}   [opts.maxOutputTokens]
   * @param {number}   [opts.topP]
   * @param {number}   [opts.topK]
   * @param {Array}    [opts.stop]
   * @param {Function} [opts.onEvent]
   * @returns {Promise<object>}  normalized response
   */
  async chat(request, opts = {}) {
    if (this.#closed) {
      throw new LlamaError('ENGINE_CLOSED', 'LlamaChat engine is closed');
    }

    const normalized = buildChatRequest({
      messages:     request.messages,
      responseMode: request.responseMode ?? ResponseMode.TEXT,
      schema:       request.schema,
      tools:        request.tools,
      toolChoice:   request.toolChoice,
      generation: {
        temperature:     opts.temperature,
        maxOutputTokens: opts.maxOutputTokens,
        topP:            opts.topP,
        topK:            opts.topK,
        stop:            opts.stop,
      },
    });

    const reqJson  = JSON.stringify(normalized);
    const respJson = this.#wasm.llama_chat_infer_json(this.#handle, reqJson);

    if (!respJson) {
      throw new LlamaError('INFERENCE_FAILED', 'WASM bridge returned null');
    }

    const resp = JSON.parse(respJson);
    if (resp.type === 'error') {
      throw new LlamaError(
        resp.error?.code    ?? 'INFERENCE_FAILED',
        resp.error?.message ?? 'unknown error',
      );
    }

    return resp;
  }

  /**
   * Release the native engine resources.
   * Safe to call multiple times.
   */
  close() {
    if (this.#closed) return;
    this.#closed = true;
    this.#wasm.llama_chat_destroy(this.#handle);
  }
}
