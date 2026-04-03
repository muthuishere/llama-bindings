/**
 * types.js – shared type definitions and constants for the llama browser binding.
 *
 * All objects here are plain JS; no runtime overhead.
 */

export const ResponseMode = Object.freeze({
  TEXT:        'text',
  JSON_SCHEMA: 'json_schema',
  TOOL_CALL:   'tool_call',
});

export const ErrorCode = Object.freeze({
  MODEL_LOAD_FAILED:     'MODEL_LOAD_FAILED',
  INVALID_REQUEST:       'INVALID_REQUEST',
  INFERENCE_FAILED:      'INFERENCE_FAILED',
  SCHEMA_VALIDATION:     'SCHEMA_VALIDATION_FAILED',
  TOOL_VALIDATION:       'TOOL_VALIDATION_FAILED',
  ENGINE_CLOSED:         'ENGINE_CLOSED',
  INTERNAL_BRIDGE_ERROR: 'INTERNAL_BRIDGE_ERROR',
});

/**
 * Build a normalized chat request object ready to be JSON-serialised and
 * forwarded to the WASM bridge.
 *
 * @param {object} params
 * @param {Array<{role:string,content:string}>} params.messages
 * @param {string} [params.responseMode]
 * @param {object} [params.schema]
 * @param {Array}  [params.tools]
 * @param {string} [params.toolChoice]
 * @param {object} [params.generation]
 * @returns {object}
 */
export function buildChatRequest({
  messages,
  responseMode = ResponseMode.TEXT,
  schema       = null,
  tools        = [],
  toolChoice   = 'auto',
  generation   = {},
} = {}) {
  return {
    messages,
    response_mode: responseMode,
    schema,
    tools,
    tool_choice: toolChoice,
    generation: {
      temperature:       generation.temperature       ?? 0.7,
      max_output_tokens: generation.maxOutputTokens   ?? 256,
      top_p:             generation.topP              ?? 0.95,
      top_k:             generation.topK              ?? 40,
      stop:              generation.stop              ?? [],
    },
  };
}
