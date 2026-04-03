/**
 * observability.js – event helpers for the llama browser binding.
 *
 * The WASM bridge emits JSON event strings.  This module parses them and
 * dispatches to user-supplied callbacks.
 */

/**
 * Parse a bridge event JSON string into a plain object.
 *
 * @param {string} json
 * @returns {object|null}
 */
export function parseEvent(json) {
  try {
    return JSON.parse(json);
  } catch {
    return null;
  }
}

/**
 * Call a user-supplied onEvent callback safely.
 * Exceptions thrown by the callback are swallowed so they cannot affect
 * the inference path.
 *
 * @param {Function|null|undefined} onEvent
 * @param {object} event
 */
export function emitEvent(onEvent, event) {
  if (typeof onEvent === 'function' && event) {
    try {
      onEvent(event);
    } catch {
      // Swallow – callbacks must not affect inference.
    }
  }
}

/**
 * Wrap a WASM bridge event callback so it receives parsed event objects.
 *
 * @param {Function|null|undefined} onEvent  user callback
 * @returns {Function|null}  C-compatible string callback for the WASM module
 */
export function makeWasmEventCb(onEvent) {
  if (!onEvent) return null;
  return (eventJson) => {
    const evt = parseEvent(eventJson);
    emitEvent(onEvent, evt);
  };
}
