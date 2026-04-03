/**
 * errors.js – error class for the llama browser binding.
 */

export class LlamaError extends Error {
  /**
   * @param {string} code    one of the ErrorCode constants
   * @param {string} message human-readable description
   */
  constructor(code, message) {
    super(`[${code}] ${message}`);
    this.name    = 'LlamaError';
    this.code    = code;
  }
}
