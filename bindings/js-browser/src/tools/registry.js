/**
 * registry.js – ToolRegistry: maps tool names to definitions and handlers.
 *
 * Usage:
 *   const reg = new ToolRegistry();
 *   reg.register(
 *     { name: 'get_weather', description: '...', parameters: {} },
 *     async ({ city }) => ({ temperature: '30°C' })
 *   );
 *   const result = await reg.execute('get_weather', { city: 'Chennai' });
 *   const defs   = reg.definitions();
 */

export class ToolRegistry {
  /** @type {Map<string, {def: object, handler: Function}>} */
  #tools = new Map();
  /** @type {Array<object>} Preserves registration order for definitions(). */
  #order = [];

  /**
   * Register a tool.
   *
   * @param {object}   def          tool definition ({ name, description, parameters })
   * @param {Function} handler      async (args: object) => result
   * @throws {Error} if name is empty or handler is not a function
   */
  register(def, handler) {
    if (!def || !def.name) {
      throw new Error('ToolRegistry: tool name must not be empty');
    }
    if (typeof handler !== 'function') {
      throw new Error(`ToolRegistry: handler for "${def.name}" must be a function`);
    }

    if (!this.#tools.has(def.name)) {
      this.#order.push(def);
    } else {
      // Update ordered definition.
      const idx = this.#order.findIndex(d => d.name === def.name);
      if (idx !== -1) this.#order[idx] = def;
    }
    this.#tools.set(def.name, { def, handler });
  }

  /**
   * Execute a registered tool by name.
   *
   * @param {string} name  registered tool name
   * @param {object} args  decoded argument object from the model
   * @returns {Promise<*>} tool result (JSON-serialisable)
   * @throws {Error} if the tool is not registered
   */
  async execute(name, args) {
    const entry = this.#tools.get(name);
    if (!entry) {
      throw new Error(`ToolRegistry: unknown tool "${name}"`);
    }
    return entry.handler(args ?? {});
  }

  /**
   * Return all registered tool definitions in registration order.
   *
   * @returns {Array<object>}
   */
  definitions() {
    return this.#order.map(d => ({ ...d }));
  }

  /**
   * Check whether a tool with the given name is registered.
   *
   * @param {string} name
   * @returns {boolean}
   */
  has(name) {
    return this.#tools.has(name);
  }
}
