/**
 * agent.js – Agent: orchestrates chat, embedding, knowledge retrieval, and
 * tool execution in an agentic loop.
 *
 * The Agent:
 *  - Owns a LlamaChat engine, a LlamaEmbed engine, a KnowledgeStore, and a
 *    ToolRegistry.
 *  - Maintains per-session conversation history in memory.
 *  - On each chat() call:
 *      1. Embeds the user message and searches the knowledge store.
 *      2. Injects relevant context into the system prompt.
 *      3. Loops up to MAX_TOOL_ITERATIONS times:
 *         - If the model returns a tool call, executes it and feeds the result
 *           back as a "tool" role message.
 *         - If the model returns text, returns it to the caller.
 *
 * Usage:
 *   const agent = await Agent.create('chat.gguf', 'embed.gguf', ':memory:');
 *   await agent.addDocument('The capital of France is Paris.');
 *   agent.addTool({ name: 'lookup', ... }, async (args) => '...');
 *   const reply = await agent.chat('session-1', 'What is the capital of France?');
 *   agent.close();
 */

import { zipSync, unzipSync } from 'fflate';
import { LlamaChat }      from '../chat.js';
import { LlamaEmbed }     from '../embed.js';
import { KnowledgeStore } from '../knowledge/store.js';
import { ToolRegistry }   from '../tools/registry.js';

const MAX_TOOL_ITERATIONS = 10;

export class Agent {
  /** @type {LlamaChat}      */ #chat;
  /** @type {LlamaEmbed}     */ #embed;
  /** @type {KnowledgeStore} */ #store;
  /** @type {ToolRegistry}   */ #registry;
  /** @type {string}         */ #chatModelPath;
  /** @type {string}         */ #embedModelPath;
  /** @type {Map<string,Array>} sessionId → message history */
  #sessions = new Map();
  /** @type {boolean}        */ #closed = false;

  /**
   * @private – use {@link Agent.create} instead.
   */
  constructor(chat, embed, store, registry, chatModelPath, embedModelPath) {
    this.#chat           = chat;
    this.#embed          = embed;
    this.#store          = store;
    this.#registry       = registry;
    this.#chatModelPath  = chatModelPath;
    this.#embedModelPath = embedModelPath;
  }

  /**
   * Create an Agent that owns all of its engines and stores.
   *
   * @param {string} chatModelPath   path / URL to the chat GGUF model
   * @param {string} embedModelPath  path / URL to the embed GGUF model
   * @param {string} [storagePath]   ignored (JS store is always in-memory)
   * @param {object} [opts]
   * @param {Function} [opts.onEvent]  optional event callback forwarded to engines
   * @returns {Promise<Agent>}
   */
  static async create(chatModelPath, embedModelPath, storagePath = ':memory:', opts = {}) {
    const chat  = await LlamaChat.load(chatModelPath,  { onEvent: opts.onEvent });
    const embed = await LlamaEmbed.load(embedModelPath, { onEvent: opts.onEvent });
    const store    = new KnowledgeStore();
    const registry = new ToolRegistry();
    return new Agent(chat, embed, store, registry, chatModelPath, embedModelPath);
  }

  /**
   * Embed text and add it to the knowledge store.
   *
   * @param {string} text  document text (must not be empty)
   * @returns {Promise<void>}
   */
  async addDocument(text) {
    this._assertOpen();
    const vec = await this.#embed.embed(text);
    this.#store.add(text, vec);
  }

  /**
   * Register a callable tool.
   *
   * @param {object}   def      tool definition ({ name, description, parameters })
   * @param {Function} handler  async (args: object) => result
   */
  addTool(def, handler) {
    this._assertOpen();
    this.#registry.register(def, handler);
  }

  /**
   * Send a user message and return the assistant's final text reply.
   *
   * @param {string} sessionId  conversation identifier
   * @param {string} message    user message
   * @returns {Promise<string>}
   */
  async chat(sessionId, message) {
    this._assertOpen();

    // 1. Retrieve knowledge context.
    const context = await this._retrieveContext(message);

    // 2. Build system prompt.
    const systemPrompt = buildSystemPrompt(context);

    // 3. Append user message to session history.
    if (!this.#sessions.has(sessionId)) {
      this.#sessions.set(sessionId, []);
    }
    const history = this.#sessions.get(sessionId);
    history.push({ role: 'user', content: message });

    // 4. Agentic loop.
    const toolDefs = this.#registry.definitions();
    const msgs = buildMessages(systemPrompt, history);

    for (let i = 0; i < MAX_TOOL_ITERATIONS; i++) {
      // On the final iteration, force text mode to avoid an infinite loop in
      // case the model keeps returning tool calls (e.g. a misbehaving model or stub).
      const isLastIteration = i === MAX_TOOL_ITERATIONS - 1;
      const req = {
        messages:     msgs,
        responseMode: (toolDefs.length > 0 && !isLastIteration) ? 'tool_call' : 'text',
        tools:        toolDefs,
        toolChoice:   'auto',
      };

      const resp = await this.#chat.chat(req);

      if (resp.type === 'assistant_text') {
        history.push({ role: 'assistant', content: resp.text });
        return resp.text;
      }

      if (resp.type === 'structured_json') {
        const text = JSON.stringify(resp.json);
        history.push({ role: 'assistant', content: text });
        return text;
      }

      if (resp.type === 'tool_call') {
        if (!resp.tool_calls || resp.tool_calls.length === 0) {
          throw new Error('Agent: model returned tool_call with no tool calls');
        }

        // Add assistant message with tool call info.
        msgs.push({ role: 'assistant', content: JSON.stringify(resp.tool_calls) });

        // Execute each tool call.
        for (const tc of resp.tool_calls) {
          let resultStr;
          try {
            const result = await this.#registry.execute(tc.name, tc.arguments ?? {});
            resultStr = JSON.stringify(result);
          } catch (err) {
            resultStr = JSON.stringify({ error: err.message });
          }
          msgs.push({ role: 'tool', content: resultStr, tool_name: tc.name });
        }
        // Continue the loop with tool results in context.
        continue;
      }

      throw new Error(`Agent: unexpected response type "${resp.type}"`);
    }

    throw new Error(`Agent: exceeded ${MAX_TOOL_ITERATIONS} tool call iterations`);
  }

  /**
   * Clear the conversation history for a session.
   *
   * @param {string} sessionId
   */
  clearSession(sessionId) {
    this.#sessions.delete(sessionId);
  }

  /**
   * Release all owned resources. Safe to call multiple times.
   */
  close() {
    if (this.#closed) return;
    this.#closed = true;
    this.#chat.close();
    this.#embed.close();
    this.#store.close();
  }

  /**
   * Export a complete Agent bundle as a ZIP Blob.
   *
   * @returns {Promise<Blob>}
   */
  async export() {
    this._assertOpen();

    const docs = this.#store.all();

    const manifest = {
      version:       '1',
      created_at:    new Date().toISOString(),
      chat_model:    basename(this.#chatModelPath),
      embed_model:   basename(this.#embedModelPath),
      storage_path:  ':memory:',
      doc_count:     docs.length,
      embedding_dim: docs[0]?.embedding.length || 0,
    };

    const knowledgeJson = docs.map(d => ({ text: d.text, embedding: d.embedding }));

    const zip = zipSync({
      'manifest.json':  new TextEncoder().encode(JSON.stringify(manifest, null, 2)),
      'knowledge.json': new TextEncoder().encode(JSON.stringify(knowledgeJson)),
      'knowledge.db':   new Uint8Array(0),
    });

    return new Blob([zip], { type: 'application/zip' });
  }

  /**
   * Restore an Agent from a ZIP bundle.
   *
   * @param {Blob|File|ArrayBuffer|Uint8Array} zipData
   * @param {string} chatModelPath
   * @param {string} embedModelPath
   * @param {object} [opts={}]
   * @returns {Promise<Agent>}
   */
  static async importFrom(zipData, chatModelPath, embedModelPath, opts = {}) {
    let uint8;
    if (zipData instanceof Uint8Array) {
      uint8 = zipData;
    } else if (zipData instanceof ArrayBuffer) {
      uint8 = new Uint8Array(zipData);
    } else if (typeof Blob !== 'undefined' && zipData instanceof Blob) {
      uint8 = new Uint8Array(await zipData.arrayBuffer());
    } else {
      throw new Error('Agent.importFrom: unsupported zipData type');
    }

    const files = unzipSync(uint8);

    const manifestRaw = new TextDecoder().decode(files['manifest.json']);
    const manifest = JSON.parse(manifestRaw);
    if (manifest.version !== '1') {
      throw new Error(`Agent.importFrom: unsupported manifest version "${manifest.version}"`);
    }

    const knowledgeRaw = new TextDecoder().decode(files['knowledge.json']);
    const docs = JSON.parse(knowledgeRaw);

    const agent = await Agent.create(chatModelPath, embedModelPath, ':memory:', opts);
    agent._restoreDocs(docs);
    return agent;
  }

  /**
   * Restore documents directly into the knowledge store (used by importFrom).
   *
   * @param {Array<{text:string, embedding:number[]}>} docs
   */
  _restoreDocs(docs) {
    for (const doc of docs) {
      this.#store.add(doc.text, doc.embedding);
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Internal
  // ─────────────────────────────────────────────────────────────────────────

  async _retrieveContext(text) {
    try {
      const vec = await this.#embed.embed(text);
      return this.#store.search(vec, text, 5);
    } catch {
      return [];
    }
  }

  _assertOpen() {
    if (this.#closed) {
      throw new Error('Agent: agent is closed');
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

function buildSystemPrompt(docs) {
  if (!docs || docs.length === 0) {
    return 'You are a helpful AI assistant.';
  }
  const ctx = docs.map(d => `- ${d.text}`).join('\n');
  return `You are a helpful AI assistant.\n\nRelevant context:\n${ctx}`;
}

function buildMessages(systemPrompt, history) {
  return [
    { role: 'system', content: systemPrompt },
    ...history,
  ];
}

/** Extract the filename from a path (works for both / and \ separators). */
function basename(path) {
  if (!path) return '';
  const parts = path.replace(/\\/g, '/').split('/');
  return parts[parts.length - 1];
}
