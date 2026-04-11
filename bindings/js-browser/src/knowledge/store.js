/**
 * store.js – In-memory knowledge store with hybrid RRF retrieval.
 *
 * Stores text documents with their embedding vectors and supports:
 *   - Vector similarity search (cosine similarity)
 *   - Keyword / text search (simple term overlap)
 *   - Reciprocal Rank Fusion (RRF) to merge both result lists
 *
 * Usage:
 *   const store = new KnowledgeStore();
 *   store.add('The sky is blue', new Float32Array([1, 0, 0]));
 *   const docs = store.search(new Float32Array([0.9, 0.1, 0]), 'sky', 5);
 *   store.close();
 */

export class KnowledgeStore {
  /** @type {Array<{id:number, text:string, embedding:Float32Array}>} */
  #docs = [];
  #nextId = 1;
  #closed = false;

  /**
   * Add a document with its pre-computed embedding vector.
   *
   * @param {string}      text       document text (must not be empty)
   * @param {Float32Array} embedding  embedding vector (must not be empty)
   */
  add(text, embedding) {
    this._assertOpen();
    if (!text || text.length === 0) {
      throw new Error('KnowledgeStore: text must not be empty');
    }
    if (!embedding || embedding.length === 0) {
      throw new Error('KnowledgeStore: embedding must not be empty');
    }
    this.#docs.push({ id: this.#nextId++, text, embedding });
  }

  /**
   * Hybrid RRF search over stored documents.
   *
   * @param {Float32Array} queryVec   query embedding (must not be empty)
   * @param {string}       queryText  query text for keyword scoring (may be empty)
   * @param {number}       [limit=5]  max results to return
   * @returns {Array<{id:number, text:string, score:number}>}
   */
  search(queryVec, queryText = '', limit = 5) {
    this._assertOpen();
    if (!queryVec || queryVec.length === 0) {
      throw new Error('KnowledgeStore: queryVec must not be empty');
    }
    if (this.#docs.length === 0) return [];

    // 1. Vector similarity ranking (descending cosine similarity).
    const vecScored = this.#docs.map(doc => ({
      doc,
      score: cosineSimilarity(queryVec, doc.embedding),
    }));
    vecScored.sort((a, b) => b.score - a.score);
    const vecRank = new Map(vecScored.map((e, i) => [e.doc.id, i + 1]));

    // 2. Keyword / text ranking.
    const ftsRank = new Map();
    if (queryText) {
      const terms = tokenize(queryText);
      const ftsScored = this.#docs.map(doc => ({
        doc,
        score: termOverlap(terms, tokenize(doc.text)),
      })).filter(e => e.score > 0);
      ftsScored.sort((a, b) => b.score - a.score);
      ftsScored.forEach((e, i) => ftsRank.set(e.doc.id, i + 1));
    }

    // 3. RRF fusion.
    const K = 60;
    const entries = this.#docs.map(doc => {
      const vr = vecRank.get(doc.id) ?? Infinity;
      const fr = ftsRank.get(doc.id) ?? Infinity;
      const rrf =
        (vr < Infinity ? 1 / (K + vr) : 0) +
        (fr < Infinity ? 1 / (K + fr) : 0);
      return { id: doc.id, text: doc.text, score: rrf };
    });

    entries.sort((a, b) => b.score - a.score);
    return entries.slice(0, limit);
  }

  /**
   * Returns all documents in insertion order.
   *
   * @returns {Array<{text:string, embedding:number[]}>}
   */
  all() {
    this._assertOpen();
    return this.#docs.map(d => ({ text: d.text, embedding: [...d.embedding] }));
  }

  /** Release resources (no-op for in-memory store, provided for API symmetry). */
  close() {
    this.#closed = true;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Internal
  // ─────────────────────────────────────────────────────────────────────────

  _assertOpen() {
    if (this.#closed) {
      throw new Error('KnowledgeStore: store is closed');
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Math helpers
// ─────────────────────────────────────────────────────────────────────────────

function cosineSimilarity(a, b) {
  const n = Math.min(a.length, b.length);
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < n; i++) {
    dot += a[i] * b[i];
    na  += a[i] * a[i];
    nb  += b[i] * b[i];
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function tokenize(text) {
  return text.toLowerCase().split(/\W+/).filter(t => t.length > 0);
}

function termOverlap(queryTerms, docTerms) {
  const docSet = new Set(docTerms);
  return queryTerms.filter(t => docSet.has(t)).length;
}
