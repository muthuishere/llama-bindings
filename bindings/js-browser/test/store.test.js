/**
 * store.test.js – unit tests for KnowledgeStore (browser binding).
 */

import { KnowledgeStore } from '../src/knowledge/store.js';

describe('KnowledgeStore', () => {

  test('add and search returns relevant document', () => {
    const store = new KnowledgeStore();
    store.add('the sky is blue', new Float32Array([1, 0]));
    store.add('the grass is green', new Float32Array([0, 1]));

    const results = store.search(new Float32Array([1, 0]), 'sky', 5);
    expect(results.length).toBeGreaterThan(0);
    expect(results[0].text).toBe('the sky is blue');
    store.close();
  });

  test('search empty store returns empty array', () => {
    const store = new KnowledgeStore();
    const results = store.search(new Float32Array([1, 0]), '', 5);
    expect(results).toEqual([]);
    store.close();
  });

  test('add empty text throws', () => {
    const store = new KnowledgeStore();
    expect(() => store.add('', new Float32Array([1]))).toThrow();
    store.close();
  });

  test('add empty embedding throws', () => {
    const store = new KnowledgeStore();
    expect(() => store.add('text', new Float32Array([]))).toThrow();
    store.close();
  });

  test('search with empty vector throws', () => {
    const store = new KnowledgeStore();
    expect(() => store.search(new Float32Array([]), '', 5)).toThrow();
    store.close();
  });

  test('vector-only search ranks by cosine similarity', () => {
    const store = new KnowledgeStore();
    store.add('A', new Float32Array([1, 0, 0]));
    store.add('B', new Float32Array([0, 1, 0]));
    store.add('C', new Float32Array([0, 0, 1]));

    const results = store.search(new Float32Array([0.9, 0.1, 0]), '', 3);
    expect(results[0].text).toBe('A');
    store.close();
  });

  test('close prevents further operations', () => {
    const store = new KnowledgeStore();
    store.close();
    expect(() => store.add('text', new Float32Array([1]))).toThrow();
  });

  test('limit is respected', () => {
    const store = new KnowledgeStore();
    for (let i = 0; i < 10; i++) {
      store.add(`doc ${i}`, new Float32Array([Math.random(), Math.random()]));
    }
    const results = store.search(new Float32Array([1, 0]), '', 3);
    expect(results.length).toBeLessThanOrEqual(3);
    store.close();
  });
});
