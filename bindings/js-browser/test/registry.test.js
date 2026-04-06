/**
 * registry.test.js – unit tests for ToolRegistry (browser binding).
 */

import { ToolRegistry } from '../src/tools/registry.js';

describe('ToolRegistry', () => {

  test('register and execute tool', async () => {
    const reg = new ToolRegistry();
    reg.register(
      { name: 'add', description: 'Add numbers', parameters: {} },
      async ({ a, b }) => a + b,
    );
    const result = await reg.execute('add', { a: 2, b: 3 });
    expect(result).toBe(5);
  });

  test('execute unknown tool throws', async () => {
    const reg = new ToolRegistry();
    await expect(reg.execute('nonexistent', {})).rejects.toThrow();
  });

  test('register with empty name throws', () => {
    const reg = new ToolRegistry();
    expect(() => reg.register({ name: '' }, () => {})).toThrow();
  });

  test('register with non-function handler throws', () => {
    const reg = new ToolRegistry();
    expect(() => reg.register({ name: 'foo' }, 'not-a-function')).toThrow();
  });

  test('definitions returns tools in registration order', () => {
    const reg = new ToolRegistry();
    reg.register({ name: 'a' }, async () => {});
    reg.register({ name: 'b' }, async () => {});
    reg.register({ name: 'c' }, async () => {});
    const defs = reg.definitions();
    expect(defs.map(d => d.name)).toEqual(['a', 'b', 'c']);
  });

  test('has returns correct boolean', () => {
    const reg = new ToolRegistry();
    expect(reg.has('foo')).toBe(false);
    reg.register({ name: 'foo' }, async () => {});
    expect(reg.has('foo')).toBe(true);
  });

  test('handler error propagates', async () => {
    const reg = new ToolRegistry();
    reg.register({ name: 'fail' }, async () => { throw new Error('tool failed'); });
    await expect(reg.execute('fail', {})).rejects.toThrow('tool failed');
  });

  test('re-register updates definition', () => {
    const reg = new ToolRegistry();
    reg.register({ name: 'x', description: 'v1' }, async () => 'v1');
    reg.register({ name: 'x', description: 'v2' }, async () => 'v2');
    const defs = reg.definitions();
    expect(defs.length).toBe(1);
    expect(defs[0].description).toBe('v2');
  });
});
