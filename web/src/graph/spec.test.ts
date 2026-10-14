import { describe, expect, it } from 'vitest';
import { decodeSpec, encodeSpec, toBuildPlan, validateSpec } from './spec.ts';
import type { GraphSpec, OpName } from './spec.ts';
import { PRESETS } from './presets.ts';
import { fromUrlHash, gradientStops, highlightAt, toUrlHash } from '../views/graph.ts';

const ops = new Map<OpName, number>(['input', 'param', 'const', 'add', 'sub', 'mul', 'div', 'neg', 'pow', 'exp', 'log', 'matmul',
  'sum', 'mean', 'reshape', 'transpose', 'relu', 'leaky_relu', 'tanh', 'sigmoid', 'softmax', 'mse_loss',
  'softmax_cross_entropy', 'detach'].map((n, i) => [n as OpName, i]));

const example: GraphSpec = {
  version: 1,
  nodes: [
    { id: 'd', op: 'add', inputs: ['c', 'a'] },   // written before its inputs on purpose
    { id: 'a', op: 'input', inputs: [], shape: [], requiresGrad: true },
    { id: 'b', op: 'input', inputs: [], shape: [], requiresGrad: true },
    { id: 'c', op: 'mul', inputs: ['a', 'b'] },
  ],
  loss: 'd',
  outputs: [],
};

describe('validateSpec', () => {
  it('accepts every preset', () => {
    for (const p of PRESETS) expect(validateSpec(p.spec), p.id).toEqual([]);
  });

  it('never throws on garbage', () => {
    const junk: unknown[] = [null, 1, 'x', [], {}, { version: 1 }, { version: 1, nodes: [null, 3, { id: 5 }] },
      { version: 1, nodes: [{ id: 'a', op: 'add', inputs: 'nope' }], loss: 7, outputs: {} },
      { version: 1, nodes: [{ id: 'a', op: 'toString', inputs: [] }], loss: 'a', outputs: [] },
      { version: 1, nodes: [{ id: 'a', op: 'param', inputs: [], shape: [-1, 'C', 1.5] }], loss: '', outputs: [] }];
    for (const j of junk) expect(validateSpec(j).length).toBeGreaterThan(0);
  });

  it('names the node in each issue', () => {
    const bad = { ...example, nodes: [...example.nodes, { id: 'e', op: 'mul', inputs: ['a'] }] };
    expect(validateSpec(bad)).toEqual([expect.objectContaining({ nodeId: 'e', code: 'bad-field' })]);
    const missing = { ...example, nodes: [...example.nodes, { id: 'e', op: 'neg', inputs: ['zz'] }] };
    expect(validateSpec(missing)[0]).toMatchObject({ code: 'missing-input', nodeId: 'e' });
  });

  it('checks no_grad: a boolean, and only on computed nodes', () => {
    const on = (id: string, noGrad: unknown) => ({ ...example, nodes: example.nodes.map((n) => (n.id === id ? { ...n, noGrad } : n)) });
    expect(validateSpec(on('c', true))).toEqual([]);
    expect(validateSpec(on('a', true))).toEqual([expect.objectContaining({ nodeId: 'a', code: 'bad-field' })]);
    expect(validateSpec(on('c', 'yes'))).toEqual([expect.objectContaining({ nodeId: 'c', code: 'bad-field' })]);
  });

  it('finds a cycle', () => {
    const loop = { version: 1, nodes: [
      { id: 'a', op: 'neg', inputs: ['b'] }, { id: 'b', op: 'neg', inputs: ['a'] },
    ], loss: 'a', outputs: [] };
    expect(validateSpec(loop)[0].code).toBe('cycle');
  });
});

describe('toBuildPlan', () => {
  it('puts inputs before the nodes that use them', () => {
    const plan = toBuildPlan(example, ops);
    expect(plan.toSpec).toEqual(['a', 'b', 'c', 'd']);
    plan.steps.forEach((s, i) => s.inputs.forEach((n) => expect(n).toBeLessThan(i)));
    expect(plan.toSpec[plan.loss]).toBe('d');
  });
});

describe('links', () => {
  it('round-trips a spec and rejects a broken one', () => {
    expect(decodeSpec(encodeSpec(example))).toEqual(example);
    expect(Array.isArray(decodeSpec('%%%not base64'))).toBe(true);
    expect(Array.isArray(decodeSpec(btoa('{"version":1}')))).toBe(true);
  });

  it('reads only well-formed URL fields', () => {
    expect(fromUrlHash('#preset=moons&seed=7&zoom=scalar&i=3')).toMatchObject({ preset: 'moons', seed: 7, zoom: 'scalar', i: 3 });
    const bad = fromUrlHash('#preset=<script>&seed=NaN&zoom=module');
    expect([bad.preset, bad.seed, bad.zoom]).toEqual([undefined, undefined, undefined]);
    for (const h of ['', '#', '#%', '#&&&=', '#seed=1e999']) expect(() => fromUrlHash(h)).not.toThrow();
    expect(fromUrlHash(toUrlHash({ preset: 'xor', seed: 3, opt: 'adam', lr: 0.01 }))).toMatchObject({ preset: 'xor', seed: 3, opt: 'adam', lr: 0.01 });
  });
});

describe('highlightAt', () => {
  it('lights the edge a gradient flows along', () => {
    const schedule = [
      { kind: 'forward', nodeId: 'a', slot: 0 }, { kind: 'forward', nodeId: 'b', slot: 0 },
      { kind: 'forward', nodeId: 'c', slot: 0 }, { kind: 'forward', nodeId: 'd', slot: 0 },
      { kind: 'seedGrad', nodeId: 'd', slot: 0 }, { kind: 'vjp', nodeId: 'd', slot: 1 },
    ] as const;
    const h = highlightAt([...schedule], 6, example);
    expect(h.direction).toBe('backward');
    expect(h.activeEdge).toEqual({ from: 'a', to: 'd', slot: 1 });
    expect([...h.gradDone].sort()).toEqual(['a', 'd']);
    expect(highlightAt([...schedule], 0, example).valueDone.size).toBe(0);
  });
});

describe('gradientStops', () => {
  it('marks a node that needs no gradient but has an input that does', () => {
    const spec = PRESETS.find((p) => p.id === 'detach')!.spec;
    // what the engine reports for this preset: detach(a) needs no gradient
    const rg = { a: true, b: true, sa: false, c: true, d: true };
    expect([...gradientStops(spec, rg)]).toEqual(['sa']);
    // plain constants feeding plain constants stop nothing
    expect(gradientStops(spec, {}).size).toBe(0);
  });
});
