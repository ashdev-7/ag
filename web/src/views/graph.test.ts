// The numbers the inspector prints, checked against the definitions they
// claim to follow.

import { describe, expect, it } from 'vitest';
import { OP_NAMES, SOURCES } from '../graph/spec.ts';
import type { OpName } from '../graph/spec.ts';
import { divergingScale, maxAbs, surfaceScale } from './draw.ts';
import { BLUE, MID, ORANGE, SURFACE_HIGH } from './theme.ts';
import { highlightAt, localDerivative, localDerivativeValue, OP_SYMBOL } from './graph.ts';

// the forward rule of every op that has a single-number derivative
const FORWARD: Partial<Record<OpName, (x: number[], s: number) => number>> = {
  add: ([a, b]) => a + b,
  sub: ([a, b]) => a - b,
  mul: ([a, b]) => a * b,
  div: ([a, b]) => a / b,
  neg: ([a]) => -a,
  pow: ([a], s) => a ** s,
  exp: ([a]) => Math.exp(a),
  log: ([a]) => Math.log(a),
  relu: ([a]) => Math.max(0, a),
  leaky_relu: ([a], s) => (a > 0 ? a : s * a),
  tanh: ([a]) => Math.tanh(a),
  sigmoid: ([a]) => 1 / (1 + Math.exp(-a)),
  mse_loss: ([a, b]) => (a - b) ** 2,
};

describe('localDerivativeValue', () => {
  // points away from relu's kink and inside log's domain
  const points = [[0.7, 1.3], [1.9, -0.4], [-0.6, 2.2], [0.25, 0.8]];
  const scalars: Partial<Record<OpName, number>> = { pow: 3, leaky_relu: 0.01 };

  for (const [op, f] of Object.entries(FORWARD) as [OpName, (x: number[], s: number) => number][]) {
    it(`${op} matches finite differences`, () => {
      const s = scalars[op] ?? 0;
      const arity = ['add', 'sub', 'mul', 'div', 'mse_loss'].includes(op) ? 2 : 1;
      for (const p of points) {
        const x = op === 'log' ? p.map(Math.abs) : p;
        for (let slot = 0; slot < arity; ++slot) {
          const h = 1e-6;
          const up = [...x], down = [...x];
          up[slot] += h;
          down[slot] -= h;
          const numeric = (f(up, s) - f(down, s)) / (2 * h);
          const shown = localDerivativeValue(op, slot, x, f(x, s), s);
          expect(shown, `${op} slot ${slot} at ${x}`).toBeCloseTo(numeric, 5);
        }
      }
    });
  }

  it('is undefined for ops that are not elementwise', () => {
    for (const op of ['matmul', 'sum', 'softmax'] as OpName[]) expect(localDerivativeValue(op, 0, [1, 2], 3)).toBeUndefined();
  });
});

describe('every op is explained', () => {
  it('has a symbol, and a written local derivative unless it is a source', () => {
    for (const op of OP_NAMES) {
      expect(OP_SYMBOL[op], op).toBeTruthy();
      if (!SOURCES.includes(op)) expect(localDerivative(op, 0, ['a', 'b']), op).not.toBe('');
    }
  });
});

describe('colour scales', () => {
  it('diverging: the page’s dark at zero, blue and orange at ±max, magenta for nan', () => {
    const s = divergingScale(2);
    expect(s(0)).toEqual(MID);
    expect(s(-2)).toEqual(BLUE);
    expect(s(2)).toEqual(ORANGE);
    expect(s(5)).toEqual(s(2));   // clamped
    expect(s(NaN)).toEqual([200, 0, 200]);
    expect(divergingScale(0)(0)).toEqual(MID);   // an all-zero tensor still draws
  });

  it('surface: darkest at the minimum, lighter as the loss rises', () => {
    const s = surfaceScale(0, 100);
    const light = (v: number) => s(v).reduce((a, b) => a + b, 0);
    expect(light(0)).toBeLessThan(light(1));
    expect(light(1)).toBeLessThan(light(100));
    expect(s(Infinity)).toEqual(SURFACE_HIGH);
  });

  it('maxAbs skips non-finite values', () => {
    expect(maxAbs([1, -3, NaN, Infinity, 2])).toBe(3);
    expect(maxAbs([])).toBe(0);
  });
});

describe('highlightAt', () => {
  const spec = {
    version: 1 as const, loss: 'd', outputs: [],
    nodes: [
      { id: 'a', op: 'input' as const, inputs: [] }, { id: 'b', op: 'input' as const, inputs: [] },
      { id: 'c', op: 'mul' as const, inputs: ['a', 'b'] }, { id: 'd', op: 'add' as const, inputs: ['c', 'a'] },
    ],
  };
  const sched = [
    ...['a', 'b', 'c', 'd'].map((nodeId) => ({ kind: 'forward' as const, nodeId, slot: 0 })),
    { kind: 'seedGrad' as const, nodeId: 'd', slot: 0 },
    { kind: 'vjp' as const, nodeId: 'd', slot: 1 },
  ];

  it('marks values, gradients and the edge the gradient travels', () => {
    const h = highlightAt(sched, 6, spec);
    expect([...h.valueDone]).toEqual(['a', 'b', 'c', 'd']);
    expect([...h.gradDone].sort()).toEqual(['a', 'd']);   // the vjp of d's input 1 reached a
    expect(h.direction).toBe('backward');
    expect(h.activeEdge).toEqual({ from: 'a', to: 'd', slot: 1 });
  });

  it('is idle before the first step and clamps past the end', () => {
    expect(highlightAt(sched, 0, spec).direction).toBe('idle');
    expect(highlightAt(sched, 99, spec).valueDone.size).toBe(4);
  });
});
