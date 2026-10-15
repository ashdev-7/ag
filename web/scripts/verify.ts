// Checks every micro-step the stepper shows against an independent reference.
//
//   node scripts/verify.ts        (after `make wasm` in engine/)
//
// The engine runs each stepping preset one step at a time. Beside it, a
// reference written here from the textbook definitions, in double precision,
// recomputes what that step should have produced. The reference never uses
// the engine's derivative code: every derivative it needs comes from finite
// differences of its own forward rules.
//
//   forward step     the node's value equals the reference forward rule
//   seed             ∂L/∂L becomes 1
//   backward step    n → input i: the input's gradient grows by exactly this
//                    edge's share, ∂L/∂n · ∂n/∂input_i, and ∂L/∂n is already
//                    complete when n sends it (what "ready" means)
//   update step      each parameter moves by −lr · ∂L/∂w (the presets use SGD)
//   forward mode     the tangent reaching the loss equals ∂L/∂(seed element)
//   end of pass      every gradient equals the finite-difference derivative

import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createSession, loadEngine } from '../src/engine/bridge.ts';
import type { EngineSession } from '../src/engine/bridge.ts';
import { PRESETS } from '../src/graph/presets.ts';
import type { GraphSpec, NodeSpec } from '../src/graph/spec.ts';

const here = dirname(fileURLToPath(import.meta.url));
const wasmDir = join(here, '../public/engine');

// ── a reference forward pass, double precision ─────────────────────────────

interface T { shape: number[]; data: Float64Array }
const size = (s: number[]) => s.reduce((a, b) => a * b, 1);
const tensor = (shape: number[], f: (i: number) => number): T => ({ shape, data: Float64Array.from({ length: size(shape) }, (_, i) => f(i)) });

/** Index into `s` for flat output index `i` of shape `out`, numpy broadcasting. */
function bidx(out: number[], s: number[], i: number): number {
  let idx = 0, stride = 1;
  for (let d = out.length - 1, k = s.length - 1; d >= 0; --d, --k) {
    const coord = i % out[d];
    i = Math.floor(i / out[d]);
    if (k >= 0) {
      if (s[k] !== 1) idx += coord * stride;
      stride *= s[k];
    }
  }
  return idx;
}

function binary(a: T, b: T, f: (x: number, y: number) => number): T {
  const n = Math.max(a.shape.length, b.shape.length);
  const shape = Array.from({ length: n }, (_, d) => {
    const x = a.shape[a.shape.length - n + d] ?? 1, y = b.shape[b.shape.length - n + d] ?? 1;
    return x === 1 ? y : x;
  });
  return tensor(shape, (i) => f(a.data[bidx(shape, a.shape, i)], b.data[bidx(shape, b.shape, i)]));
}

function reduce(x: T, node: NodeSpec, mean: boolean): T {
  if (node.axis === undefined || node.axis === 'all') {
    const total = x.data.reduce((p, v) => p + v, 0) / (mean ? x.data.length : 1);
    return tensor(node.keepDim ? x.shape.map(() => 1) : [], () => total);
  }
  const ax = node.axis < 0 ? node.axis + x.shape.length : node.axis;
  const len = x.shape[ax], inner = size(x.shape.slice(ax + 1));
  const shape = node.keepDim ? x.shape.map((d, k) => (k === ax ? 1 : d)) : x.shape.filter((_, k) => k !== ax);
  return tensor(shape, (i) => {
    const o = Math.floor(i / inner), r = i % inner;
    let t = 0;
    for (let k = 0; k < len; ++k) t += x.data[(o * len + k) * inner + r];
    return mean ? t / len : t;
  });
}

function rows(x: T) {
  const c = x.shape[x.shape.length - 1];
  return { c, n: x.data.length / c };
}

/** The forward rule of one node, from its inputs. Leaves are given. */
function forward(node: NodeSpec, ins: T[]): T {
  const [a, b] = ins, s = node.scalar ?? 0;
  const map = (f: (v: number) => number) => tensor(a.shape, (i) => f(a.data[i]));
  switch (node.op) {
    case 'add': return binary(a, b, (x, y) => x + y);
    case 'sub': return binary(a, b, (x, y) => x - y);
    case 'mul': return binary(a, b, (x, y) => x * y);
    case 'div': return binary(a, b, (x, y) => x / y);
    case 'neg': return map((v) => -v);
    case 'pow': return map((v) => v ** s);
    case 'exp': return map(Math.exp);
    case 'log': return map(Math.log);
    case 'relu': return map((v) => Math.max(0, v));
    case 'leaky_relu': return map((v) => (v > 0 ? v : s * v));
    case 'tanh': return map(Math.tanh);
    case 'sigmoid': return map((v) => 1 / (1 + Math.exp(-v)));
    case 'detach': return map((v) => v);
    case 'reshape': return { shape: (node.shape ?? []) as number[], data: a.data.slice() };
    case 'transpose': return tensor([a.shape[1], a.shape[0]], (i) => a.data[(i % a.shape[0]) * a.shape[1] + Math.floor(i / a.shape[0])]);
    case 'matmul': {
      const [m, k] = a.shape, n = b.shape[1];
      return tensor([m, n], (i) => {
        const r = Math.floor(i / n), c = i % n;
        let t = 0;
        for (let p = 0; p < k; ++p) t += a.data[r * k + p] * b.data[p * n + c];
        return t;
      });
    }
    case 'sum': return reduce(a, node, false);
    case 'mean': return reduce(a, node, true);
    case 'softmax': {
      const { c } = rows(a);   // the presets use the last axis
      return tensor(a.shape, (i) => {
        const r = Math.floor(i / c);
        let t = 0;
        for (let k = 0; k < c; ++k) t += Math.exp(a.data[r * c + k]);
        return Math.exp(a.data[i]) / t;
      });
    }
    case 'mse_loss': return tensor([], () => a.data.reduce((p, v, i) => p + (v - b.data[i]) ** 2, 0) / a.data.length);
    case 'softmax_cross_entropy': {
      // the mean over rows of −Σ_c t_c log softmax(z)_c
      const { c, n } = rows(a);
      let total = 0;
      for (let r = 0; r < n; ++r) {
        let z = 0;
        for (let k = 0; k < c; ++k) z += Math.exp(a.data[r * c + k]);
        for (let k = 0; k < c; ++k) total -= b.data[r * c + k] * Math.log(Math.exp(a.data[r * c + k]) / z);
      }
      return tensor([], () => total / n);
    }
    default: throw new Error(`no reference rule for ${node.op}`);
  }
}

const isLeaf = (n: NodeSpec) => n.op === 'input' || n.op === 'param' || n.op === 'const';

/**
 * Run the graph from `start`, with `vals` holding every node before it.
 * `edit` may replace what one node sees as one of its inputs.
 */
function run(spec: GraphSpec, order: string[], vals: Map<string, T>, from = 0,
  edit?: { node: string; slot: number; value: T }): Map<string, T> {
  const out = new Map(vals);
  const byId = new Map(spec.nodes.map((n) => [n.id, n]));
  for (let k = from; k < order.length; ++k) {
    const n = byId.get(order[k])!;
    if (isLeaf(n)) continue;
    const ins = n.inputs.map((i, slot) => (edit && edit.node === n.id && edit.slot === slot ? edit.value : out.get(i)!));
    out.set(n.id, forward(n, ins));
  }
  return out;
}

const H = 1e-6;
const lossOf = (spec: GraphSpec, m: Map<string, T>) => m.get(spec.loss)!.data[0];

/** ∂L/∂node by central differences, the node's value nudged and everything after it recomputed. */
function gradOf(spec: GraphSpec, order: string[], vals: Map<string, T>, id: string): Float64Array {
  const v = vals.get(id)!, from = order.indexOf(id) + 1, g = new Float64Array(v.data.length);
  const byId = new Map(spec.nodes.map((n) => [n.id, n]));
  // detach and no_grad: no gradient passes through them, so nudging behind
  // them must not count. The reference stops the nudge there too
  const stop = (n: NodeSpec) => n.op === 'detach' || n.noGrad === true;
  for (let i = 0; i < v.data.length; ++i) {
    const at = (d: number) => {
      const nudged = { shape: v.shape, data: v.data.slice() };
      nudged.data[i] += d;
      const m = new Map(vals);
      m.set(id, nudged);
      for (let k = from; k < order.length; ++k) {
        const n = byId.get(order[k])!;
        if (isLeaf(n)) continue;
        const ins = n.inputs.map((x) => m.get(x)!);
        m.set(n.id, stop(n) ? vals.get(n.id)! : forward(n, ins));
      }
      return lossOf(spec, m);
    };
    g[i] = (at(H) - at(-H)) / (2 * H);
  }
  return g;
}

/** The share of ∂L/∂(input `slot` of `node`) that arrives along this one edge. */
function edgeShare(vals: Map<string, T>, node: NodeSpec, slot: number, gNode: Float64Array): Float64Array {
  const x = vals.get(node.inputs[slot])!, share = new Float64Array(x.data.length);
  for (let i = 0; i < x.data.length; ++i) {
    const out = (d: number) => {
      const nudged = { shape: x.shape, data: x.data.slice() };
      nudged.data[i] += d;
      return forward(node, node.inputs.map((id, k) => (k === slot ? nudged : vals.get(id)!))).data;
    };
    const up = out(H), down = out(-H);
    for (let j = 0; j < up.length; ++j) share[i] += gNode[j] * (up[j] - down[j]) / (2 * H);
  }
  return share;
}

// ── comparing ───────────────────────────────────────────────────────────────

let failed = 0, checked = 0;
const rel = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let w = 0;
  for (let i = 0; i < a.length; ++i) w = Math.max(w, Math.abs(a[i] - b[i]) / Math.max(1, Math.abs(b[i])));
  return a.length === b.length ? w : Infinity;
};
function expectClose(label: string, got: ArrayLike<number>, want: ArrayLike<number>, tol = 1e-4) {
  checked++;
  const e = rel(got, want);
  if (e > tol) {
    failed++;
    console.log(`  FAIL  ${label}: engine ${Array.from(got).slice(0, 4).map((v) => v.toFixed(6))} reference ${Array.from(want).slice(0, 4).map((v) => v.toFixed(6))} (rel err ${e.toExponential(1)})`);
  }
}

function leafValues(s: EngineSession, spec: GraphSpec): Map<string, T> {
  const vals = new Map<string, T>();
  for (const n of spec.nodes)
    if (isLeaf(n)) {
      const r = s.read(n.id, 'value');
      vals.set(n.id, { shape: r.shape, data: Float64Array.from(r.data) });
    }
  return vals;
}

// ── one preset ──────────────────────────────────────────────────────────────

const engine = await loadEngine(pathToFileURL(join(wasmDir, 'ag.mjs')).href, (f) => join(wasmDir, f));

for (const p of PRESETS.filter((q) => q.mode === 'step')) {
  const before = checked, failedBefore = failed;
  const s = createSession(engine, p.spec, p.config, 1);
  const byId = new Map(p.spec.nodes.map((n) => [n.id, n]));

  // reverse pass, one step at a time
  let cur = s.begin('reverse');
  const schedule = s.info.schedules.reverse;
  // the leaves are set when their forward step runs; read them at the end of the forward part
  const nForward = schedule.filter((st) => st.kind === 'forward').length;
  cur = s.advance(nForward);
  const order = schedule.filter((st) => st.kind === 'forward').map((st) => st.nodeId);
  const vals = run(p.spec, order, leafValues(s, p.spec));
  const want = new Map(p.spec.nodes.map((n) => [n.id, n.op === 'const' ? null : gradOf(p.spec, order, vals, n.id)]));

  // forward steps: replay from the start and compare each value as it appears
  s.seek(0);
  for (const st of schedule.slice(0, nForward)) {
    s.advance(1);
    expectClose(`${p.id}: forward ${st.nodeId}`, s.read(st.nodeId, 'value').data, vals.get(st.nodeId)!.data);
  }
  const running = new Map<string, Float64Array>();
  for (const st of schedule.slice(nForward)) {
    if (st.kind === 'seedGrad') {
      s.advance(1);
      expectClose(`${p.id}: seed`, s.read(st.nodeId, 'grad').data, [1]);
      running.set(st.nodeId, Float64Array.from([1]));
      continue;
    }
    const node = byId.get(st.nodeId)!, target = node.inputs[st.slot];
    // the sender's gradient is complete before it sends
    expectClose(`${p.id}: ${st.nodeId} is complete before it sends`, s.read(st.nodeId, 'grad').data, want.get(st.nodeId)!);
    const share = edgeShare(vals, node, st.slot, want.get(st.nodeId)!);
    const prev = running.get(target) ?? new Float64Array(share.length);
    const next = prev.map((v, i) => v + share[i]);
    running.set(target, next);
    s.advance(1);
    expectClose(`${p.id}: backward ${st.nodeId} → ${target}`, s.read(target, 'grad').data, next);
  }
  // end of the pass: every gradient the engine has equals the finite difference
  for (const n of p.spec.nodes) {
    const info = s.info.nodes.find((x) => x.id === n.id)!;
    if (!info.requiresGrad || n.op === 'const') continue;
    let got: ArrayLike<number>;
    try {
      got = s.read(n.id, 'grad').data;
    } catch {
      // no gradient buffer: nothing on the way to the loss reads this node
      // (here, the softmax kept for predictions), so its gradient must be 0
      got = new Float64Array(want.get(n.id)!.length);
    }
    expectClose(`${p.id}: final ∂L/∂${n.id}`, got, want.get(n.id)!);
  }

  // train step: the same pass on a fresh batch, then w ← w − lr ∂L/∂w
  const lr = p.config.optim.lr;
  if (p.config.optim.kind !== 'sgd') throw new Error(`${p.id}: verify.ts only knows SGD`);
  const before0 = s.exportParams();
  cur = s.begin('train');
  const tsched = s.info.schedules.train;
  // presets whose leaves are all inputs have no parameters, so no update steps
  const firstUpdate = tsched.findIndex((st) => st.kind === 'update');
  s.advance(firstUpdate < 0 ? tsched.length : firstUpdate);
  const params = p.spec.nodes.filter((n) => n.op === 'param');
  const grads = new Map(params.map((n) => [n.id, s.read(n.id, 'grad').data]));
  if (firstUpdate >= 0) s.advance(tsched.length - firstUpdate);
  const after = s.exportParams();
  let off = 0;
  for (const n of params) {
    const g = grads.get(n.id)!;
    const k = g.length;
    const expected = Array.from({ length: k }, (_, i) => before0[off + i] - lr * g[i]);
    expectClose(`${p.id}: update ${n.id}`, after.subarray(off, off + k), expected, 1e-6);
    off += k;
  }

  // forward mode: one pass per input number, each against the finite difference
  s.importParams(before0);
  s.begin('reverse');
  s.advance(s.info.schedules.reverse.length);
  const vals2 = run(p.spec, order, leafValues(s, p.spec));
  for (const n of p.spec.nodes.filter((x) => x.op === 'param' || (x.op === 'input' && x.requiresGrad))) {
    const g = gradOf(p.spec, order, vals2, n.id);
    for (let e = 0; e < g.length; ++e) {
      s.begin('forwardMode', { nodeId: n.id, element: e });
      s.advance(s.info.schedules.forwardMode.length);
      expectClose(`${p.id}: forward mode ∂L/∂${n.id}[${e}]`, s.read(p.spec.loss, 'tangent').data, [g[e]]);
    }
  }
  void cur;
  s.free();
  console.log(`  ${failed === failedBefore ? 'PASS' : 'FAIL'}  ${p.id}: ${checked - before} checks`);
}

console.log(failed ? `=== ${failed} of ${checked} checks failed ===` : `=== all ${checked} checks passed ===`);
process.exit(failed ? 1 : 0);
