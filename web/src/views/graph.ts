// What the graph canvas and the stepper need beyond the engine: layout,
// which nodes and edges a point in the schedule lights up, and the local
// derivative of each op written out for the chain-rule readout.

import dagre from '@dagrejs/dagre';
import type { GraphSpec, NodeSpec, OpName } from '../graph/spec.ts';
import type { Step } from '../engine/protocol.ts';

// ── layout ──────────────────────────────────────────────────────────────────

export interface LaidOutNode { id: string; x: number; y: number; width: number; height: number }
export interface LaidOutEdge { id: string; from: string; to: string; slot: number }

/** Left to right, inputs first. dagre places nodes to keep edges short and uncrossed. */
export function layoutGraph(spec: GraphSpec, size: (n: NodeSpec) => { width: number; height: number }) {
  const g = new dagre.graphlib.Graph();
  const big = spec.nodes.length > 60;
  g.setGraph({ rankdir: 'LR', nodesep: big ? 8 : 24, ranksep: big ? 36 : 60, marginx: 10, marginy: 10 });
  g.setDefaultEdgeLabel(() => ({}));
  spec.nodes.forEach((n) => g.setNode(n.id, size(n)));
  const edges: LaidOutEdge[] = [];
  spec.nodes.forEach((n) =>
    n.inputs.forEach((i, slot) => {
      g.setEdge(i, n.id);
      edges.push({ id: `${i}->${n.id}:${slot}`, from: i, to: n.id, slot });
    }),
  );
  dagre.layout(g);
  const nodes: LaidOutNode[] = spec.nodes.map((n) => {
    const p = g.node(n.id);
    return { id: n.id, x: p.x - p.width / 2, y: p.y - p.height / 2, width: p.width, height: p.height };
  });
  return { nodes, edges };
}

/**
 * Nodes where a gradient stops: they need none themselves, but an input does
 * (a detach, or a no_grad node). Read from the engine's requiresGrad flags, so
 * it works the same for the tensor graph and the single-number graph.
 */
export function gradientStops(spec: GraphSpec, requiresGrad: Record<string, boolean>): Set<string> {
  const stops = new Set<string>();
  for (const n of spec.nodes)
    if (!requiresGrad[n.id] && n.inputs.some((i) => requiresGrad[i])) stops.add(n.id);
  return stops;
}

// ── highlighting ────────────────────────────────────────────────────────────

export interface Highlight {
  /** The step that just ran, if any. */
  step?: Step;
  active?: string;
  /** For a vjp step: the edge the gradient is flowing back along. */
  activeEdge?: { from: string; to: string; slot: number };
  /** Nodes whose value has been computed in this pass. */
  valueDone: Set<string>;
  /** Nodes that have received at least one gradient (or tangent) contribution. */
  gradDone: Set<string>;
  direction: 'idle' | 'forward' | 'backward' | 'update' | 'tangent';
}

/** `index` is the cursor: steps [0, index) have run. */
export function highlightAt(schedule: Step[], index: number, spec: GraphSpec, seedNode = ''): Highlight {
  const byId = new Map(spec.nodes.map((n) => [n.id, n]));
  const h: Highlight = { valueDone: new Set(), gradDone: new Set(), direction: 'idle' };
  for (let k = 0; k < Math.min(index, schedule.length); ++k) {
    const st = schedule[k];
    if (st.kind === 'forward') h.valueDone.add(st.nodeId);
    else if (st.kind === 'seedGrad') h.gradDone.add(st.nodeId);
    else if (st.kind === 'vjp') h.gradDone.add(byId.get(st.nodeId)?.inputs[st.slot] ?? '');
    else if (st.kind === 'seedTangent') h.gradDone.add(seedNode);
    else if (st.kind === 'jvp') h.gradDone.add(st.nodeId);
  }
  if (index > 0 && index <= schedule.length) {
    const st = schedule[index - 1];
    h.step = st;
    h.active = st.kind === 'seedTangent' ? seedNode : st.nodeId;
    h.direction = st.kind === 'forward' ? 'forward' : st.kind === 'update' ? 'update'
      : st.kind === 'jvp' || st.kind === 'seedTangent' ? 'tangent' : 'backward';
    if (st.kind === 'vjp') {
      const from = byId.get(st.nodeId)?.inputs[st.slot];
      if (from) h.activeEdge = { from, to: st.nodeId, slot: st.slot };
    }
  }
  return h;
}

// ── local derivatives ───────────────────────────────────────────────────────

/** ∂out/∂input[slot], in words. `ins` are the input names. */
export function localDerivative(op: OpName, slot: number, ins: string[], scalar = 0): string {
  const [a, b] = ins;
  switch (op) {
    case 'add': return '1';
    case 'sub': return slot === 0 ? '1' : '−1';
    case 'mul': return slot === 0 ? b : a;
    case 'div': return slot === 0 ? `1 / ${b}` : `−${a} / ${b}²`;
    case 'neg': return '−1';
    case 'pow': return `${scalar} · ${a}^${fmtExp(scalar - 1)}`;
    case 'exp': return 'exp(x), the output itself';
    case 'log': return `1 / ${a}`;
    case 'matmul': return slot === 0 ? `grad @ ${b}ᵀ` : `${a}ᵀ @ grad`;
    case 'sum': return 'the gradient copied back to every summed element';
    case 'mean': return 'the gradient copied back, divided by the count';
    case 'reshape': return 'the gradient reshaped back';
    case 'transpose': return 'the gradient transposed back';
    case 'relu': return `1 where ${a} > 0, else 0`;
    case 'leaky_relu': return `1 where ${a} > 0, else ${scalar}`;
    case 'tanh': return '1 − out²';
    case 'sigmoid': return 'out · (1 − out)';
    case 'softmax': return 'out ⊙ (grad − Σ grad ⊙ out)';
    case 'mse_loss': return slot === 0 ? `2 (${a} − ${b}) / N` : `−2 (${a} − ${b}) / N`;
    case 'softmax_cross_entropy': return slot === 0 ? `(softmax(${a}) − ${b}) / rows` : `(logsumexp(${a}) − ${a}) / rows`;
    case 'detach': return '0: the gradient stops here';
    default: return '';
  }
}

const fmtExp = (n: number) => (Number.isInteger(n) ? String(n) : n.toFixed(2));

/**
 * The same derivative as a number, for scalar nodes: `x` holds the input
 * values, `out` the output. Undefined for ops that are not elementwise.
 */
export function localDerivativeValue(op: OpName, slot: number, x: number[], out: number, scalar = 0): number | undefined {
  const [a, b] = x;
  switch (op) {
    case 'add': return 1;
    case 'sub': return slot === 0 ? 1 : -1;
    case 'mul': return slot === 0 ? b : a;
    case 'div': return slot === 0 ? 1 / b : -a / (b * b);
    case 'neg': return -1;
    case 'pow': return scalar * a ** (scalar - 1);
    case 'exp': return out;
    case 'log': return 1 / a;
    case 'relu': return a > 0 ? 1 : 0;
    case 'leaky_relu': return a > 0 ? 1 : scalar;
    case 'tanh': return 1 - out * out;
    case 'sigmoid': return out * (1 - out);
    case 'mse_loss': return (slot === 0 ? 2 : -2) * (a - b);
    default: return undefined;
  }
}

export const OP_SYMBOL: Partial<Record<OpName, string>> = {
  add: '+', sub: '−', mul: '×', div: '÷', neg: '−x', pow: 'xⁿ', exp: 'exp', log: 'log', matmul: '@',
  sum: 'Σ', mean: 'mean', reshape: 'reshape', transpose: 'ᵀ', relu: 'relu', leaky_relu: 'lrelu', tanh: 'tanh',
  sigmoid: 'σ', softmax: 'softmax', mse_loss: 'mse', softmax_cross_entropy: 'cross-entropy',
  input: 'input', param: 'param', const: 'const', detach: 'detach',
};

// ── URL ─────────────────────────────────────────────────────────────────────
// the shareable part of the state. unknown or malformed fields are dropped;
// parsing never throws

export interface UrlState {
  preset?: string;
  seed?: number;
  pass?: string;
  i?: number;
  zoom?: 'tensor' | 'scalar';
  opt?: string;
  lr?: number;
  cmp?: string;
}

export function toUrlHash(s: UrlState): string {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(s)) if (v !== undefined && v !== '') p.set(k, String(v));
  return '#' + p.toString();
}

export function fromUrlHash(hash: string): UrlState {
  const out: UrlState = {};
  let p: URLSearchParams;
  try {
    p = new URLSearchParams(hash.replace(/^#/, ''));
  } catch {
    return out;
  }
  const word = (k: string) => {
    const v = p.get(k);
    return v && /^[\w-]{1,40}$/.test(v) ? v : undefined;
  };
  const num = (k: string) => {
    const v = Number(p.get(k));
    return p.has(k) && Number.isFinite(v) ? v : undefined;
  };
  out.preset = word('preset');
  out.seed = num('seed');
  out.pass = word('pass');
  out.i = num('i');
  const z = word('zoom');
  if (z === 'tensor' || z === 'scalar') out.zoom = z;
  out.opt = word('opt');
  out.lr = num('lr');
  out.cmp = word('cmp');
  return out;
}
