// GraphSpec — the saved form of a network. This is what presets are written
// in, what goes in a shared link, and what the builder edits in Phase 2.
// It is JSON; the engine never parses it. `toBuildPlan` turns it into the
// ordered addNode calls the engine accepts.

export const SPEC_VERSION = 1;

/** Must match `name` in the engine's op table (engine/src/ops.cpp). */
export type OpName =
  | 'input' | 'param' | 'const'
  | 'add' | 'sub' | 'mul' | 'div' | 'neg' | 'pow' | 'exp' | 'log'
  | 'matmul' | 'sum' | 'mean' | 'reshape' | 'transpose'
  | 'relu' | 'leaky_relu' | 'tanh' | 'sigmoid' | 'softmax'
  | 'mse_loss' | 'softmax_cross_entropy';

const ARITY: Record<OpName, [number, number]> = {
  input: [0, 0], param: [0, 0], const: [0, 0],
  add: [2, 2], sub: [2, 2], mul: [2, 2], div: [2, 2], neg: [1, 1], pow: [1, 1], exp: [1, 1], log: [1, 1],
  matmul: [2, 2], sum: [1, 1], mean: [1, 1], reshape: [1, 1], transpose: [1, 1],
  relu: [1, 1], leaky_relu: [1, 1], tanh: [1, 1], sigmoid: [1, 1], softmax: [1, 1],
  mse_loss: [2, 2], softmax_cross_entropy: [2, 2],
};

/** In engine order (enum Init in engine/include/ag/graph.hpp). */
export const INIT_NAMES = [
  'zeros', 'constant', 'uniform', 'normal',
  'xavier_uniform', 'xavier_normal', 'he_uniform', 'he_normal',
] as const;
export type InitName = (typeof INIT_NAMES)[number];

/** 'B' stands for the batch dimension. */
export type Dim = number | 'B';

export interface NodeSpec {
  /** Unique within the graph. Stable across edits, unlike engine node numbers. */
  id: string;
  op: OpName;
  /** Ids of the nodes feeding this one, in argument order. */
  inputs: string[];
  /** Label on the canvas. Defaults to `id`. */
  name?: string;
  /** Group for charts, e.g. "layer 1". */
  module?: string;
  shape?: Dim[];
  axis?: number | 'all';
  keepDim?: boolean;
  scalar?: number;
  role?: 'features' | 'targets';
  init?: InitName;
  initScale?: number;
  requiresGrad?: boolean;
}

export interface GraphSpec {
  version: typeof SPEC_VERSION;
  nodes: NodeSpec[];
  /** Id of the scalar node the backward pass starts from. */
  loss: string;
  /** Ids of prediction nodes, used for the decision boundary. */
  outputs: string[];
}

export interface SpecIssue {
  code: 'unknown-op' | 'duplicate-id' | 'missing-input' | 'cycle' | 'bad-field' | 'version';
  message: string;
  nodeId?: string;
}

/** One addNode call. `inputs` are engine node numbers, already resolved. */
export interface BuildStep {
  specId: string;
  op: number;
  inputs: number[];
  spec: NodeSpec;
}

export interface BuildPlan {
  steps: BuildStep[];
  /** spec id -> engine node number, and back. */
  toEngine: Map<string, number>;
  toSpec: string[];
  loss: number;
  outputs: number[];
}

const isObj = (x: unknown): x is Record<string, unknown> => typeof x === 'object' && x !== null && !Array.isArray(x);

/** Structural checks that need no engine: ids, references, cycles, fields. Never throws. */
export function validateSpec(spec: unknown): SpecIssue[] {
  const issues: SpecIssue[] = [];
  if (!isObj(spec)) return [{ code: 'bad-field', message: 'the graph is not an object' }];
  if (spec.version !== SPEC_VERSION) issues.push({ code: 'version', message: `unsupported version ${String(spec.version)}` });
  if (!Array.isArray(spec.nodes) || spec.nodes.length === 0) return [...issues, { code: 'bad-field', message: 'the graph has no nodes' }];

  const ids = new Set<string>();
  for (const raw of spec.nodes as unknown[]) {
    if (!isObj(raw) || typeof raw.id !== 'string' || raw.id === '') {
      issues.push({ code: 'bad-field', message: 'a node has no id' });
      continue;
    }
    const id = raw.id;
    if (ids.has(id)) issues.push({ code: 'duplicate-id', message: `two nodes are called "${id}"`, nodeId: id });
    ids.add(id);
    const op = raw.op as OpName;
    if (typeof raw.op !== 'string' || !Object.prototype.hasOwnProperty.call(ARITY, op)) {
      issues.push({ code: 'unknown-op', message: `unknown op "${String(raw.op)}"`, nodeId: id });
      continue;
    }
    const inputs = raw.inputs;
    if (!Array.isArray(inputs) || inputs.some((i) => typeof i !== 'string')) {
      issues.push({ code: 'bad-field', message: 'inputs must be a list of node ids', nodeId: id });
      continue;
    }
    const [lo, hi] = ARITY[op];
    if (inputs.length < lo || inputs.length > hi)
      issues.push({ code: 'bad-field', message: `${op} takes ${lo === hi ? lo : `${lo} to ${hi}`} inputs, got ${inputs.length}`, nodeId: id });
    if (raw.shape !== undefined) {
      const ok = Array.isArray(raw.shape) && raw.shape.length <= 4 &&
        raw.shape.every((d) => d === 'B' || (Number.isInteger(d) && (d as number) > 0 && (d as number) <= 65536));
      if (!ok) issues.push({ code: 'bad-field', message: 'shape must be up to 4 positive sizes or "B"', nodeId: id });
    }
    if (raw.init !== undefined && !(INIT_NAMES as readonly unknown[]).includes(raw.init))
      issues.push({ code: 'bad-field', message: `unknown init "${String(raw.init)}"`, nodeId: id });
    for (const f of ['scalar', 'initScale'] as const)
      if (raw[f] !== undefined && !Number.isFinite(raw[f])) issues.push({ code: 'bad-field', message: `${f} must be a number`, nodeId: id });
  }
  for (const raw of spec.nodes as unknown[]) {
    if (!isObj(raw) || !Array.isArray(raw.inputs)) continue;
    for (const i of raw.inputs)
      if (typeof i === 'string' && !ids.has(i))
        issues.push({ code: 'missing-input', message: `input "${i}" does not exist`, nodeId: String(raw.id) });
  }
  if (typeof spec.loss !== 'string' || (spec.loss !== '' && !ids.has(spec.loss)))
    issues.push({ code: 'bad-field', message: 'the loss must name a node' });
  if (!Array.isArray(spec.outputs) || spec.outputs.some((o) => typeof o !== 'string' || !ids.has(o)))
    issues.push({ code: 'bad-field', message: 'outputs must name nodes' });
  if (issues.length === 0) {
    const cyc = findCycle(spec as unknown as GraphSpec);
    if (cyc) issues.push({ code: 'cycle', message: `these nodes form a loop: ${cyc.join(' → ')}`, nodeId: cyc[0] });
  }
  return issues;
}

function findCycle(spec: GraphSpec): string[] | null {
  const byId = new Map(spec.nodes.map((n) => [n.id, n]));
  const state = new Map<string, 1 | 2>();
  const path: string[] = [];
  const visit = (id: string): string[] | null => {
    if (state.get(id) === 2) return null;
    if (state.get(id) === 1) return [...path.slice(path.indexOf(id)), id];
    state.set(id, 1);
    path.push(id);
    for (const i of byId.get(id)!.inputs) {
      const c = visit(i);
      if (c) return c;
    }
    path.pop();
    state.set(id, 2);
    return null;
  };
  for (const n of spec.nodes) {
    const c = visit(n.id);
    if (c) return c;
  }
  return null;
}

/**
 * Topologically sort and number the nodes: inputs before the node, otherwise
 * in the order written. Throws only if `validateSpec` would have reported an
 * issue; call that first and show its messages.
 */
export function toBuildPlan(spec: GraphSpec, opNumbers: Map<OpName, number>): BuildPlan {
  const byId = new Map(spec.nodes.map((n) => [n.id, n]));
  const toEngine = new Map<string, number>();
  const toSpec: string[] = [];
  const steps: BuildStep[] = [];
  const place = (id: string, depth: number) => {
    if (toEngine.has(id)) return;
    if (depth > spec.nodes.length) throw new Error('cycle');
    const n = byId.get(id);
    if (!n) throw new Error(`missing node ${id}`);
    n.inputs.forEach((i) => place(i, depth + 1));
    const op = opNumbers.get(n.op);
    if (op === undefined) throw new Error(`unknown op ${n.op}`);
    toEngine.set(id, toSpec.length);
    toSpec.push(id);
    steps.push({ specId: id, op, inputs: n.inputs.map((i) => toEngine.get(i)!), spec: n });
  };
  spec.nodes.forEach((n) => place(n.id, 0));
  return {
    steps, toEngine, toSpec,
    loss: spec.loss ? toEngine.get(spec.loss)! : -1,
    outputs: spec.outputs.map((o) => toEngine.get(o)!),
  };
}

function toBase64Url(bytes: Uint8Array): string {
  let s = '';
  bytes.forEach((b) => (s += String.fromCharCode(b)));
  return btoa(s).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function fromBase64Url(text: string): Uint8Array {
  const s = atob(text.replace(/-/g, '+').replace(/_/g, '/'));
  return Uint8Array.from(s, (c) => c.charCodeAt(0));
}

/** Compact, URL-safe encoding for shared links. */
export function encodeSpec(spec: GraphSpec): string {
  return toBase64Url(new TextEncoder().encode(JSON.stringify(spec)));
}

/** Never throws: anything malformed comes back as issues. */
export function decodeSpec(encoded: string): GraphSpec | SpecIssue[] {
  let parsed: unknown;
  try {
    parsed = JSON.parse(new TextDecoder().decode(fromBase64Url(encoded)));
  } catch {
    return [{ code: 'bad-field', message: 'the link does not contain a readable graph' }];
  }
  const issues = validateSpec(parsed);
  return issues.length ? issues : (parsed as GraphSpec);
}
