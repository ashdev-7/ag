// bridge — the only code that touches the WebAssembly module. Turns
// GraphSpecs into addNode calls, packs the C structs, copies tensors out, and
// turns status codes into EngineErrors that name spec ids, not engine numbers.
//
// Runs in the worker, and under node for tests (engine/tests/wasm_moons.mjs).
// Struct offsets are pinned by sizeof checks in engine/tests/test_skeleton.cpp.

import { toBuildPlan, validateSpec, INIT_NAMES } from '../graph/spec.ts';
import type { GraphSpec, NodeSpec, OpName } from '../graph/spec.ts';
import { ENGINE_CODES, PASS_KINDS, STEP_KINDS, WHICH } from './protocol.ts';
import type {
  Cursor, Dataset, EngineError, Grid, NodeInfo, OptimConfig, PassKind, ScalarView,
  SessionConfig, SessionInfo, Stats, Step, TangentSeed, TrainResult, Which,
} from './protocol.ts';

export const ABI_VERSION = 1;
const ALL_AXES = -128;
const BATCH = -1;

type Fn = (...args: number[]) => number;

export interface AgModule {
  HEAPU8: Uint8Array;
  HEAP32: Int32Array;
  HEAPU32: Uint32Array;
  HEAPF32: Float32Array;
  UTF8ToString(ptr: number): string;
  stringToUTF8(s: string, ptr: number, max: number): void;
  lengthBytesUTF8(s: string): number;
  _malloc(n: number): number;
  _free(p: number): void;
  [fn: `_ag_${string}`]: Fn;
}

/** Thrown inside the worker and turned into an error reply; never crosses to the page. */
export class EngineFault extends Error {
  error: EngineError;
  constructor(error: EngineError) {
    super(error.message);
    this.error = error;
  }
}

export interface Engine {
  m: AgModule;
  ops: { name: string; symbol: string; category: number }[];
  opNumbers: Map<OpName, number>;
  /** 4 KB of scratch for structs and strings; the module is single threaded. */
  scratch: number;
}

export async function loadEngine(moduleUrl: string, locate?: (file: string) => string): Promise<Engine> {
  let m: AgModule;
  try {
    const mod = await import(/* @vite-ignore */ moduleUrl);
    m = await mod.default({ locateFile: locate ?? ((file: string) => new URL(file, moduleUrl).href) });
  } catch (e) {
    throw new EngineFault({ code: 'WasmLoadFailed', message: `the engine could not be loaded: ${String(e)}` });
  }
  if (m._ag_abi_version() !== ABI_VERSION)
    throw new EngineFault({ code: 'WasmLoadFailed', message: `engine ABI ${m._ag_abi_version()}, page expects ${ABI_VERSION}` });
  const ops = [];
  const opNumbers = new Map<OpName, number>();
  for (let i = 0; i < m._ag_op_count(); ++i) {
    const name = m.UTF8ToString(m._ag_op_name(i));
    ops.push({ name, symbol: m.UTF8ToString(m._ag_op_symbol(i)), category: 0 });
    opNumbers.set(name as OpName, i);
  }
  const scratch = m._malloc(4096);
  return { m, ops, opNumbers, scratch };
}

// ── memory helpers ──────────────────────────────────────────────────────────
// the heap never grows (ALLOW_MEMORY_GROWTH=0), so views stay valid

const i32 = (m: AgModule, ptr: number, v: number) => (m.HEAP32[ptr >> 2] = v);
const f32 = (m: AgModule, ptr: number, v: number) => (m.HEAPF32[ptr >> 2] = v);
const ri32 = (m: AgModule, ptr: number) => m.HEAP32[ptr >> 2];
const rf32 = (m: AgModule, ptr: number) => m.HEAPF32[ptr >> 2];

/** Allocate, run, free. A failed malloc is reported, never dereferenced. */
function withBuffer<T>(m: AgModule, bytes: number, fn: (ptr: number) => T): T {
  const ptr = m._malloc(Math.max(bytes, 4));
  if (!ptr) throw new EngineFault({ code: 'OutOfMemory', message: `could not allocate ${bytes} bytes` });
  try {
    return fn(ptr);
  } finally {
    m._free(ptr);
  }
}

function writeString(m: AgModule, s: string, ptr: number, max: number): number {
  m.stringToUTF8(s.slice(0, 120), ptr, max);
  return ptr;
}

// ── errors ──────────────────────────────────────────────────────────────────

/** Throw if `status` is not Ok. Engine node numbers are mapped back to ids. */
function check(e: Engine, status: number, ids?: readonly string[]) {
  if (status === 0) return;
  throw new EngineFault(lastError(e, status, ids));
}

function lastError(e: Engine, status: number, ids?: readonly string[]): EngineError {
  const name = ENGINE_CODES[status];
  const code: EngineError['code'] = name && name !== 'Ok' ? name : 'Internal';
  const node = e.m._ag_last_error_node();
  const slot = e.m._ag_last_error_slot();
  const err: EngineError = { code, message: e.m.UTF8ToString(e.m._ag_last_error()) };
  if (node >= 0 && ids && node < ids.length) err.nodeId = ids[node];
  if (slot >= 0) err.slot = slot;
  return err;
}

// ── struct packing ──────────────────────────────────────────────────────────

const OPTIM_KINDS: OptimConfig['kind'][] = ['sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'adamw'];
const DATASETS = ['moons', 'circles', 'spirals', 'xor', 'gaussians', 'sine1d', 'none'];

/** ag_attrs, 48 bytes. */
function writeAttrs(m: AgModule, p: number, n: NodeSpec) {
  const shape = n.shape ?? [];
  i32(m, p, shape.length);
  for (let i = 0; i < 4; ++i) i32(m, p + 4 + 4 * i, i < shape.length ? (shape[i] === 'B' ? BATCH : (shape[i] as number)) : 0);
  i32(m, p + 20, n.axis === undefined || n.axis === 'all' ? ALL_AXES : n.axis);
  i32(m, p + 24, n.keepDim ? 1 : 0);
  f32(m, p + 28, n.scalar ?? 0);
  i32(m, p + 32, n.role === 'features' ? 1 : n.role === 'targets' ? 2 : 0);
  i32(m, p + 36, Math.max(0, INIT_NAMES.indexOf(n.init ?? 'zeros')));
  f32(m, p + 40, n.initScale ?? 1);
  i32(m, p + 44, n.requiresGrad ? 1 : 0);
}

/** ag_optim_config, 56 bytes. */
function writeOptim(m: AgModule, p: number, o: OptimConfig) {
  const s = o.schedule ?? { kind: 'constant' };
  i32(m, p, OPTIM_KINDS.indexOf(o.kind));
  i32(m, p + 4, s.kind === 'constant' ? 0 : s.kind === 'stepDecay' ? 1 : 2);
  f32(m, p + 8, o.lr);
  f32(m, p + 12, s.kind === 'cosine' ? s.minLr : 0);
  f32(m, p + 16, s.kind === 'stepDecay' ? s.gamma : 0.5);
  i32(m, p + 20, s.kind === 'stepDecay' ? s.period : 100);
  i32(m, p + 24, s.kind === 'cosine' ? (s.warmupSteps ?? 0) : 0);
  i32(m, p + 28, s.kind === 'cosine' ? s.totalSteps : 1000);
  f32(m, p + 32, o.momentum ?? 0.9);
  f32(m, p + 36, o.beta1 ?? 0.9);
  f32(m, p + 40, o.beta2 ?? 0.999);
  f32(m, p + 44, o.eps ?? 1e-8);
  f32(m, p + 48, o.weightDecay ?? 0);
  f32(m, p + 52, o.clipNorm ?? 0);
}

/** ag_session_config, 92 bytes. */
function writeSessionConfig(m: AgModule, p: number, c: SessionConfig) {
  i32(m, p, c.batchSize);
  i32(m, p + 4, c.evalBatchSize ?? 256);
  i32(m, p + 8, c.seed >>> 0);
  i32(m, p + 12, 0);
  i32(m, p + 16, 1);
  i32(m, p + 20, DATASETS.indexOf(c.data.kind));
  i32(m, p + 24, c.data.n);
  f32(m, p + 28, c.data.noise);
  f32(m, p + 32, c.data.valFraction);
  writeOptim(m, p + 36, c.optim);
}

/** ag_grid, 28 bytes. */
function writeGrid(m: AgModule, p: number, g: Grid, node: number) {
  f32(m, p, g.x0);
  f32(m, p + 4, g.x1);
  f32(m, p + 8, g.y0);
  f32(m, p + 12, g.y1);
  i32(m, p + 16, g.res);
  i32(m, p + 20, node);
  i32(m, p + 24, g.element ?? 0);
}

/** ag_stats, 164 bytes. */
function readStats(m: AgModule, p: number): Stats {
  return {
    min: rf32(m, p), max: rf32(m, p + 4), mean: rf32(m, p + 8), std: rf32(m, p + 12),
    l2: rf32(m, p + 16), fracZero: rf32(m, p + 20), nonFinite: m.HEAPU32[(p + 24) >> 2],
    histLo: rf32(m, p + 28), histHi: rf32(m, p + 32),
    hist: m.HEAPU32.slice((p + 36) >> 2, ((p + 36) >> 2) + 32),
  };
}

// ── sessions ────────────────────────────────────────────────────────────────

export class EngineSession {
  e: Engine;
  handle: number;
  /** engine node number -> spec id, and back */
  ids: string[];
  index: Map<string, number>;
  info: SessionInfo;

  constructor(e: Engine, handle: number, ids: string[], key: number) {
    this.e = e;
    this.handle = handle;
    this.ids = ids;
    this.index = new Map(ids.map((id, i) => [id, i]));
    this.info = readInfo(e, handle, ids, key);
  }

  node(id: string): number {
    const n = this.index.get(id);
    if (n === undefined) throw new EngineFault({ code: 'NotFound', message: `no node "${id}"` });
    return n;
  }

  check(status: number) {
    check(this.e, status, this.ids);
  }

  free() {
    this.e.m._ag_session_free(this.handle);
  }

  reset(seed: number) {
    this.check(this.e.m._ag_session_reset(this.handle, seed >>> 0, 0));
  }

  setOptimizer(o: OptimConfig) {
    writeOptim(this.e.m, this.e.scratch, o);
    this.check(this.e.m._ag_session_set_optimizer(this.handle, this.e.scratch));
  }

  cursor(fault?: EngineError): Cursor {
    const { m, scratch } = this.e;
    this.check(m._ag_session_cursor(this.handle, scratch));
    const index = ri32(m, scratch);
    return { index, length: this.length(), fault };
  }

  private passLength = 0;
  private length() {
    return this.passLength;
  }

  /** Begin a pass. A NonFinite during stepping is part of the result, not a failure. */
  begin(pass: PassKind, seed?: TangentSeed): Cursor {
    const node = seed ? this.node(seed.nodeId) : -1;
    this.check(this.e.m._ag_session_begin(this.handle, PASS_KINDS.indexOf(pass), node, seed?.element ?? 0));
    this.passLength = this.info.schedules[pass].length;
    return this.cursor();
  }

  advance(n: number): Cursor {
    return this.stepping(this.e.m._ag_session_advance(this.handle, n, 0));
  }

  seek(index: number): Cursor {
    return this.stepping(this.e.m._ag_session_seek(this.handle, index));
  }

  private stepping(status: number): Cursor {
    if (status === ENGINE_CODES.indexOf('NonFinite')) return this.cursor(lastError(this.e, status, this.ids));
    this.check(status);
    return this.cursor();
  }

  train(steps: number): TrainResult {
    const { m, scratch } = this.e;
    const status = m._ag_session_train(this.handle, steps, scratch);
    const result: TrainResult = {
      step: ri32(m, scratch) + ri32(m, scratch + 4) * 2 ** 32,
      stepsRun: ri32(m, scratch + 8),
      loss: rf32(m, scratch + 12),
      gradNorm: rf32(m, scratch + 16),
      lr: rf32(m, scratch + 20),
    };
    if (status === ENGINE_CODES.indexOf('NonFinite')) result.fault = lastError(this.e, status, this.ids);
    else this.check(status);
    return result;
  }

  read(id: string, which: Which): { shape: number[]; data: Float32Array } {
    const n = this.node(id);
    const info = this.info.nodes[n];
    const count = info.shape.reduce((a, b) => a * b, 1);
    return withBuffer(this.e.m, count * 4, (p) => {
      this.check(this.e.m._ag_session_read(this.handle, n, WHICH.indexOf(which), p, count, 0));
      return { shape: info.shape, data: this.e.m.HEAPF32.slice(p >> 2, (p >> 2) + count) };
    });
  }

  readOptim(id: string, moment: 0 | 1): { shape: number[]; data: Float32Array } {
    const n = this.node(id);
    const info = this.info.nodes[n];
    const count = info.shape.reduce((a, b) => a * b, 1);
    return withBuffer(this.e.m, count * 4, (p) => {
      this.check(this.e.m._ag_session_read_optim(this.handle, n, moment, p, count));
      return { shape: info.shape, data: this.e.m.HEAPF32.slice(p >> 2, (p >> 2) + count) };
    });
  }

  /** Nodes without that buffer (no gradient, say) are left out rather than failing the call. */
  stats(ids: string[], which: Which): Record<string, Stats> {
    const { m, scratch } = this.e;
    const out: Record<string, Stats> = {};
    for (const id of ids) {
      const n = this.index.get(id);
      if (n === undefined) continue;
      if (m._ag_session_stats(this.handle, n, WHICH.indexOf(which), scratch) === 0) out[id] = readStats(m, scratch);
    }
    return out;
  }

  evalLoss(split: 'train' | 'val'): number {
    const { m, scratch } = this.e;
    this.check(m._ag_session_eval_loss(this.handle, split === 'train' ? 0 : 1, scratch));
    return rf32(m, scratch);
  }

  evalGrid(g: Grid): { res: number; data: Float32Array } {
    const node = g.nodeId ? this.node(g.nodeId) : -1;
    const count = g.res * g.res;
    return withBuffer(this.e.m, count * 4, (p) => {
      writeGrid(this.e.m, this.e.scratch, g, node);
      this.check(this.e.m._ag_session_eval_grid(this.handle, this.e.scratch, p, count));
      return { res: g.res, data: this.e.m.HEAPF32.slice(p >> 2, (p >> 2) + count) };
    });
  }

  lossSurface(x: TangentSeed, y: TangentSeed, g: Grid): { res: number; data: Float32Array } {
    const count = g.res * g.res;
    const px = this.node(x.nodeId), py = this.node(y.nodeId);
    return withBuffer(this.e.m, count * 4, (p) => {
      writeGrid(this.e.m, this.e.scratch, g, -1);
      this.check(this.e.m._ag_session_loss_surface(this.handle, px, x.element, py, y.element, this.e.scratch, p, count));
      return { res: g.res, data: this.e.m.HEAPF32.slice(p >> 2, (p >> 2) + count) };
    });
  }

  dataset(): Dataset {
    const { m, scratch } = this.e;
    this.check(m._ag_session_dataset(this.handle, scratch, 0, 0, 0, 0));
    const [n, nTrain, xDim, yDim] = [0, 1, 2, 3].map((i) => ri32(m, scratch + 4 * i));
    return withBuffer(m, (n * xDim + n * yDim) * 4, (p) => {
      const py = p + n * xDim * 4;
      this.check(m._ag_session_dataset(this.handle, scratch, p, n * xDim, py, n * yDim));
      return {
        n, nTrain, xDim, yDim,
        x: m.HEAPF32.slice(p >> 2, (p >> 2) + n * xDim),
        y: m.HEAPF32.slice(py >> 2, (py >> 2) + n * yDim),
      };
    });
  }

  exportParams(): Float32Array {
    const count = this.info.paramCount;
    return withBuffer(this.e.m, count * 4, (p) => {
      this.check(this.e.m._ag_session_export_params(this.handle, p, count));
      return this.e.m.HEAPF32.slice(p >> 2, (p >> 2) + count);
    });
  }

  importParams(params: Float32Array) {
    withBuffer(this.e.m, params.length * 4, (p) => {
      this.e.m.HEAPF32.set(params, p >> 2);
      this.check(this.e.m._ag_session_import_params(this.handle, p, params.length));
    });
  }
}

function readInfo(e: Engine, handle: number, ids: string[], key: number): SessionInfo {
  const { m, scratch } = e;
  check(e, m._ag_session_node_count(handle, scratch), ids);
  const count = ri32(m, scratch);
  const nodes: NodeInfo[] = [];
  for (let i = 0; i < count; ++i) {
    check(e, m._ag_session_node_shape(handle, i, scratch, scratch + 64), ids);
    const rank = ri32(m, scratch);
    const shape = [];
    for (let d = 0; d < rank; ++d) shape.push(ri32(m, scratch + 4 + 4 * d));
    nodes.push({ id: ids[i], shape, requiresGrad: ri32(m, scratch + 64) !== 0 });
  }
  const schedules = {} as Record<PassKind, Step[]>;
  PASS_KINDS.forEach((pass, k) => {
    check(e, m._ag_session_schedule_len(handle, k, scratch), ids);
    const len = ri32(m, scratch);
    schedules[pass] = withBuffer(m, len * 12, (p) => {
      check(e, m._ag_session_schedule(handle, k, p, len), ids);
      const steps: Step[] = [];
      for (let i = 0; i < len; ++i) {
        const node = ri32(m, p + 12 * i + 8);
        steps.push({ kind: STEP_KINDS[ri32(m, p + 12 * i)], slot: ri32(m, p + 12 * i + 4), nodeId: node >= 0 ? ids[node] : '' });
      }
      return steps;
    });
  });
  check(e, m._ag_session_param_count(handle, scratch), ids);
  return { session: key, nodes, schedules, paramCount: ri32(m, scratch) };
}

/** Validate, build the graph, create the session. The graph is freed either way. */
export function createSession(e: Engine, spec: GraphSpec, config: SessionConfig, key: number): EngineSession {
  const issues = validateSpec(spec);
  if (issues.length) throw new EngineFault({ code: 'BadSpec', message: issues[0].message, nodeId: issues[0].nodeId });
  const plan = toBuildPlan(spec, e.opNumbers);
  const { m, scratch } = e;
  check(e, m._ag_graph_new(scratch));
  const graph = ri32(m, scratch);
  try {
    for (const step of plan.steps) {
      // scratch layout: 0 out id, 16 inputs, 64 attrs, 128 name, 256 module
      step.inputs.forEach((n, i) => i32(m, scratch + 16 + 4 * i, n));
      writeAttrs(m, scratch + 64, step.spec);
      writeString(m, step.spec.name ?? step.spec.id, scratch + 128, 128);
      writeString(m, step.spec.module ?? '', scratch + 256, 128);
      check(e, m._ag_graph_add_node(graph, step.op, scratch + 16, step.inputs.length, scratch + 64, scratch + 128, scratch + 256, scratch), plan.toSpec);
    }
    if (plan.loss >= 0) check(e, m._ag_graph_set_loss(graph, plan.loss), plan.toSpec);
    for (const o of plan.outputs) check(e, m._ag_graph_add_output(graph, o), plan.toSpec);
    writeSessionConfig(m, scratch, config);
    check(e, m._ag_session_new(graph, scratch, scratch + 128), plan.toSpec);
    return new EngineSession(e, ri32(m, scratch + 128), plan.toSpec, key);
  } finally {
    m._ag_graph_free(graph);
  }
}

/** The scalar zoom: a second session over the element-wise expansion. */
export function scalarize(s: EngineSession, key: number): { session: EngineSession; view: ScalarView } {
  const { m, scratch } = s.e;
  const cap = 1024;
  return withBuffer(m, cap * 8, (p) => {
    s.check(m._ag_session_scalarize(s.handle, scratch, p, p + cap * 4, cap));
    const handle = ri32(m, scratch);
    check(s.e, m._ag_session_node_count(handle, scratch));
    const count = ri32(m, scratch);
    const ids = Array.from({ length: count }, (_, i) => `s${i}`);
    const session = new EngineSession(s.e, handle, ids, key);
    const nodes: NodeSpec[] = [];
    const origin: ScalarView['origin'] = [];
    for (let i = 0; i < count; ++i) {
      check(s.e, m._ag_session_node_op(handle, i, scratch, scratch + 16, scratch + 8));
      const op = s.e.ops[ri32(m, scratch)].name as OpName;
      const nIn = ri32(m, scratch + 8);
      const inputs = Array.from({ length: nIn }, (_, k) => ids[ri32(m, scratch + 16 + 4 * k)]);
      const name = m.UTF8ToString(m._ag_session_node_name(handle, i));
      const source = s.ids[ri32(m, p + 4 * i)];
      nodes.push({
        id: ids[i], op, inputs, name: name || s.e.ops[ri32(m, scratch)].symbol, module: source,
        requiresGrad: op === 'input' ? session.info.nodes[i].requiresGrad : undefined,
      });
      origin.push({ nodeId: source, element: ri32(m, p + cap * 4 + 4 * i) });
    }
    const loss = session.info.schedules.reverse.find((st) => st.kind === 'seedGrad')?.nodeId ?? '';
    const view: ScalarView = {
      ...session.info,
      spec: { version: 1, nodes, loss, outputs: [] },
      origin,
    };
    return { session, view };
  });
}
