// Messages between the page and the engine worker. Every request carries an
// id and gets exactly one reply with the same id. Frames, stops and fatal
// errors are the only messages the worker sends unprompted.

import type { GraphSpec } from '../graph/spec.ts';

export const PROTOCOL_VERSION = 1;

// ── shared types ────────────────────────────────────────────────────────────

/** Mirrors ag::Code in engine/include/ag/core.hpp, in order. */
export const ENGINE_CODES = [
  'Ok', 'InvalidArgument', 'InvalidGraph', 'ShapeMismatch', 'LimitExceeded',
  'OutOfMemory', 'NonFinite', 'InvalidState', 'NotFound', 'Unimplemented', 'Internal',
] as const;

export type ErrorCode =
  | Exclude<(typeof ENGINE_CODES)[number], 'Ok'>
  // raised on the JS side of the bridge
  | 'WorkerCrashed' | 'Timeout' | 'WasmLoadFailed' | 'Cancelled' | 'BadSpec';

export interface EngineError {
  code: ErrorCode;
  message: string;
  /** Spec id of the node the error is about, when there is one. */
  nodeId?: string;
  /** Which input of that node. */
  slot?: number;
}

export type PassKind = 'forward' | 'reverse' | 'train' | 'forwardMode';
export type StepKind = 'forward' | 'seedGrad' | 'vjp' | 'update' | 'seedTangent' | 'jvp';
export type Which = 'value' | 'grad' | 'tangent';
export type SessionId = number;

/** Engine enum order (ag::PassKind, ag::StepKind, ag::Which). */
export const PASS_KINDS: readonly PassKind[] = ['forward', 'reverse', 'train', 'forwardMode'];
export const STEP_KINDS: readonly StepKind[] = ['forward', 'seedGrad', 'vjp', 'update', 'seedTangent', 'jvp'];
export const WHICH: readonly Which[] = ['value', 'grad', 'tangent'];

/** One micro-step of a pass. The list of these is the timeline. */
export interface Step {
  kind: StepKind;
  /** '' for the seedTangent step, whose node is chosen when the pass begins. */
  nodeId: string;
  /** For 'vjp': which input of `nodeId` receives gradient. */
  slot: number;
}

export interface NodeInfo {
  id: string;
  shape: number[];
  requiresGrad: boolean;
}

export interface Stats {
  min: number; max: number; mean: number; std: number;
  l2: number;
  fracZero: number;
  nonFinite: number;
  histLo: number; histHi: number;
  hist: Uint32Array;
}

export interface OptimConfig {
  kind: 'sgd' | 'momentum' | 'nesterov' | 'rmsprop' | 'adam' | 'adamw';
  lr: number;
  schedule?:
    | { kind: 'constant' }
    | { kind: 'stepDecay'; gamma: number; period: number }
    | { kind: 'cosine'; totalSteps: number; minLr: number; warmupSteps?: number };
  momentum?: number;
  beta1?: number;
  beta2?: number;
  eps?: number;
  weightDecay?: number;
  clipNorm?: number;
}

export interface DatasetConfig {
  kind: 'moons' | 'circles' | 'spirals' | 'xor' | 'gaussians' | 'sine1d' | 'none';
  n: number;
  noise: number;
  valFraction: number;
}

export interface SessionConfig {
  batchSize: number;
  evalBatchSize?: number;
  seed: number;
  optim: OptimConfig;
  data: DatasetConfig;
}

export interface TangentSeed { nodeId: string; element: number }

export interface Grid {
  x0: number; x1: number; y0: number; y1: number;
  res: number;
  nodeId?: string;
  element?: number;
}

/** What the worker should attach to each frame while training. */
export interface Watch {
  /** Nodes whose value and gradient statistics are wanted. */
  stats: string[];
  /** Nodes whose full tensors are wanted (the selected node, usually one). */
  tensors: { nodeId: string; which: Which }[];
  /** Decision boundary grid, recomputed every `gridEvery` frames. */
  grid?: Grid;
  gridEvery?: number;
  /** Validation loss every this many frames. 0 disables. */
  valEvery?: number;
}

export interface Dataset {
  n: number;
  nTrain: number;
  xDim: number;
  yDim: number;
  x: Float32Array;
  y: Float32Array;
}

// ── requests ────────────────────────────────────────────────────────────────

export type Request =
  | { type: 'init'; wasmUrl: string; protocol: typeof PROTOCOL_VERSION }
  | { type: 'createSession'; spec: GraphSpec; config: SessionConfig }
  | { type: 'destroySession'; session: SessionId }
  | { type: 'reset'; session: SessionId; seed: number }
  | { type: 'setOptimizer'; session: SessionId; optim: OptimConfig }
  // stepping
  | { type: 'begin'; session: SessionId; pass: PassKind; seed?: TangentSeed }
  | { type: 'advance'; session: SessionId; n: number }
  | { type: 'seek'; session: SessionId; index: number }
  // training
  | { type: 'run'; session: SessionId; watch: Watch; stepsPerFrame: number; maxSteps?: number }
  | { type: 'pause'; session: SessionId }
  | { type: 'ackFrame'; session: SessionId; seq: number }
  | { type: 'train'; session: SessionId; steps: number }
  // reading
  | { type: 'read'; session: SessionId; nodeId: string; which: Which }
  | { type: 'readOptim'; session: SessionId; nodeId: string; moment: 0 | 1 }
  | { type: 'stats'; session: SessionId; nodeIds: string[]; which: Which }
  | { type: 'evalLoss'; session: SessionId; split: 'train' | 'val' }
  | { type: 'evalGrid'; session: SessionId; grid: Grid }
  | { type: 'lossSurface'; session: SessionId; x: TangentSeed; y: TangentSeed; grid: Grid }
  | { type: 'dataset'; session: SessionId }
  | { type: 'exportParams'; session: SessionId }
  | { type: 'importParams'; session: SessionId; params: Float32Array }
  | { type: 'scalarize'; session: SessionId };

export interface SessionInfo {
  session: SessionId;
  nodes: NodeInfo[];
  schedules: Record<PassKind, Step[]>;
  paramCount: number;
}

export interface Cursor {
  /** Next step to run; equals the schedule length when the pass is finished. */
  index: number;
  length: number;
  /** Set when the pass stopped on a nan or inf. */
  fault?: EngineError;
}

export interface TrainResult {
  step: number;
  stepsRun: number;
  loss: number;
  gradNorm: number;
  lr: number;
  fault?: EngineError;
}

export interface ScalarView extends SessionInfo {
  spec: GraphSpec;
  /** For each scalar node: the tensor node and element it came from (-1 for intermediates). */
  origin: { nodeId: string; element: number }[];
}

/** Reply payload for each request type. */
export interface Replies {
  init: { abi: number; ops: { name: string; symbol: string; category: number }[] };
  createSession: SessionInfo;
  destroySession: {};
  reset: {};
  setOptimizer: {};
  begin: Cursor;
  advance: Cursor;
  seek: Cursor;
  run: {};
  pause: { step: number };
  ackFrame: {};
  train: TrainResult;
  read: { shape: number[]; data: Float32Array };
  readOptim: { shape: number[]; data: Float32Array };
  stats: { stats: Record<string, Stats> };
  evalLoss: { loss: number };
  evalGrid: { res: number; data: Float32Array };
  lossSurface: { res: number; data: Float32Array };
  dataset: Dataset;
  exportParams: { params: Float32Array };
  importParams: {};
  scalarize: ScalarView;
}

export type RequestType = Request['type'];

export type Envelope = { id: number } & Request;

export type Reply<T extends RequestType = RequestType> =
  | { id: number; ok: true; result: Replies[T] }
  | { id: number; ok: false; error: EngineError };

// ── worker-initiated messages ───────────────────────────────────────────────

/**
 * One consistent snapshot taken between training steps. The worker sends the
 * next frame only after the page has acknowledged this one, so a slow page
 * slows the frame rate, not the training, and messages never queue up.
 */
export interface Frame {
  type: 'frame';
  session: SessionId;
  seq: number;
  step: number;
  loss: number;
  valLoss?: number;
  gradNorm: number;
  lr: number;
  stepsPerSecond: number;
  valueStats: Record<string, Stats>;
  gradStats: Record<string, Stats>;
  tensors: { nodeId: string; which: Which; shape: number[]; data: Float32Array }[];
  grid?: { res: number; data: Float32Array };
}

/** Training stopped by itself: a fault (rolled back) or `maxSteps` reached. */
export interface Stopped {
  type: 'stopped';
  session: SessionId;
  step: number;
  reason: 'fault' | 'maxSteps';
  error?: EngineError;
}

/** The engine trapped (out-of-bounds, abort). The worker is unusable and closes itself. */
export interface Fatal {
  type: 'fatal';
  error: EngineError;
}

export type WorkerEvent = Frame | Stopped | Fatal;
export type FromWorker = Reply | WorkerEvent;
