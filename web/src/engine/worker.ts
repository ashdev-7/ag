// The engine worker. Owns the WebAssembly module and every session in it.
// Each request gets exactly one reply. Training runs as a loop of ticks: a
// tick trains `stepsPerFrame` steps, posts one Frame, and waits for the page
// to acknowledge it before the next tick, so frames never queue up and a slow
// page slows the frame rate, not the training. Between ticks the message
// queue runs, which is how pause and reset get in.

import { createSession, EngineFault, loadEngine, scalarize } from './bridge.ts';
import type { Engine, EngineSession } from './bridge.ts';
import { PROTOCOL_VERSION } from './protocol.ts';
import type { EngineError, Envelope, Frame, FromWorker, Replies, Request, Watch } from './protocol.ts';

interface WorkerScope {
  postMessage(msg: FromWorker, transfer?: Transferable[]): void;
  onmessage: ((ev: MessageEvent<Envelope>) => void) | null;
  close(): void;
}
const scope = self as unknown as WorkerScope;

let engine: Engine | null = null;
const sessions = new Map<number, EngineSession>();
let nextKey = 1;

interface Run {
  watch: Watch;
  stepsPerFrame: number;
  maxSteps?: number;
  seq: number;
  waiting: boolean;
  frames: number;
  step: number;
  clock: { t: number; step: number };
}
const runs = new Map<number, Run>();

function session(key: number): EngineSession {
  const s = sessions.get(key);
  if (!s) throw new EngineFault({ code: 'NotFound', message: 'this session no longer exists' });
  return s;
}

function needEngine(): Engine {
  if (!engine) throw new EngineFault({ code: 'InvalidState', message: 'the engine is not loaded yet' });
  return engine;
}

// typed arrays in a reply are moved, not copied
function transferables(x: unknown, out: Transferable[] = [], depth = 0): Transferable[] {
  if (depth > 3 || x === null || typeof x !== 'object') return out;
  if (ArrayBuffer.isView(x)) {
    out.push(x.buffer as ArrayBuffer);
    return out;
  }
  for (const v of Object.values(x)) transferables(v, out, depth + 1);
  return out;
}

async function handle(req: Request): Promise<Replies[Request['type']]> {
  switch (req.type) {
    case 'init': {
      if (req.protocol !== PROTOCOL_VERSION)
        throw new EngineFault({ code: 'WasmLoadFailed', message: `page speaks protocol ${req.protocol}, worker ${PROTOCOL_VERSION}` });
      engine = await loadEngine(req.wasmUrl);
      return { abi: engine.m._ag_abi_version(), ops: engine.ops };
    }
    case 'createSession': {
      const key = nextKey++;
      const s = createSession(needEngine(), req.spec, req.config, key);
      sessions.set(key, s);
      return s.info;
    }
    case 'destroySession':
      runs.delete(req.session);
      sessions.get(req.session)?.free();
      sessions.delete(req.session);
      return {};
    case 'reset':
      runs.delete(req.session);
      session(req.session).reset(req.seed);
      return {};
    case 'setOptimizer':
      session(req.session).setOptimizer(req.optim);
      return {};
    case 'begin':
      return session(req.session).begin(req.pass, req.seed);
    case 'advance':
      return session(req.session).advance(req.n);
    case 'seek':
      return session(req.session).seek(req.index);
    case 'run': {
      session(req.session);
      runs.set(req.session, {
        watch: req.watch, stepsPerFrame: Math.max(1, Math.floor(req.stepsPerFrame)), maxSteps: req.maxSteps,
        seq: 0, waiting: false, frames: 0, step: 0, clock: { t: performance.now(), step: -1 },
      });
      setTimeout(() => tick(req.session), 0);
      return {};
    }
    case 'pause': {
      const run = runs.get(req.session);
      runs.delete(req.session);
      return { step: run?.step ?? 0 };
    }
    case 'ackFrame': {
      const run = runs.get(req.session);
      if (run && run.seq === req.seq) {
        run.waiting = false;
        setTimeout(() => tick(req.session), 0);
      }
      return {};
    }
    case 'train':
      return session(req.session).train(req.steps);
    case 'read':
      return session(req.session).read(req.nodeId, req.which);
    case 'readOptim':
      return session(req.session).readOptim(req.nodeId, req.moment);
    case 'stats':
      return { stats: session(req.session).stats(req.nodeIds, req.which) };
    case 'evalLoss':
      return { loss: session(req.session).evalLoss(req.split) };
    case 'evalGrid':
      return session(req.session).evalGrid(req.grid);
    case 'lossSurface':
      return session(req.session).lossSurface(req.x, req.y, req.grid);
    case 'dataset':
      return session(req.session).dataset();
    case 'exportParams':
      return { params: session(req.session).exportParams() };
    case 'importParams':
      session(req.session).importParams(req.params);
      return {};
    case 'scalarize': {
      const key = nextKey++;
      const { session: s, view } = scalarize(session(req.session), key);
      sessions.set(key, s);
      return view;
    }
  }
}

// ── training loop ───────────────────────────────────────────────────────────

function tick(key: number) {
  const run = runs.get(key);
  const s = sessions.get(key);
  if (!run || !s || run.waiting) return;
  try {
    const steps = run.maxSteps ? Math.min(run.stepsPerFrame, run.maxSteps - run.step) : run.stepsPerFrame;
    const r = s.train(steps);
    run.step = r.step;
    if (r.fault) {
      runs.delete(key);
      scope.postMessage({ type: 'stopped', session: key, step: r.step, reason: 'fault', error: r.fault });
      return;
    }
    const now = performance.now();
    const rate = run.clock.step < 0 ? 0 : ((r.step - run.clock.step) * 1000) / Math.max(1, now - run.clock.t);
    if (now - run.clock.t > 500 || run.clock.step < 0) run.clock = { t: now, step: r.step };

    const w = run.watch;
    const frame: Frame = {
      type: 'frame', session: key, seq: ++run.seq, step: r.step, loss: r.loss, gradNorm: r.gradNorm, lr: r.lr,
      stepsPerSecond: rate,
      valueStats: s.stats(w.stats, 'value'),
      gradStats: s.stats(w.stats, 'grad'),
      tensors: w.tensors.map((t) => ({ nodeId: t.nodeId, which: t.which, ...s.read(t.nodeId, t.which) })),
    };
    if (w.grid && run.frames % Math.max(1, w.gridEvery ?? 1) === 0) frame.grid = s.evalGrid(w.grid);
    if (w.valEvery && run.frames % w.valEvery === 0) frame.valLoss = s.evalLoss('val');
    run.frames++;
    run.waiting = true;
    scope.postMessage(frame, transferables(frame));
    if (run.maxSteps && r.step >= run.maxSteps) {
      runs.delete(key);
      scope.postMessage({ type: 'stopped', session: key, step: r.step, reason: 'maxSteps' });
    }
  } catch (e) {
    runs.delete(key);
    const error = toError(e);
    if (error) scope.postMessage({ type: 'stopped', session: key, step: run.step, reason: 'fault', error });
  }
}

// ── errors ──────────────────────────────────────────────────────────────────
// an EngineFault is an ordinary error reply. a trap inside WebAssembly leaves
// the module in an unknown state: report it once and close, and the client
// starts a fresh worker

function toError(e: unknown): EngineError | null {
  if (e instanceof EngineFault) return e.error;
  if (e instanceof WebAssembly.RuntimeError) {
    scope.postMessage({ type: 'fatal', error: { code: 'WorkerCrashed', message: `the engine trapped: ${e.message}` } });
    scope.close();
    return null;
  }
  return { code: 'Internal', message: e instanceof Error ? e.message : String(e) };
}

scope.onmessage = async (ev) => {
  const { id, ...req } = ev.data;
  try {
    const result = await handle(req as Request);
    scope.postMessage({ id, ok: true, result }, transferables(result));
  } catch (e) {
    const error = toError(e);
    if (error) scope.postMessage({ id, ok: false, error });
  }
};
