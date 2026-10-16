// The engine host: owns the engine module and every session in it. It runs
// in a worker (worker.ts) or, where a browser cannot start that worker, on
// the page itself (local.ts); either way it talks only through `scope`.
// Each request gets exactly one reply. Training runs as a loop of ticks: a
// tick trains `stepsPerFrame` steps, posts one Frame, and waits for the page
// to acknowledge it before the next tick, so frames never queue up and a slow
// page slows the frame rate, not the training. Between ticks the message
// queue runs, which is how pause and reset get in.
//
// The engine is the WebAssembly build where the browser runs WebAssembly, and
// the same C++ compiled to plain JavaScript where it does not (iOS Lockdown
// Mode, browsers with WebAssembly switched off by policy).

import { refreshDue, stepsDue } from './pacing.ts';
import { createSession, EngineFault, loadEngine, scalarize } from './bridge.ts';
import type { Engine, EngineSession } from './bridge.ts';
import { PROTOCOL_VERSION } from './protocol.ts';
import type { EngineError, Envelope, Frame, FromWorker, Replies, Request, Watch } from './protocol.ts';

export interface HostScope {
  postMessage(msg: FromWorker, transfer?: Transferable[]): void;
  onmessage: ((ev: { data: Envelope }) => void) | null;
  close(): void;
}

/** True where WebAssembly exists and can compile a module. */
export function canRunWasm(): boolean {
  try {
    if (typeof WebAssembly !== 'object') return false;
    // the smallest valid module: magic number and version
    return new WebAssembly.Module(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0])) instanceof WebAssembly.Module;
  } catch {
    return false;
  }
}

export function startHost(scope: HostScope) {
  let engine: Engine | null = null;
  const sessions = new Map<number, EngineSession>();
  let nextKey = 1;

  interface Run {
    watch: Watch;
    stepsPerSecond: number;
    maxSteps?: number;
    seq: number;
    /** pacing: when this run's clock started, and steps taken since */
    t0: number;
    done: number;
    /** extras: when each was last done and what it cost, in ms */
    grid: { last: number; cost: number };
    val: { last: number; cost: number };
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
        // WebAssembly where it runs; where it is missing, blocked, or the build
        // fails to load, the same engine compiled to JavaScript
        let build: 'wasm' | 'js' = 'wasm';
        if (canRunWasm()) {
          try {
            engine = await loadEngine(req.wasmUrl);
          } catch (e) {
            if (!req.jsUrl) throw e;
          }
        }
        if (!engine) {
          if (!req.jsUrl) throw new EngineFault({ code: 'WasmLoadFailed', message: 'WebAssembly is unavailable and no JavaScript build was given' });
          engine = await loadEngine(req.jsUrl);
          build = 'js';
        }
        return { abi: engine.m._ag_abi_version(), ops: engine.ops, build };
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
        // maxSteps counts the session's steps, not this run's: read where the
        // session is (training 0 steps reports it), so a resumed run stops at
        // the same place a straight one would
        const current = session(req.session).train(0).step;
        runs.set(req.session, {
          watch: req.watch, stepsPerSecond: Math.max(0.1, Math.min(10000, req.stepsPerSecond || 1)), maxSteps: req.maxSteps,
          seq: 0, waiting: false, frames: 0, step: current, t0: performance.now(), done: 0,
          grid: { last: -Infinity, cost: 0 }, val: { last: -Infinity, cost: 0 }, clock: { t: performance.now(), step: -1 },
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
      // as many steps as the clock says are due; none due yet: come back when one is.
      // a timer left over from an earlier run of this session must not tick this one
      const due = stepsDue(run.stepsPerSecond, performance.now() - run.t0, run.done);
      if (due.steps === 0) {
        setTimeout(() => runs.get(key) === run && tick(key), Math.max(1, due.waitMs));
        return;
      }
      if (due.behind) run.t0 = performance.now() - ((run.done + due.steps) / run.stepsPerSecond) * 1000;
      run.done += due.steps;
      const steps = run.maxSteps ? Math.min(due.steps, run.maxSteps - run.step) : due.steps;
      if (steps <= 0) {
        runs.delete(key);
        scope.postMessage({ type: 'stopped', session: key, step: run.step, reason: 'maxSteps' });
        return;
      }
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
      // the boundary and the test loss on a time budget: see refreshDue
      if (w.grid && refreshDue(now, run.grid.last, w.gridEveryMs ?? 0, run.grid.cost)) {
        const t = performance.now();
        frame.grid = s.evalGrid(w.grid);
        run.grid = { last: now, cost: performance.now() - t };
      }
      if (w.valEveryMs && refreshDue(now, run.val.last, w.valEveryMs, run.val.cost)) {
        const t = performance.now();
        frame.valLoss = s.evalLoss('val');
        run.val = { last: now, cost: performance.now() - t };
      }
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
    // a trap: WebAssembly's RuntimeError, or the JavaScript build's abort
    if ((typeof WebAssembly === 'object' && e instanceof WebAssembly.RuntimeError) || (e instanceof Error && e.name === 'RuntimeError')) {
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
}
