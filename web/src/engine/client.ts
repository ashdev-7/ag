// EngineClient — the only thing the rest of the page uses to talk to the
// engine. It owns the worker, matches replies to requests, and recovers when
// the worker dies. No view imports the worker or the protocol envelope.
//
// Nothing here throws or rejects: every call resolves to a Result.

import type { GraphSpec } from '../graph/spec.ts';
import type {
  Cursor, Dataset, EngineError, Envelope, Frame, FromWorker, Grid, OptimConfig, PassKind, Replies,
  Request, RequestType, ScalarView, SessionConfig, SessionInfo, Stats, Stopped, TangentSeed,
  TrainResult, Watch, Which,
} from './protocol.ts';
import { PROTOCOL_VERSION } from './protocol.ts';

export type Result<T> = { ok: true; value: T } | { ok: false; error: EngineError };

export type EngineStatus =
  | { state: 'loading' }
  | { state: 'ready'; generation: number; where: 'worker' | 'page'; build: 'wasm' | 'js' }
  /** The worker died or hung and is being restarted. Sessions were lost. */
  | { state: 'restarting'; error: EngineError; attempt: number }
  /** Gave up: the engine crashed again and again, on the page as well as in a worker. */
  | { state: 'unavailable'; error: EngineError };

export interface EngineClientOptions {
  createWorker: () => Worker;
  /** The engine on the page's thread, where a worker cannot start or never gets ready (local.ts). */
  createLocal?: () => Worker;
  wasmUrl: string;
  /** The engine compiled to JavaScript, for browsers without WebAssembly. */
  jsUrl?: string;
  /** A request with no reply after this long means the worker is hung. Default 5000. */
  timeoutMs?: number;
  /** Restarts allowed in a row before giving up. Default 2. */
  maxRestarts?: number;
  onStatus?: (status: EngineStatus) => void;
}

export interface RunOptions {
  watch: Watch;
  /** Training speed, kept by the worker's clock (pacing.ts). */
  stepsPerSecond: number;
  maxSteps?: number;
  onFrame: (frame: Frame) => void;
  onStopped: (stopped: Stopped) => void;
}

/** A handle to one session inside the worker. */
export interface Session {
  readonly info: SessionInfo;
  /** False once the worker restarted; every call then fails with NotFound. */
  readonly alive: boolean;

  begin(pass: PassKind, seed?: TangentSeed): Promise<Result<Cursor>>;
  advance(n: number): Promise<Result<Cursor>>;
  seek(index: number): Promise<Result<Cursor>>;

  run(options: RunOptions): Promise<Result<{}>>;
  pause(): Promise<Result<{ step: number }>>;
  train(steps: number): Promise<Result<TrainResult>>;
  reset(seed: number): Promise<Result<{}>>;
  setOptimizer(optim: OptimConfig): Promise<Result<{}>>;

  read(nodeId: string, which: Which): Promise<Result<{ shape: number[]; data: Float32Array }>>;
  readOptim(nodeId: string, moment: 0 | 1): Promise<Result<{ shape: number[]; data: Float32Array }>>;
  stats(nodeIds: string[], which: Which): Promise<Result<Record<string, Stats>>>;
  evalLoss(split: 'train' | 'val'): Promise<Result<number>>;
  evalGrid(grid: Grid): Promise<Result<{ res: number; data: Float32Array }>>;
  lossSurface(x: TangentSeed, y: TangentSeed, grid: Grid): Promise<Result<{ res: number; data: Float32Array }>>;
  dataset(): Promise<Result<Dataset>>;
  exportParams(): Promise<Result<Float32Array>>;
  importParams(params: Float32Array): Promise<Result<{}>>;

  /** The scalar zoom level: a second session over the element-wise expansion. */
  scalarize(): Promise<Result<{ session: Session; view: ScalarView }>>;
  destroy(): Promise<void>;
}

export interface EngineClient {
  readonly status: EngineStatus;
  /** Resolves when the worker has loaded the engine, or with why it could not. */
  ready(): Promise<Result<{}>>;
  createSession(spec: GraphSpec, config: SessionConfig): Promise<Result<Session>>;
  /** Terminate the worker. Pending calls resolve with code 'Cancelled'. */
  dispose(): void;
}

type Pending = { resolve: (r: Result<unknown>) => void; timer: ReturnType<typeof setTimeout> };

const LONG_CALLS: Partial<Record<RequestType, number>> = { init: 30000, train: 60000, lossSurface: 20000, scalarize: 15000 };

/**
 * Run `f` at the start of the next display frame, after this frame's work has
 * been rendered; the worker then computes while the browser paints. In a page
 * without frames (tests), soon.
 */
function nextFrame(f: () => void) {
  if (typeof requestAnimationFrame === 'function') requestAnimationFrame(f);
  else setTimeout(f, 0);
}

export function createEngineClient(options: EngineClientOptions): EngineClient {
  const timeoutMs = options.timeoutMs ?? 5000;
  const maxRestarts = options.maxRestarts ?? 2;
  let worker: Worker | null = null;
  // in a worker first; on the page if the worker cannot start or never gets ready
  let onPage = false;
  let everReady = false;
  let status: EngineStatus = { state: 'loading' };
  let generation = 0;
  let restartsInARow = 0;
  let healthyTimer: ReturnType<typeof setTimeout> | undefined;
  let nextId = 1;
  let disposed = false;
  const pending = new Map<number, Pending>();
  const sessions = new Map<number, SessionImpl>();
  let readyWaiters: ((r: Result<{}>) => void)[] = [];

  const setStatus = (s: EngineStatus) => {
    status = s;
    options.onStatus?.(s);
    if (s.state === 'ready' || s.state === 'unavailable') {
      const r: Result<{}> = s.state === 'ready' ? { ok: true, value: {} } : { ok: false, error: s.error };
      readyWaiters.forEach((w) => w(r));
      readyWaiters = [];
    }
  };

  function send<T extends RequestType>(req: Extract<Request, { type: T }>, transfer: Transferable[] = []): Promise<Result<Replies[T]>> {
    return new Promise((resolve) => {
      if (!worker || (status.state !== 'ready' && req.type !== 'init')) {
        resolve({ ok: false, error: { code: 'WorkerCrashed', message: 'the engine is not running' } });
        return;
      }
      const id = nextId++;
      const timer = setTimeout(() => crash({ code: 'Timeout', message: `the engine did not answer "${req.type}" in time` }),
        LONG_CALLS[req.type] ?? timeoutMs);
      pending.set(id, { resolve: resolve as (r: Result<unknown>) => void, timer });
      worker.postMessage({ id, ...req } as Envelope, transfer);
    });
  }

  function onMessage(msg: FromWorker) {
    if ('id' in msg) {
      const p = pending.get(msg.id);
      if (!p) return;
      clearTimeout(p.timer);
      pending.delete(msg.id);
      p.resolve(msg.ok ? { ok: true, value: msg.result } : { ok: false, error: msg.error });
      return;
    }
    if (msg.type === 'fatal') return crash(msg.error);
    sessions.get(msg.session)?.event(msg);
  }

  /** The worker never got going: run the engine on the page instead. True if it switched. */
  function moveToPage(): boolean {
    if (onPage || everReady || !options.createLocal) return false;
    onPage = true;
    restartsInARow = 0;
    return true;
  }

  function start() {
    try {
      worker = onPage ? options.createLocal!() : options.createWorker();
    } catch (e) {
      if (moveToPage()) return start();
      setStatus({ state: 'unavailable', error: { code: 'WasmLoadFailed', message: `the engine could not start: ${String(e)}` } });
      return;
    }
    worker.onmessage = (ev: MessageEvent<FromWorker>) => onMessage(ev.data);
    worker.onerror = (ev) => {
      ev.preventDefault();
      crash({ code: 'WorkerCrashed', message: ev.message || 'the engine worker failed' });
    };
    const mine = worker;
    send({ type: 'init', wasmUrl: options.wasmUrl, jsUrl: options.jsUrl, protocol: PROTOCOL_VERSION }).then((r) => {
      if (disposed || worker !== mine) return;
      if (!r.ok) {
        // a load failure will not fix itself by retrying the same way, but a
        // worker that could not load the engine may manage on the page; a
        // crash or timeout during init has already been handled by crash()
        if (r.error.code === 'WasmLoadFailed' || r.error.code === 'Internal') {
          if (moveToPage()) {
            stop(r.error);
            return start();
          }
          giveUp(r.error);
        }
        return;
      }
      everReady = true;
      generation++;
      setStatus({ state: 'ready', generation, where: onPage ? 'page' : 'worker', build: r.value.build ?? 'wasm' });
      clearTimeout(healthyTimer);
      healthyTimer = setTimeout(() => (restartsInARow = 0), 10000);
    });
  }

  function stop(error: EngineError) {
    worker?.terminate();
    worker = null;
    pending.forEach((p) => {
      clearTimeout(p.timer);
      p.resolve({ ok: false, error });
    });
    pending.clear();
    sessions.forEach((s) => s.kill(error));
    sessions.clear();
  }

  function giveUp(error: EngineError) {
    stop(error);
    setStatus({ state: 'unavailable', error });
  }

  function crash(error: EngineError) {
    if (disposed || status.state === 'unavailable') return;
    stop(error);
    clearTimeout(healthyTimer);
    if (moveToPage()) return start();
    if (++restartsInARow > maxRestarts) return giveUp(error);
    setStatus({ state: 'restarting', error, attempt: restartsInARow });
    start();
  }

  // ── sessions ──────────────────────────────────────────────────────────────

  class SessionImpl implements Session {
    info: SessionInfo;
    alive = true;
    private key: number;
    private runOpts: RunOptions | null = null;

    constructor(info: SessionInfo) {
      this.info = info;
      this.key = info.session;
      sessions.set(this.key, this);
    }

    kill(error: EngineError) {
      this.alive = false;
      this.runOpts?.onStopped({ type: 'stopped', session: this.key, step: 0, reason: 'fault', error });
      this.runOpts = null;
    }

    event(msg: Frame | Stopped) {
      const opts = this.runOpts;
      if (!opts) return;
      if (msg.type === 'stopped') {
        this.runOpts = null;
        opts.onStopped(msg);
        return;
      }
      try {
        opts.onFrame(msg);
      } finally {
        // acknowledged on the next display frame: one frame in flight and at
        // most one per display frame. acking on receipt let a fast worker keep
        // the page drawing back to back, which dropped every other frame
        nextFrame(() => {
          if (this.runOpts) void this.call({ type: 'ackFrame', session: this.key, seq: msg.seq });
        });
      }
    }

    private call<T extends RequestType>(req: Extract<Request, { type: T }>, transfer?: Transferable[]): Promise<Result<Replies[T]>> {
      if (!this.alive) return Promise.resolve({ ok: false, error: { code: 'NotFound', message: 'the engine restarted and this session was lost' } });
      return send(req, transfer);
    }

    private map<T, U>(p: Promise<Result<T>>, f: (t: T) => U): Promise<Result<U>> {
      return p.then((r) => (r.ok ? { ok: true, value: f(r.value) } : r));
    }

    begin(pass: PassKind, seed?: TangentSeed) { return this.call({ type: 'begin', session: this.key, pass, seed }); }
    advance(n: number) { return this.call({ type: 'advance', session: this.key, n }); }
    seek(index: number) { return this.call({ type: 'seek', session: this.key, index }); }

    run(o: RunOptions) {
      this.runOpts = o;
      return this.call({ type: 'run', session: this.key, watch: o.watch, stepsPerSecond: o.stepsPerSecond, maxSteps: o.maxSteps });
    }
    pause() {
      this.runOpts = null;
      return this.call({ type: 'pause', session: this.key });
    }
    train(steps: number) { return this.call({ type: 'train', session: this.key, steps }); }
    reset(seed: number) {
      this.runOpts = null;
      return this.call({ type: 'reset', session: this.key, seed });
    }
    setOptimizer(optim: OptimConfig) { return this.call({ type: 'setOptimizer', session: this.key, optim }); }

    read(nodeId: string, which: Which) { return this.call({ type: 'read', session: this.key, nodeId, which }); }
    readOptim(nodeId: string, moment: 0 | 1) { return this.call({ type: 'readOptim', session: this.key, nodeId, moment }); }
    stats(nodeIds: string[], which: Which) {
      return this.map(this.call({ type: 'stats', session: this.key, nodeIds, which }), (r) => r.stats);
    }
    evalLoss(split: 'train' | 'val') { return this.map(this.call({ type: 'evalLoss', session: this.key, split }), (r) => r.loss); }
    evalGrid(grid: Grid) { return this.call({ type: 'evalGrid', session: this.key, grid }); }
    lossSurface(x: TangentSeed, y: TangentSeed, grid: Grid) { return this.call({ type: 'lossSurface', session: this.key, x, y, grid }); }
    dataset() { return this.call({ type: 'dataset', session: this.key }); }
    exportParams() { return this.map(this.call({ type: 'exportParams', session: this.key }), (r) => r.params); }
    importParams(params: Float32Array) {
      const copy = params.slice();
      return this.call({ type: 'importParams', session: this.key, params: copy }, [copy.buffer]);
    }

    scalarize() {
      return this.map(this.call({ type: 'scalarize', session: this.key }), (view) => ({ session: new SessionImpl(view) as Session, view }));
    }

    async destroy() {
      this.runOpts = null;
      if (this.alive) await this.call({ type: 'destroySession', session: this.key });
      this.alive = false;
      sessions.delete(this.key);
    }
  }

  start();

  return {
    get status() {
      return status;
    },
    ready() {
      if (status.state === 'ready') return Promise.resolve({ ok: true, value: {} });
      if (status.state === 'unavailable') return Promise.resolve({ ok: false, error: status.error });
      return new Promise((resolve) => readyWaiters.push(resolve));
    },
    async createSession(spec: GraphSpec, config: SessionConfig): Promise<Result<Session>> {
      const r = await send({ type: 'createSession', spec, config });
      return r.ok ? { ok: true, value: new SessionImpl(r.value) } : r;
    },
    dispose() {
      disposed = true;
      clearTimeout(healthyTimer);
      stop({ code: 'Cancelled', message: 'the engine was shut down' });
    },
  };
}
