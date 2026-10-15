// The client's recovery rules, against a fake worker that can be told to
// hang, trap or answer.

import { describe, expect, it } from 'vitest';
import { createEngineClient } from './client.ts';
import type { EngineStatus } from './client.ts';
import type { Envelope, FromWorker } from './protocol.ts';

type Mode = 'ok' | 'hang' | 'fatal' | 'loadFails';

class FakeWorker {
  onmessage: ((ev: MessageEvent<FromWorker>) => void) | null = null;
  onerror: ((ev: ErrorEvent) => void) | null = null;
  terminated = false;
  sent: Envelope[] = [];
  mode: () => Mode;
  constructor(mode: () => Mode) {
    this.mode = mode;
  }
  postMessage(msg: Envelope) {
    this.sent.push(msg);
    const reply = (m: FromWorker) => setTimeout(() => !this.terminated && this.onmessage?.({ data: m } as MessageEvent<FromWorker>), 1);
    const mode = this.mode();
    if (msg.type === 'init') {
      if (mode === 'loadFails') return reply({ id: msg.id, ok: false, error: { code: 'WasmLoadFailed', message: 'no wasm' } });
      return reply({ id: msg.id, ok: true, result: { abi: 2, ops: [] } });
    }
    if (mode === 'hang') return;
    if (mode === 'fatal') return reply({ type: 'fatal', error: { code: 'WorkerCrashed', message: 'trap' } });
    if (msg.type === 'createSession')
      return reply({ id: msg.id, ok: true, result: { session: 1, nodes: [], schedules: { forward: [], reverse: [], train: [], forwardMode: [] }, paramCount: 0 } });
    reply({ id: msg.id, ok: true, result: { index: 1, length: 1 } as never });
  }
  terminate() {
    this.terminated = true;
  }
}

const spec = { version: 1 as const, nodes: [{ id: 'a', op: 'input' as const, inputs: [] }], loss: '', outputs: [] };
const config = { batchSize: 1, seed: 1, optim: { kind: 'sgd' as const, lr: 0.1 }, data: { kind: 'none' as const, n: 0, noise: 0, valFraction: 0 } };

function setup(initial: Mode) {
  const statuses: EngineStatus[] = [];
  let workers = 0;
  let mode = initial;
  let last: FakeWorker | null = null;
  const client = createEngineClient({
    createWorker: () => {
      workers++;
      last = new FakeWorker(() => mode);
      return last as unknown as Worker;
    },
    wasmUrl: 'x',
    timeoutMs: 40,
    onStatus: (s) => statuses.push(s),
  });
  return { client, statuses, workers: () => workers, setMode: (m: Mode) => (mode = m), worker: () => last! };
}

describe('EngineClient', () => {
  it('reports ready and creates sessions', async () => {
    const { client } = setup('ok');
    expect((await client.ready()).ok).toBe(true);
    expect((await client.createSession(spec, config)).ok).toBe(true);
    client.dispose();
  });

  it('restarts a hung worker', async () => {
    const t = setup('ok');
    await t.client.ready();
    t.setMode('hang');
    const r = await t.client.createSession(spec, config);
    expect(r.ok ? '' : r.error.code).toBe('Timeout');
    t.setMode('ok');
    await t.client.ready();
    expect(t.workers()).toBe(2);
    expect(t.statuses.map((s) => s.state)).toEqual(['ready', 'restarting', 'ready']);
    expect((await t.client.createSession(spec, config)).ok).toBe(true);
    t.client.dispose();
  });

  it('restarts after a trap; sessions from before answer NotFound', async () => {
    const t = setup('ok');
    await t.client.ready();
    const s = await t.client.createSession(spec, config);
    if (!s.ok) throw new Error('setup');
    expect((await s.value.advance(1)).ok).toBe(true);
    t.setMode('fatal');
    const r = await s.value.advance(1);
    expect(r.ok ? '' : r.error.code).toBe('WorkerCrashed');
    t.setMode('ok');
    await t.client.ready();
    expect(t.workers()).toBe(2);
    const after = await s.value.advance(1);
    expect(after.ok ? '' : after.error.code).toBe('NotFound');
    expect(s.value.alive).toBe(false);
    t.client.dispose();
  });

  it('gives up after repeated crashes', async () => {
    const t = setup('ok');
    await t.client.ready();
    t.setMode('hang');
    for (let i = 0; i < 3; ++i) {
      await t.client.ready();
      await t.client.createSession(spec, config);
    }
    expect(t.client.status.state).toBe('unavailable');
    const r = await t.client.createSession(spec, config);
    expect(r.ok).toBe(false);
    t.client.dispose();
  });

  it('gives up at once when the engine cannot load', async () => {
    const { client } = setup('loadFails');
    expect((await client.ready()).ok).toBe(false);
    expect(client.status.state).toBe('unavailable');
    client.dispose();
  });

  it('acknowledges each frame once, after the page has handled it, and not after a pause', async () => {
    const t = setup('ok');
    await t.client.ready();
    const r = await t.client.createSession(spec, config);
    if (!r.ok) throw new Error('no session');
    const session = r.value;
    const seen: number[] = [];
    await session.run({ watch: { stats: [], tensors: [] }, stepsPerSecond: 60, onFrame: (f) => seen.push(f.seq), onStopped: () => {} });
    const w = t.worker();
    const acks = () => w.sent.filter((m) => m.type === 'ackFrame').map((m) => (m as unknown as { seq: number }).seq);
    const frame = (seq: number) => w.onmessage?.({ data: { type: 'frame', session: 1, seq, step: seq, loss: 0, gradNorm: 0, lr: 0,
      stepsPerSecond: 0, valueStats: {}, gradStats: {}, tensors: [] } } as unknown as MessageEvent<FromWorker>);
    const settle = () => new Promise((res) => setTimeout(res, 20));

    frame(1);
    expect(seen).toEqual([1]);
    expect(acks()).toEqual([]);   // not on receipt: on the next display frame
    await settle();
    expect(acks()).toEqual([1]);
    frame(2);
    await session.pause();
    await settle();
    expect(acks()).toEqual([1]);   // paused before the ack went out: no ack, no new frame
    t.client.dispose();
  });
});

