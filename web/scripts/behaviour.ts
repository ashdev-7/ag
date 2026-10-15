// Checks that every training lab and every race does what its text says, by
// running it in the engine exactly as the page does (same presets, same
// seed) and measuring.
//
//   node scripts/behaviour.ts        (after `make wasm` in engine/)
//
// The site's sentences are claims ("the test loss turns around", "sgd
// lingers on the pass", "by layer 1 the gradient is about 10,000× weaker").
// Each claim here is a measurement with a threshold; if a preset, the engine
// or the text changes, this says which claim stopped being true. It also
// checks the default speeds: what a lab shows must take long enough to watch.

import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createSession, loadEngine } from '../src/engine/bridge.ts';
import type { EngineSession } from '../src/engine/bridge.ts';
import { classifier, findPreset, trainPreset } from '../src/graph/presets.ts';
import type { Preset } from '../src/graph/presets.ts';
import { accuracy, testStory } from '../src/views/narrate.ts';

const here = dirname(fileURLToPath(import.meta.url));
const wasmDir = join(here, '../public/engine');
const engine = await loadEngine(pathToFileURL(join(wasmDir, 'ag.mjs')).href, (f) => join(wasmDir, f));
const DOMAIN = { x0: -1.5, x1: 1.5, y0: -1.5, y1: 1.5 };

let failed = 0, key = 1;
function check(ok: boolean, what: string, detail: string) {
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${what}  (${detail})`);
  if (!ok) failed++;
}
const f = (x: number) => (Math.abs(x) >= 0.01 || x === 0 ? x.toFixed(3) : x.toExponential(1));

// ── training labs ───────────────────────────────────────────────────────────

interface Trace { step: number[]; train: number[]; val: number[]; acc: { train: number; val: number }[]; grads: number[][]; zeros: number[][] }

/** Train to `until`, recording at `at` steps: whole-set losses, accuracy, each layer's gradient size. */
function trace(p: Preset, at: number[]): { trace: Trace; session: EngineSession } {
  const s = createSession(engine, p.spec, p.config, key++);
  const data = s.dataset();
  const t: Trace = { step: [], train: [], val: [], acc: [], grads: [], zeros: [] };
  for (const step of at) {
    const now = s.train(0).step;
    if (step > now) s.train(step - now);
    t.step.push(step);
    t.train.push(s.evalLoss('train'));
    t.val.push(s.evalLoss('val'));
    const grid = s.evalGrid({ ...DOMAIN, res: 60, nodeId: 'probs', element: 1 });
    t.acc.push(accuracy(grid, DOMAIN, data));
    const g = s.stats(p.layers.map((l) => l.weight), 'grad');
    t.grads.push(p.layers.map((l) => g[l.weight]?.l2 ?? 0));
    const v = s.stats(p.layers.slice(0, -1).map((l) => l.act), 'value');
    t.zeros.push(p.layers.slice(0, -1).map((l) => v[l.act]?.fracZero ?? 0));
  }
  return { trace: t, session: s };
}

const STEPS = [0, 25, 50, 100, 200, 400, 800, 1600, 3200];
const firstAt = (t: Trace, ok: (i: number) => boolean) => { const i = t.step.findIndex((_, k) => ok(k)); return i < 0 ? Infinity : t.step[i]; };

function lab(id: string, claims: (t: Trace, p: Preset) => void, at = STEPS) {
  const p = findPreset(id)!;
  console.log(`${id}  (opens at ${p.speed ?? 10} steps/s)`);
  const { trace: t, session } = trace(p, at);
  claims(t, p);
  session.free();
}

/** What the lab shows must not be over before a reader can see it. */
function watchable(p: Preset, step: number, what: string, minSeconds = 3) {
  const secs = step / (p.speed ?? 10);
  check(secs >= minSeconds && secs <= 120, `${what} takes ${minSeconds}-120 s at the default speed`, `step ${step} = ${secs.toFixed(1)} s`);
}

function learns(id: string, minAcc: number) {
  lab(id, (t, p) => {
    const end = t.acc.at(-1)!;
    check(end.val >= minAcc, `test accuracy reaches ${minAcc * 100}%`, `train ${f(end.train)}, test ${f(end.val)}`);
    const at90 = firstAt(t, (i) => t.acc[i].val >= 0.9);
    watchable(p, at90, 'reaching 90% test accuracy');
    // the caption under the loss chart: test follows training, no overfitting
    const story = testStory(t.val, t.train.at(-1)!, t.step, true);
    check(/learned the shape/.test(story), 'the caption says it learned the shape, not memorised',
      `train ${f(t.train.at(-1)!)}, test ${f(t.val.at(-1)!)}`);
  });
}

learns('moons', 0.97);
learns('circles', 0.95);
learns('xor', 0.95);
lab('spirals', (t, p) => {
  const end = t.acc.at(-1)!;
  check(end.val >= 0.9, 'test accuracy reaches 90%', `train ${f(end.train)}, test ${f(end.val)}`);
  // "for a while nothing improves, then the boundary suddenly starts tracing the arms"
  const early = t.acc[t.step.indexOf(100)].val;
  check(early < 0.75, 'little improvement early on (step 100 under 75%)', `test ${f(early)} at step 100`);
  watchable(p, firstAt(t, (i) => t.acc[i].val >= 0.9), 'reaching 90% test accuracy');
}, [0, 50, 100, 200, 400, 800, 1600, 3200, 6400]);

lab('vanishing', (t) => {
  // "the last layer gets about 10,000× more gradient than layer 1": the ratio
  // at every recorded step after the first
  const ratios = t.grads.slice(1).map((g) => g.at(-1)! / Math.max(g[0], 1e-30));
  const lo = Math.min(...ratios), hi = Math.max(...ratios);
  check(lo > 500 && hi < 3e4, 'layer 1 gets 700 to 10,000 times less gradient than the output (lessons, panel 4)',
    ratios.map((r) => r.toExponential(0)).join(' '));
  check(t.val.at(-1)! > 0.6, 'it does not learn: test loss stays near a coin flip (0.69)', `test loss ${f(t.val.at(-1)!)} at ${t.step.at(-1)}`);
});

lab('dead-relu', (t) => {
  const hidden = t.grads.map((g) => Math.max(...g.slice(0, -1)));
  check(hidden.slice(1).every((g) => g === 0), 'every hidden layer’s gradient is exactly zero', `largest ${Math.max(...hidden.slice(1))}`);
  check(Math.abs(t.val.at(-1)! - t.val[1]) < 0.05, 'the test loss does not move', `${f(t.val[1])} → ${f(t.val.at(-1)!)}`);
  check(t.zeros.every((z) => z.every((x) => x === 1)), 'both hidden layers are 100% zero (relu-init lesson)', t.zeros.at(-1)!.map(f).join(', '));
});

lab('spirals-small', (t) => {
  // capacity lesson: "the loss stops near 0.6 and accuracy around 60%,
  // and stays there"; "training and test loss stay together"
  const end = t.step.length - 1;
  check(t.train[end] > 0.5, 'four neurons never get the training loss down: stuck above 0.5', `train ${f(t.train[end])} at ${t.step[end]}`);
  check(t.acc[end].val > 0.5 && t.acc[end].val < 0.72, 'and accuracy stays around 60%', `test ${f(t.acc[end].val)}`);
  check(Math.abs(t.val[end] - t.train[end]) < 0.15, 'training and test loss stay together: underfitting, not overfitting', `train ${f(t.train[end])}, test ${f(t.val[end])}`);
}, [0, 400, 1600, 6400]);

// ── what the lessons say about the labs ─────────────────────────────────────

console.log('lesson claims');
{
  const run = (id: string, at: number[]) => { const r = trace(findPreset(id)!, at); r.session.free(); return r.trace; };
  // vanishing lesson: "in Moons the layers stay within about 20x of each other"
  const moons = run('moons', [50, 100, 200, 400, 800, 1600]);
  const spread = Math.max(...moons.grads.map((g) => Math.max(...g) / Math.min(...g)));
  check(spread < 20, 'Moons: every layer’s gradient within 20× of the others', `largest spread ${spread.toFixed(1)}×`);
  // relu-init lesson: "97% test accuracy within about 50 steps, about half of each layer's outputs zero"
  const circles = run('circles', [50, 400]);
  check(circles.acc[0].val >= 0.97, 'Circles: 97% test accuracy by step 50', `${f(circles.acc[0].val)}`);
  const z = circles.zeros.flat();
  check(z.every((x) => x > 0.3 && x < 0.7), 'Circles: about half of each hidden layer’s outputs are zero', z.map(f).join(', '));
  // non-linearity lesson: "within about 100 steps the boundary splits the square into opposite corners"
  const xor = run('xor', [100]);
  check(xor.acc[0].val >= 0.9, 'XOR: 90% test accuracy by step 100', `${f(xor.acc[0].val)}`);
  // "watch the loss wobble up and down before it falls"
  const p = findPreset('xor')!;
  const s = createSession(engine, p.spec, p.config, key++);
  const ls: number[] = [];
  for (let i = 0; i < 60; ++i) { s.train(5); ls.push(s.evalLoss('train')); }
  s.free();
  const rises = ls.slice(1).filter((l, i) => l > ls[i] * 1.02).length;
  check(rises >= 3 && ls.at(-1)! < ls[0] / 2, 'XOR: the loss rises several times before it falls', `${rises} rises in 300 steps, ${f(ls[0])} → ${f(ls.at(-1)!)}`);
  // capacity lesson: "two layers of 8 also get there, about four times slower; one layer of 4 never does"
  const sp = findPreset('spirals')!;
  const eight = trainPreset('spirals-8', '', '', 'spirals', classifier([8, 8], 'relu'), sp.config.optim, { batchSize: 32, n: 600, noise: 0.05 });
  const r8 = trace(eight, [1600, 6400]);
  r8.session.free();
  const r32 = run('spirals', [1600]);
  check(r32.acc[0].val >= 0.95 && r8.trace.acc[0].val < 0.95 && r8.trace.acc[1].val >= 0.95,
    'Spirals: 2 × 32 gets there by step 1,600; 2 × 8 only by 6,400',
    `32: ${f(r32.acc[0].val)} at 1,600; 8: ${f(r8.trace.acc[0].val)} at 1,600, ${f(r8.trace.acc[1].val)} at 6,400`);
  // modes lesson: "on the two-layer MLP the gap is 1 against 22"
  const mlp = findPreset('mlp')!;
  const ms = createSession(engine, mlp.spec, mlp.config, key++);
  const n = mlp.spec.nodes.filter((q) => q.op === 'param')
    .reduce((a, q) => a + ms.info.nodes.find((i) => i.id === q.id)!.shape.reduce((x, y) => x * y, 1), 0);
  ms.free();
  check(n === 22, 'two-layer MLP has 22 parameters (forward mode needs 22 passes)', `${n}`);
  // classification lesson's table: the push on the right class's score, two classes
  const push = (pc: number) => [4 * (1 - pc) * pc * (1 - pc), 1 - pc];
  const table: [number, number, number][] = [[0.5, 0.5, 0.5], [0.1, 0.32, 0.9], [0.01, 0.04, 0.99]];
  check(table.every(([pc, se, ce]) => Math.abs(push(pc)[0] - se) < 0.006 && Math.abs(push(pc)[1] - ce) < 0.006),
    'classification lesson: the squared-error and cross-entropy pushes in its table', table.map(([pc]) => push(pc).map(f).join('/')).join(', '));
  // non-linearity lesson: XOR by hand with two ReLUs
  const relu = (x: number) => Math.max(0, x);
  const ok = [[1, 1, 1], [-1, -1, 1], [1, -1, -1], [-1, 1, -1]].every(([a, b, cls]) => Math.sign(relu(a + b) + relu(-a - b) - 1) === cls);
  check(ok, 'non-linearity lesson: the two-neuron XOR solution gets all four corners right', '');
}


lab('overfit', (t, p) => {
  const low = Math.min(...t.val), lowAt = t.step[t.val.indexOf(low)];
  const at800 = t.val[t.step.indexOf(800)];
  check(t.train[t.step.indexOf(800)] < 0.05, 'training loss heads to 0', `train ${f(t.train[t.step.indexOf(800)])} at 800`);
  check(lowAt > 0 && lowAt <= 200, 'the test loss is lowest early on', `lowest ${f(low)} at step ${lowAt}`);
  check(at800 > 2 * low, 'then it climbs: test loss at step 800 is over twice its lowest', `${f(at800)} vs ${f(low)}`);
  check(t.acc.at(-1)!.train - t.acc.at(-1)!.val > 0.1, 'training accuracy well above test accuracy at the end',
    `train ${f(t.acc.at(-1)!.train)}, test ${f(t.acc.at(-1)!.val)}`);
  check(/overfitting/.test(testStory(t.val, t.train.at(-1)!, t.step)), 'the caption names overfitting', '');
  watchable(p, lowAt, 'reaching the lowest test loss', 2);
  watchable(p, 800, 'the climb to step 800');
});

lab('overfit-more', (t) => {
  const end = t.step.indexOf(800);
  check(t.val[end] < 0.45, 'with 8× the data the test loss stays low', `test ${f(t.val[end])} at 800`);
  const low = Math.min(...t.val.slice(0, end + 1));
  check(t.val[end] < 1.5 * low, 'and does not climb far from its lowest by step 800', `${f(t.val[end])} vs lowest ${f(low)}`);
  check(!/overfitting/.test(testStory(t.val.slice(0, end + 1), t.train[end], t.step)), 'the caption does not call it overfitting', '');
});

// ── races ───────────────────────────────────────────────────────────────────

interface Run { kind: string; loss: number[]; x: number[]; y: number[] }

function race(id: string, claims: (runs: Run[], a: Record<string, number>) => void) {
  const p = findPreset(id)!;
  const r = p.race!;
  console.log(`${id}  (race of ${r.maxSteps} steps at ${p.speed} steps/s)`);
  const runs = r.optimizers.map((optim) => {
    const s = createSession(engine, p.spec, { ...p.config, optim }, key++);
    const run: Run = { kind: optim.kind, loss: [], x: [], y: [] };
    for (let i = 0; i < r.maxSteps; ++i) {
      run.loss.push(s.train(1).loss);
      run.x.push(s.read(r.x, 'value').data[0]);
      run.y.push(s.read(r.y, 'value').data[0]);
    }
    s.free();
    return run;
  });
  const best = Math.min(...runs.flatMap((u) => u.loss)), start = runs[0].loss[0];
  // arrived: within 0.1% of the drop, the same rule as the race caption
  const arrive = (u: Run) => { const i = u.loss.findIndex((l) => l - best <= 0.001 * (start - best)); return i < 0 ? Infinity : i; };
  const arrivals = Object.fromEntries(runs.map((u) => [u.kind, arrive(u)]));
  console.log(`  arrivals: ${runs.map((u) => `${u.kind} ${arrivals[u.kind]}`).join(', ')}`);
  const speed = p.speed!;
  const finite = Object.values(arrivals).filter(Number.isFinite);
  const first = Math.min(...finite), last = Math.max(...finite);
  check(first / speed >= 1.5, 'the first arrival is not over in a blink (1.5 s or more)', `step ${first} = ${(first / speed).toFixed(1)} s`);
  check(r.maxSteps / speed <= 45, 'the whole race takes 45 s or less', `${(r.maxSteps / speed).toFixed(0)} s`);
  check(last <= r.maxSteps, 'the race lasts until the last arrival', `last ${last}, race ${r.maxSteps}`);
  claims(runs, arrivals);
}

race('rosenbrock', (runs, a) => {
  // momentum lesson: "all four reach the valley quickly"
  const floor = runs.map((u) => u.x.findIndex((x, i) => Math.abs(u.y[i] - x * x) < 0.05));
  check(floor.every((i) => i >= 0 && i < 20), 'all four reach the valley floor within 20 steps', floor.join(', '));
  check(a.momentum < a.sgd && a.adam < a.sgd, 'momentum and adam get to the bottom well before sgd', `sgd ${a.sgd}, momentum ${a.momentum}, adam ${a.adam}`);
  const sgd = runs.find((u) => u.kind === 'sgd')!;
  check(sgd.loss[999] > 1e-3, 'sgd is still crawling along the floor at step 1,000', `loss ${f(sgd.loss[999])}`);
});
race('saddle', (_, a) => {
  // "sgd lingers on the pass; rmsprop and adam turn off at once"
  check(a.sgd > 2 * a.rmsprop && a.sgd > 2 * a.adam, 'sgd takes over twice as long as rmsprop and adam', `sgd ${a.sgd}, rmsprop ${a.rmsprop}, adam ${a.adam}`);
});
race('bowl', (runs) => {
  // "sgd zig-zags; momentum averages the zig-zags out": count sign flips of the steep coordinate
  const flips = (u: Run) => u.y.slice(0, 30).reduce((n, y, i, ys) => n + (i > 0 && Math.sign(y) !== Math.sign(ys[i - 1]) ? 1 : 0), 0);
  const sgd = runs.find((u) => u.kind === 'sgd')!, mom = runs.find((u) => u.kind === 'momentum')!;
  check(flips(sgd) >= 10, 'sgd zig-zags: the steep coordinate flips sign at most steps', `${flips(sgd)} flips in 30 steps`);
  check(flips(mom) < flips(sgd) / 2, 'momentum zig-zags far less', `${flips(mom)} flips in 30 steps`);
});

// descent lesson: "set sgd's lr to 0.11 ... the race marks it diverged": its
// loss passes a thousand times its start (RaceView's rule) within the race
{
  const p = findPreset('bowl')!;
  const s = createSession(engine, p.spec, { ...p.config, optim: { kind: 'sgd', lr: 0.11 } }, key++);
  let at = -1, start = 0;
  for (let i = 0; i < p.race!.maxSteps && at < 0; ++i) {
    const l = s.train(1).loss;
    if (i === 0) start = l;
    if (l > start + 1000 * Math.max(1, Math.abs(start))) at = i;
  }
  s.free();
  check(at > 0, 'bowl: sgd at lr 0.11 diverges within the race', `marked at step ${at} of ${p.race!.maxSteps}`);
}

console.log(failed ? `=== ${failed} failed ===` : '=== all passed ===');

process.exit(failed ? 1 : 0);
