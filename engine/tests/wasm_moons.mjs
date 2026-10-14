// Runs the moons check through the WebAssembly build and the same bridge the
// browser worker uses, then compares with the native run.
//
//   make test && make wasm && node tests/wasm_moons.mjs
//
// The native test writes its batch losses to build/cases.json ("moons"). The
// two builds use different maths libraries (exp, tanh), so they agree to a
// tolerance, not bit for bit. Exits non-zero on disagreement.

import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, join } from 'node:path';
import { loadEngine, createSession } from '../../web/src/engine/bridge.ts';

const here = dirname(fileURLToPath(import.meta.url));
const wasmDir = join(here, '../../web/public/engine');
const moduleUrl = pathToFileURL(join(wasmDir, 'ag.mjs')).href;

let failed = 0;
const check = (ok, label) => {
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${label}`);
  if (!ok) failed++;
};

// same node order as moonsMlp(16) in tests/test_engine.cpp, so every
// parameter draws from the same rng stream in both builds
const spec = {
  version: 1,
  nodes: [
    { id: 'x', op: 'input', inputs: [], shape: ['B', 2], role: 'features' },
    { id: 'y', op: 'input', inputs: [], shape: ['B', 2], role: 'targets' },
    { id: 'W1', op: 'param', inputs: [], shape: [2, 16], init: 'xavier_normal' },
    { id: 'b1', op: 'param', inputs: [], shape: [16], init: 'zeros' },
    { id: 'W2', op: 'param', inputs: [], shape: [16, 2], init: 'xavier_normal' },
    { id: 'b2', op: 'param', inputs: [], shape: [2], init: 'zeros' },
    { id: 'xW1', op: 'matmul', inputs: ['x', 'W1'] },
    { id: 'z1', op: 'add', inputs: ['xW1', 'b1'] },
    { id: 'h', op: 'tanh', inputs: ['z1'] },
    { id: 'hW2', op: 'matmul', inputs: ['h', 'W2'] },
    { id: 'logits', op: 'add', inputs: ['hW2', 'b2'] },
    { id: 'probs', op: 'softmax', inputs: ['logits'] },
    { id: 'loss', op: 'softmax_cross_entropy', inputs: ['logits', 'y'] },
  ],
  loss: 'loss',
  outputs: ['probs'],
};

const config = {
  batchSize: 32, evalBatchSize: 128, seed: 42,
  optim: { kind: 'sgd', lr: 0.5 },
  data: { kind: 'moons', n: 400, noise: 0.1, valFraction: 0.2 },
};

console.log('=== wasm moons ===');
const engine = await loadEngine(moduleUrl, (file) => join(wasmDir, file));
const s = createSession(engine, spec, config, 1);
check(s.info.schedules.train.length > 0, `session built: ${s.info.nodes.length} nodes, train pass of ${s.info.schedules.train.length} steps`);

const losses = [];
const t0 = performance.now();
for (let k = 0; k < 6; ++k) {
  const r = s.train(250);
  losses.push(r.loss);
  console.log(`  step ${String(r.step).padStart(5)}  batch loss ${r.loss.toFixed(4)}`);
}
const ms = performance.now() - t0;
console.log(`  1500 steps in ${ms.toFixed(0)} ms (${((ms * 1000) / 1500).toFixed(1)} us/step, batch 32, hidden 16)`);
const val = s.evalLoss('val');
check(val < 0.15, `val loss ${val.toFixed(4)} < 0.15`);

const native = join(here, '../build/cases.json');
if (existsSync(native)) {
  const ref = JSON.parse(readFileSync(native, 'utf8')).moons;
  const worst = Math.max(...ref.map((v, i) => Math.abs(v - losses[i]) / Math.max(1, Math.abs(v))));
  check(worst < 1e-2, `native and wasm batch losses agree within 1e-2 (worst ${worst.toExponential(1)})`);
} else {
  console.log('  skip  native comparison: run `make test` first');
}

// a fault comes back as data and leaves the session usable
const bad = createSession(engine, {
  version: 1,
  nodes: [
    { id: 'p', op: 'param', inputs: [], shape: [3], init: 'constant', initScale: -1 },
    { id: 'l', op: 'log', inputs: ['p'] },
    { id: 'loss', op: 'sum', inputs: ['l'] },
  ],
  loss: 'loss', outputs: [],
}, { ...config, batchSize: 1, data: { kind: 'none', n: 0, noise: 0, valFraction: 0 } }, 2);
const r = bad.train(3);
check(r.fault?.code === 'NonFinite' && r.fault.nodeId === 'l', `nan reported at the spec id "l": ${r.fault?.message}`);

try {
  createSession(engine, { ...spec, nodes: spec.nodes.map((n) => (n.id === 'W2' ? { ...n, shape: [5, 2] } : n)) }, config, 3);
  check(false, 'a shape error is reported');
} catch (e) {
  check(e.error?.code === 'ShapeMismatch' && e.error.nodeId === 'hW2' && e.error.slot === 1,
    `a shape error names the spec node and input: ${e.error?.nodeId} input ${e.error?.slot}: ${e.message}`);
}

console.log(`=== ${failed ? `${failed} failed` : 'all passed'} ===`);
process.exit(failed ? 1 : 0);
