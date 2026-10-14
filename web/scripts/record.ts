// Records what each training and race preset does, using the real engine
// (the WASM build, under node). The page shows these recordings when the
// engine cannot run: WebAssembly unavailable, or the worker crashed twice.
//
//   npm run record        (after `make wasm` in engine/)
//
// Also prints what each preset is meant to demonstrate, so a preset that
// stops demonstrating it is noticed here first.

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createSession, loadEngine } from '../src/engine/bridge.ts';
import { PRESETS } from '../src/graph/presets.ts';
import type { Preset } from '../src/graph/presets.ts';
import type { Engine } from '../src/engine/bridge.ts';

const here = dirname(fileURLToPath(import.meta.url));
const wasmDir = join(here, '../public/engine');
const outDir = join(here, '../public/recordings');
const TRAIN_STEPS = 2000;
const EVERY = 20;
const GRID = 50;

const round = (x: number, d = 4) => (Number.isFinite(x) ? Number(x.toFixed(d)) : null);

function recordTrain(e: Engine, p: Preset) {
  const s = createSession(e, p.spec, p.config, 1);
  const data = s.dataset();
  const loss: (number | null)[] = [];
  const val: (number | null)[] = [];
  const gradNorms: Record<string, (number | null)[]> = {};
  p.layers.forEach((l) => (gradNorms[l.label] = []));
  for (let step = 0; step < TRAIN_STEPS; step += EVERY) {
    const r = s.train(EVERY);
    if (r.fault) {
      console.log(`  ${p.id}: stopped at step ${r.step}: ${r.fault.message}`);
      break;
    }
    loss.push(round(r.loss));
    val.push(round(s.evalLoss('val')));
    const g = s.stats(p.layers.map((l) => l.weight), 'grad');
    p.layers.forEach((l) => gradNorms[l.label].push(round(g[l.weight]?.l2 ?? 0, 6)));
  }
  const grid = s.evalGrid({ x0: -1.5, x1: 1.5, y0: -1.5, y1: 1.5, res: GRID, nodeId: p.boundary!.nodeId, element: p.boundary!.element });
  const acts = s.stats(p.layers.map((l) => l.act), 'value');
  const deadFrac = p.layers.map((l) => round(acts[l.act]?.fracZero ?? 0, 3));
  const firstNorm = p.layers.map((l) => gradNorms[l.label][1]);
  console.log(`  ${p.id.padEnd(11)} final loss ${loss.at(-1)}  val ${val.at(-1)}  zeros per layer ${deadFrac.join(' ')}  early grad norm per layer ${firstNorm.join(' ')}`);
  return {
    every: EVERY, loss, val, gradNorms, deadFrac,
    grid: { res: GRID, data: Array.from(grid.data, (v) => round(v, 3)) },
    points: { x: Array.from(data.x, (v) => round(v, 3)), y: Array.from(data.y, (v) => Math.round(v)), xDim: data.xDim, yDim: data.yDim },
  };
}

function recordRace(e: Engine, p: Preset) {
  const race = p.race!;
  const res = 60;
  const first = createSession(e, p.spec, p.config, 1);
  const surface = first.lossSurface({ nodeId: race.x, element: 0 }, { nodeId: race.y, element: 0 }, { ...race.domain, res });
  const paths = race.optimizers.map((optim, i) => {
    const s = createSession(e, p.spec, { ...p.config, optim }, i + 2);
    const pts: [number, number][] = [];
    for (let k = 0; k < 300; ++k) {
      const xy = s.exportParams();
      pts.push([round(xy[0])!, round(xy[1])!]);
      if (s.train(5).fault) break;
    }
    s.free();
    const end = pts.at(-1)!;
    console.log(`  ${p.id.padEnd(11)} ${optim.kind.padEnd(8)} ends at (${end[0]}, ${end[1]})`);
    return { kind: optim.kind, points: pts };
  });
  first.free();
  return { surface: { res, data: Array.from(surface.data, (v) => round(v, 3)) }, paths };
}

const engine = await loadEngine(pathToFileURL(join(wasmDir, 'ag.mjs')).href, (f) => join(wasmDir, f));
mkdirSync(outDir, { recursive: true });
for (const p of PRESETS) {
  // every preset must at least build
  const s = createSession(engine, p.spec, p.config, 99);
  s.free();
  if (p.mode === 'step') {
    console.log(`  ${p.id.padEnd(11)} builds: ${s.info.nodes.length} nodes, reverse pass of ${s.info.schedules.reverse.length} steps`);
    continue;
  }
  const rec = p.mode === 'train' ? recordTrain(engine, p) : recordRace(engine, p);
  writeFileSync(join(outDir, `${p.id}.json`), JSON.stringify({ preset: p.id, ...rec }));
}
console.log(`wrote ${outDir}`);
