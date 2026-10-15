// The Phase 1 presets. Each is a GraphSpec plus what the views need to know
// about it: which nodes are layers, where the decision boundary is read, and
// for the race, which two parameters span the loss surface.

import type { Dim, GraphSpec, InitName, NodeSpec, OpName } from './spec.ts';
import type { OptimConfig, SessionConfig } from '../engine/protocol.ts';

export type PresetMode = 'step' | 'train' | 'race';

export interface Layer {
  label: string;
  /** Activation node: its value histogram and fraction of zeros. */
  act: string;
  /** Weight parameter: its gradient norm over time. */
  weight: string;
}

export interface Preset {
  id: string;
  title: string;
  /** One line under the title. */
  blurb: string;
  mode: PresetMode;
  spec: GraphSpec;
  config: SessionConfig;
  layers: Layer[];
  /** Decision boundary: node and element sampled over the input plane. */
  boundary?: { nodeId: string; element: number };
  race?: {
    x: string;
    y: string;
    domain: { x0: number; x1: number; y0: number; y1: number };
    optimizers: OptimConfig[];
  };
}

// ── a small builder ─────────────────────────────────────────────────────────

const node = (id: string, op: OpName, inputs: string[] = [], extra: Partial<NodeSpec> = {}): NodeSpec => ({ id, op, inputs, ...extra });
const param = (id: string, shape: Dim[], init: InitName, initScale = 1, module?: string): NodeSpec =>
  node(id, 'param', [], { shape, init, initScale, module });
const spec = (nodes: NodeSpec[], loss: string, outputs: string[] = []): GraphSpec => ({ version: 1, nodes, loss, outputs });

const noData = { kind: 'none', n: 0, noise: 0, valFraction: 0 } as const;

/** A classifier on 2-d points: x -> [dense -> act] * k -> dense -> logits; softmax cross-entropy. */
function classifier(hidden: number[], act: OpName, opts: { init?: InitName; initScale?: number; bias?: { init: InitName; scale: number } } = {}) {
  const nodes: NodeSpec[] = [
    node('x', 'input', [], { shape: ['B', 2], role: 'features', name: 'x' }),
    node('y', 'input', [], { shape: ['B', 2], role: 'targets', name: 'y' }),
  ];
  const layers: Layer[] = [];
  let h = 'x';
  let width = 2;
  const dims = [...hidden, 2];
  dims.forEach((out, i) => {
    const last = i === dims.length - 1;
    const k = i + 1;
    const module = last ? 'output' : `layer ${k}`;
    const init = opts.init ?? (act === 'relu' ? 'he_normal' : 'xavier_normal');
    nodes.push(param(`W${k}`, [width, out], init, opts.initScale ?? 1, module));
    nodes.push(param(`b${k}`, [out], opts.bias && !last ? opts.bias.init : 'zeros', opts.bias && !last ? opts.bias.scale : 1, module));
    nodes.push(node(`mm${k}`, 'matmul', [h, `W${k}`], { name: `${h}·W${k}`, module }));
    nodes.push(node(last ? 'logits' : `z${k}`, 'add', [`mm${k}`, `b${k}`], { module }));
    if (!last) {
      nodes.push(node(`h${k}`, act, [`z${k}`], { module, scalar: act === 'leaky_relu' ? 0.01 : undefined }));
      layers.push({ label: module, act: `h${k}`, weight: `W${k}` });
      h = `h${k}`;
    } else {
      layers.push({ label: 'output', act: 'logits', weight: `W${k}` });
    }
    width = out;
  });
  nodes.push(node('probs', 'softmax', ['logits'], { name: 'p' }));
  nodes.push(node('loss', 'softmax_cross_entropy', ['logits', 'y'], { name: 'loss' }));
  return { spec: spec(nodes, 'loss', ['probs']), layers };
}

function trainConfig(data: SessionConfig['data']['kind'], optim: OptimConfig, batchSize = 16, n = 400, noise = 0.1): SessionConfig {
  return { batchSize, evalBatchSize: 256, seed: 1, optim, data: { kind: data, n, noise, valFraction: 0.2 } };
}

function trainPreset(id: string, title: string, blurb: string, data: SessionConfig['data']['kind'],
  net: ReturnType<typeof classifier>, optim: OptimConfig, batchSize = 16, n = 400, noise = 0.1): Preset {
  return {
    id, title, blurb, mode: 'train', spec: net.spec, layers: net.layers,
    config: trainConfig(data, optim, batchSize, n, noise),
    boundary: { nodeId: 'probs', element: 1 },
  };
}

// ── stepping presets ────────────────────────────────────────────────────────

const expression: Preset = {
  id: 'expression',
  title: 'd = a·b + a',
  blurb: 'The chain rule in two operations.',
  mode: 'step',
  spec: spec([
    node('a', 'input', [], { shape: [], init: 'constant', initScale: 2, requiresGrad: true }),
    node('b', 'input', [], { shape: [], init: 'constant', initScale: 3, requiresGrad: true }),
    node('c', 'mul', ['a', 'b']),
    node('d', 'add', ['c', 'a']),
  ], 'd'),
  config: { batchSize: 1, evalBatchSize: 1, seed: 1, optim: { kind: 'sgd', lr: 0.1 }, data: noData },
  layers: [],
};

const stopGrad: Preset = {
  id: 'detach',
  title: 'd = detach(a)·b + a',
  blurb: 'The same sum with one gradient path cut.',
  mode: 'step',
  spec: spec([
    node('a', 'input', [], { shape: [], init: 'constant', initScale: 2, requiresGrad: true }),
    node('b', 'input', [], { shape: [], init: 'constant', initScale: 3, requiresGrad: true }),
    node('sa', 'detach', ['a'], { name: 'detach(a)' }),
    node('c', 'mul', ['sa', 'b']),
    node('d', 'add', ['c', 'a']),
  ], 'd'),
  config: { batchSize: 1, evalBatchSize: 1, seed: 1, optim: { kind: 'sgd', lr: 0.1 }, data: noData },
  layers: [],
};

const neuron: Preset = {
  id: 'neuron',
  title: 'One neuron',
  blurb: 'A single neuron learning one number.',
  mode: 'step',
  spec: spec([
    node('x', 'input', [], { shape: [1, 2], init: 'uniform', initScale: 1 }),
    param('w', [2, 1], 'normal', 1),
    param('b', [1], 'zeros'),
    node('xw', 'matmul', ['x', 'w'], { name: 'x·w' }),
    node('z', 'add', ['xw', 'b']),
    node('out', 'tanh', ['z']),
    node('t', 'input', [], { shape: [1, 1], init: 'constant', initScale: 0.5, name: 'target' }),
    node('loss', 'mse_loss', ['out', 't']),
  ], 'loss', ['out']),
  config: { batchSize: 1, evalBatchSize: 1, seed: 3, optim: { kind: 'sgd', lr: 0.5 }, data: noData },
  layers: [],
};

const broadcast: Preset = {
  id: 'broadcast',
  title: 'Broadcasting',
  blurb: 'One bias shared by every row.',
  mode: 'step',
  spec: spec([
    node('x', 'input', [], { shape: [3, 2], init: 'uniform', initScale: 1 }),
    param('b', [2], 'normal', 0.5),
    node('y', 'add', ['x', 'b'], { name: 'x + b' }),
    node('sq', 'mul', ['y', 'y'], { name: 'y²' }),
    node('loss', 'mean', ['sq']),
  ], 'loss'),
  config: { batchSize: 1, evalBatchSize: 1, seed: 2, optim: { kind: 'sgd', lr: 0.1 }, data: noData },
  layers: [],
};

const tinyMlp = classifier([4], 'tanh');
const mlpStep: Preset = {
  id: 'mlp',
  title: 'Two-layer MLP',
  blurb: 'A tiny classifier, opened up number by number.',
  mode: 'step',
  spec: tinyMlp.spec,
  layers: tinyMlp.layers,
  config: trainConfig('moons', { kind: 'sgd', lr: 0.3 }, 4),
  boundary: { nodeId: 'probs', element: 1 },
};

// ── training presets ────────────────────────────────────────────────────────

const moons = trainPreset('moons', 'Moons', 'Bend a boundary around two half-moons.',
  'moons', classifier([8, 8], 'tanh'), { kind: 'adam', lr: 0.02 });
const circles = trainPreset('circles', 'Circles', 'Draw a loop around the centre.',
  'circles', classifier([8, 8], 'relu'), { kind: 'adam', lr: 0.02 });
// tanh networks sit on a plateau here for thousands of steps; ReLU does not
const spirals = trainPreset('spirals', 'Spirals', 'The hardest shape: needs a wider network.',
  'spirals', classifier([32, 32], 'relu'), { kind: 'adam', lr: 0.01 }, 32, 600, 0.05);
const xor = trainPreset('xor', 'XOR', 'The puzzle no single line can solve.',
  'xor', classifier([8], 'tanh'), { kind: 'momentum', lr: 0.1, momentum: 0.9 });

const vanishing = trainPreset('vanishing', 'Vanishing gradients',
  'A deep network whose early layers cannot learn.',
  'moons', classifier([8, 8, 8, 8, 8, 8], 'sigmoid'), { kind: 'sgd', lr: 0.5 });
const deadRelu = trainPreset('dead-relu', 'Dead ReLUs',
  'Neurons stuck at zero, forever.',
  'circles', classifier([16, 16], 'relu', { init: 'normal', initScale: 0.3, bias: { init: 'constant', scale: -1.5 } }),
  { kind: 'sgd', lr: 0.1 });

// ── optimizer race ──────────────────────────────────────────────────────────
// x and y are two scalar parameters; the loss is a function of the plane

const racers: OptimConfig[] = [
  { kind: 'sgd', lr: 0.001 },
  { kind: 'momentum', lr: 0.001, momentum: 0.9 },
  { kind: 'rmsprop', lr: 0.01, beta2: 0.99 },
  { kind: 'adam', lr: 0.05 },
];

function racePreset(id: string, title: string, blurb: string, start: [number, number], body: NodeSpec[], loss: string,
  domain: NonNullable<Preset['race']>['domain'], optimizers: OptimConfig[]): Preset {
  return {
    id, title, blurb, mode: 'race',
    spec: spec([
      param('x', [], 'constant', start[0]),
      param('y', [], 'constant', start[1]),
      ...body,
    ], loss),
    config: { batchSize: 1, evalBatchSize: 1, seed: 1, optim: optimizers[0], data: noData },
    layers: [],
    race: { x: 'x', y: 'y', domain, optimizers },
  };
}

const c = (id: string, value: number) => node(id, 'const', [], { shape: [], scalar: value, name: String(value) });

const rosenbrock = racePreset('rosenbrock', 'Rosenbrock valley',
  'Easy to fall in, slow to follow.',
  [-1.5, 2],
  [
    c('one', 1), c('hundred', 100),
    node('u', 'sub', ['one', 'x'], { name: '1 − x' }),
    node('u2', 'pow', ['u'], { scalar: 2, name: '(1 − x)²' }),
    node('x2', 'pow', ['x'], { scalar: 2, name: 'x²' }),
    node('v', 'sub', ['y', 'x2'], { name: 'y − x²' }),
    node('v2', 'pow', ['v'], { scalar: 2 }),
    node('w', 'mul', ['hundred', 'v2']),
    node('f', 'add', ['u2', 'w'], { name: 'f' }),
  ], 'f', { x0: -2, x1: 2, y0: -1, y1: 3 }, racers);

const saddle = racePreset('saddle', 'Saddle point',
  'A mountain pass that traps slow optimizers.',
  [-1.8, 0.001],
  [
    c('quarter', 0.25),
    node('x2', 'pow', ['x'], { scalar: 2 }),
    node('y2', 'pow', ['y'], { scalar: 2 }),
    node('y4', 'pow', ['y'], { scalar: 4 }),
    node('q', 'mul', ['quarter', 'y4']),
    node('s', 'sub', ['x2', 'y2']),
    node('f', 'add', ['s', 'q'], { name: 'f' }),
  ], 'f', { x0: -2, x1: 2, y0: -2, y1: 2 },
  [{ kind: 'sgd', lr: 0.01 }, { kind: 'momentum', lr: 0.01, momentum: 0.9 }, { kind: 'rmsprop', lr: 0.01, beta2: 0.99 }, { kind: 'adam', lr: 0.02 }]);

const bowl = racePreset('bowl', 'Narrow bowl',
  'Steep one way, shallow the other: zig-zags.',
  [-1.8, 0.9],
  [
    c('half', 0.5), c('ten', 10),
    node('x2', 'pow', ['x'], { scalar: 2 }),
    node('y2', 'pow', ['y'], { scalar: 2 }),
    node('a', 'mul', ['half', 'x2']),
    node('b2', 'mul', ['ten', 'y2']),
    node('f', 'add', ['a', 'b2'], { name: 'f' }),
  ], 'f', { x0: -2, x1: 2, y0: -1.2, y1: 1.2 },
  [{ kind: 'sgd', lr: 0.09 }, { kind: 'momentum', lr: 0.02, momentum: 0.9 }, { kind: 'rmsprop', lr: 0.02, beta2: 0.99 }, { kind: 'adam', lr: 0.05 }]);

export const PRESETS: Preset[] = [
  expression, stopGrad, neuron, broadcast, mlpStep,
  moons, circles, spirals, xor, vanishing, deadRelu,
  rosenbrock, saddle, bowl,
];

export function findPreset(id: string | undefined): Preset | undefined {
  return PRESETS.find((p) => p.id === id);
}
