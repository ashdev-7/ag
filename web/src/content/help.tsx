// Every explanation the site shows, in one place so it can be read, checked
// against the engine, and kept consistent. Each claim here describes what the
// code does (engine/src, src/graph/presets.ts); if one changes, change both.

import type { ReactNode } from 'react';
import type { OptimConfig } from '../engine/protocol.ts';

// ── the three kinds of lab ─────────────────────────────────────────────────

export const MODE_INTRO: Record<'step' | 'train' | 'race', ReactNode> = {
  step: <>Slow motion: one calculation, every value and gradient, one step at a time. Start here.</>,
  train: <>Real time: a small network learns to sort coloured dots while you watch.</>,
  race: <>Four optimizers race downhill on the same landscape.</>,
};

// ── stepping: what is drawn ────────────────────────────────────────────────

/** The graph key: a swatch class (styles.css, .gk-*) and what it means. */
export const GRAPH_KEY: { swatch: string; label: ReactNode }[] = [
  { swatch: 'gk-value', label: <>value, flows along the arrows</> },
  { swatch: 'gk-grad', label: <><b>∂</b> gradient: how much a nudge here moves the result</> },
  { swatch: 'gk-param', label: <>parameter: learned</> },
  { swatch: 'gk-leaf', label: <>fixed input</> },
  { swatch: 'gk-active', label: <>just ran</> },
  { swatch: 'gk-stop', label: <>no gradient flows back</> },
];

export const MODES_HELP: ReactNode = (
  <>Both modes compute the same derivatives. Reverse mode gets them all in one backward pass; forward mode needs one pass per input number. Differences below 0.0001 are float32 rounding.</>
);

// ── stepping presets ────────────────────────────────────────────────────────

export interface Guide { idea: ReactNode; watch: ReactNode[] }

export const STEP_GUIDES: Record<string, Guide> = {
  expression: {
    idea: <>The smallest chain rule. With <code>a = 2</code>, <code>b = 3</code>: <code>c = a·b = 6</code>, <code>d = c + a = 8</code>. Here <code>d</code> plays the loss.</>,
    watch: [
      <>Pick Backward and press ⏭ (jump to the end): <code>∂d/∂b = 2</code> (that is <code>a</code>), <code>∂d/∂a = 4</code> (that is <code>b + 1</code>).</>,
      <><code>a</code> is used twice, so its gradient arrives in two pieces, 1 and 3, that add up.</>,
    ],
  },
  detach: {
    idea: <>The same sum, but one <code>a</code> passes through <code>detach</code>: its value flows forward, its gradient does not flow back. PyTorch’s <code>x.detach()</code>.</>,
    watch: [
      <>At the end, <code>∂d/∂a = 1</code> instead of 4: only the direct path counts. The dashed arrow marks where it stops.</>,
    ],
  },
  neuron: {
    idea: <>One neuron, <code>out = tanh(x·w + b)</code>, learning to output 0.5. The loss is <code>(out − 0.5)²</code>.</>,
    watch: [
      <>Backward: tanh scales the gradient by <code>1 − out²</code>; <code>w</code> gets <code>x</code> times that, <code>b</code> gets it as is.</>,
      <>Switch to <i>Train step</i>, press ⏭ (jump to the end), then <i>Next train step</i> a few times: the value in the <code>loss</code> box shrinks.</>,
    ],
  },
  broadcast: {
    idea: <>A bias <code>b</code> of 2 numbers is added to each of the 3 rows of <code>x</code> (<i>broadcasting</i>). The loss is the mean square of the result.</>,
    watch: [
      <>Each number of <code>b</code> was used in 3 rows, so its gradient is the sum of 3 pieces. Click <code>b</code> after Backward to see it.</>,
    ],
  },
  mlp: {
    idea: <>A real classifier, small enough to open up: 4 points, a layer of 4 tanh neurons, 2 outputs (one score per colour), cross-entropy loss.</>,
    watch: [
      <>Switch to <i>Single numbers</i>: about 240 boxes, every multiply and add visible.</>,
      <>In <i>Train step</i>, each run draws 4 new points and ends by updating all four weight arrays.</>,
    ],
  },
};

// ── optimizers ─────────────────────────────────────────────────────────────

/** One line each, shown next to the optimizer it describes. */
export const OPTIM_HELP: Record<OptimConfig['kind'], ReactNode> = {
  sgd: <>Plain gradient descent: a step of fixed size downhill, <code>w ← w − lr × ∂</code>.</>,
  momentum: <>Rolls like a ball: keeps part of its last step, so a steady direction speeds up.</>,
  nesterov: <>Momentum that looks one step ahead, so it overshoots less.</>,
  rmsprop: <>Each weight gets its own step size, so all of them move at a similar pace.</>,
  adam: <>Momentum plus a step size per weight. The usual default.</>,
  adamw: <>Adam plus a pull toward zero (weight decay); that pull is off here, so it equals adam.</>,
};

// ── the race ────────────────────────────────────────────────────────────────

export const RACE_GUIDES: Record<string, Guide & { formula: ReactNode; goals: [number, number][] }> = {
  rosenbrock: {
    formula: <code>f(x, y) = (1 − x)² + 100·(y − x²)²</code>,
    goals: [[1, 1]],
    idea: <>A long curved valley with an almost flat floor. Falling in is easy; following the floor is slow, because downhill points at the walls.</>,
    watch: [<><b>sgd</b> crawls along the floor; <b>momentum</b> and <b>adam</b> build up speed.</>],
  },
  saddle: {
    formula: <code>f(x, y) = x² − y² + y⁴/4</code>,
    goals: [[0, Math.SQRT2], [0, -Math.SQRT2]],
    idea: <>A mountain pass. The race starts almost on its centre line, where the slope toward the two low points is tiny.</>,
    watch: [<><b>sgd</b> lingers on the pass; <b>rmsprop</b> and <b>adam</b> amplify small slopes and turn off at once.</>],
  },
  bowl: {
    formula: <code>f(x, y) = 0.5·x² + 10·y²</code>,
    goals: [[0, 0]],
    idea: <>A bowl 20× steeper up-and-down than sideways. A step big enough for the sideways direction overshoots the steep one.</>,
    watch: [<><b>sgd</b> zig-zags; <b>momentum</b> averages the zig-zags out; <b>rmsprop</b> and <b>adam</b> even out the step sizes.</>],
  },
};

export const RACE_SCREEN: ReactNode = (
  <>Training is walking downhill on the loss. Here the loss has just two knobs, <b>x</b> and <b>y</b>, so the whole landscape fits in one picture.</>
);

export const RACE_HELP = {
  surface: <>Darker is lower. Each line is one optimizer’s path, a dot per step.</>,
  loss: <>Each optimizer’s height as it goes; the line that falls fastest is winning. Steps that blow up are stopped and marked “diverged”.</>,
};
