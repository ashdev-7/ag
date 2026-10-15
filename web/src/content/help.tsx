// Every explanation the site shows, in one place so it can be read, checked
// against the engine, and kept consistent. Each claim here describes what the
// code does (engine/src, src/graph/presets.ts); if one changes, change both.

import type { ReactNode } from 'react';
import type { OptimConfig } from '../engine/protocol.ts';
import { Term } from './glossary.tsx';

// ── the three kinds of lab ─────────────────────────────────────────────────

export const MODE_INTRO: Record<'step' | 'train' | 'race', ReactNode> = {
  step: <>Slow motion. Run one forward and one backward pass one micro-step at a time, and read every number and every gradient as it is computed. Start here.</>,
  train: <>Real time. Train a small network on 2-d points and watch the decision boundary, the loss and each layer’s gradients change as it learns.</>,
  race: <>Four optimizers start from the same point on the same 2-parameter loss and race downhill, so their different behaviour is visible as paths.</>,
};

// ── stepping: passes and zoom ───────────────────────────────────────────────

export const GRAPH_LEGEND: ReactNode = (
  <>
    <span className="key-value">blue</span> value ·{' '}
    <span className="key-grad">∂</span> <Term t="gradient" /> of the loss with respect to that box (<span className="key-tan">ṫ</span> in forward mode: the <Term t="tangent" />) ·
    purple box: a parameter the optimizer learns · grey box: a fixed input ·
    thick outline: the step that just ran · dashed outline: the box you selected · faded: not computed yet · dashed arrow: no gradient flows back through it
  </>
);

export const MODES_HELP: ReactNode = (
  <>Both modes compute the same derivatives of the loss. Reverse mode got all of them in one backward pass; forward mode needed one pass per input number. The table shows each derivative from both; the difference is <code>|reverse − forward| / max(1, |reverse|)</code>, and anything below 0.0001 is float32 rounding.</>
);

// ── stepping presets ────────────────────────────────────────────────────────

export interface Guide { idea: ReactNode; watch: ReactNode[] }

export const STEP_GUIDES: Record<string, Guide> = {
  expression: {
    idea: <>The smallest example of the chain rule. With <code>a = 2</code>, <code>b = 3</code>: forward gives <code>c = a·b = 6</code> and <code>d = c + a = 8</code>. Here <code>d</code> plays the role of the loss. Backward asks how <code>d</code> changes with <code>a</code> and <code>b</code>.</>,
    watch: [
      <>Step to the end of the backward pass: <code>∂d/∂b = 2</code> (that is <code>a</code>) and <code>∂d/∂a = 4</code> (that is <code>b + 1</code>).</>,
      <><code>a</code> is used twice, so its gradient arrives in two pieces: 1 directly from <code>d = c + a</code>, then 3 through <code>c = a·b</code>. Gradients from different uses add up.</>,
      <>Open <i>Reverse vs forward mode</i>: both give 4 and 2, but forward mode needed two passes.</>,
    ],
  },
  detach: {
    idea: <>The same expression, but the <code>a</code> going into the multiply passes through <code>detach</code>. Its value still flows forward unchanged (so <code>d</code> is still 8), but no gradient flows back through it, as if it were a constant. PyTorch’s <code>x.detach()</code> does exactly this.</>,
    watch: [
      <>At the end, <code>∂d/∂a = 1</code> instead of 4: only the direct path <code>d = c + a</code> counts. <code>b</code> still gets 2.</>,
      <>The dashed arrow into <code>detach(a)</code> marks where the gradient stops.</>,
    ],
  },
  neuron: {
    idea: <>One neuron: <code>out = tanh(x·w + b)</code>, with <code>x</code> a fixed input of 2 numbers, weights <code>w</code> and bias <code>b</code> to learn, and the loss <code>(out − 0.5)²</code>, the squared distance from a target of 0.5.</>,
    watch: [
      <>Backward: <code>∂L/∂out = 2(out − 0.5)</code>, then tanh multiplies it by <code>1 − out²</code>, then <code>w</code> gets <code>x</code> times that and <code>b</code> gets it unchanged.</>,
      <>Switch to <i>Train step</i> and step to the end: the last steps update <code>w</code> and <code>b</code> by <code>w ← w − 0.5 · ∂L/∂w</code>. Press <i>Next train step</i> a few times: the loss of each step is listed, and it shrinks as <code>out</code> moves toward 0.5.</>,
    ],
  },
  broadcast: {
    idea: <>A bias <code>b</code> of 2 numbers is added to each of the 3 rows of <code>x</code> (shape [3, 2]); this is <i>broadcasting</i>. The loss is the mean of the squares of the result.</>,
    watch: [
      <>Going back, the add sends a [3, 2] gradient toward <code>b</code>, but <code>b</code> has only 2 numbers. Because each <code>b</code> number was used in 3 rows, its gradient is the sum over those rows.</>,
      <>Click <code>b</code> after the backward pass and compare it with the column sums of <code>x + b</code>’s gradient.</>,
    ],
  },
  mlp: {
    idea: <>A real classifier, small enough to open up: 4 points from the moons dataset (2 coordinates each) go through a layer of 4 tanh units and a layer of 2 outputs (the <i>logits</i>, one per class). The loss is cross-entropy against the true classes. <code>p</code> is the softmax of the logits, the predicted probabilities, shown for reading but not used by the loss.</>,
    watch: [
      <>Switch to <i>Single numbers</i>: the same graph becomes about 240 one-number boxes, each multiply and add of the matrix products visible.</>,
      <>In <i>Train step</i>, each pass draws a new batch of 4 points and ends with an SGD update of all four parameter arrays.</>,
      <>Gradients reach <code>W1</code> only through <code>W2</code> and the tanh: compare their sizes.</>,
    ],
  },
};

// ── training ────────────────────────────────────────────────────────────────

export const OPTIM_HELP: Record<OptimConfig['kind'], ReactNode> = {
  sgd: <><b>sgd</b> (plain gradient descent): every step, move each weight against its correction by a fixed amount, <code>w ← w − lr × gradient</code>. Simple, but slow on flat ground and jittery in narrow valleys.</>,
  momentum: <><b>momentum</b>: like a ball rolling downhill, it keeps part of its previous direction. Steps that keep pointing the same way build up speed; steps that flip back and forth cancel out.</>,
  nesterov: <><b>nesterov</b>: momentum that looks one step ahead before correcting, so it overshoots less.</>,
  rmsprop: <><b>rmsprop</b>: gives every weight its own step size. A weight whose corrections have recently been large takes smaller steps, one whose corrections are tiny takes bigger steps, so all of them move at a similar pace.</>,
  adam: <><b>adam</b>: momentum and rmsprop combined: it keeps a running direction like momentum and an own step size per weight like rmsprop. The most popular default, because it works well without much tuning.</>,
  adamw: <><b>adamw</b>: adam plus a gentle pull of every weight toward zero (“weight decay”). That pull is off here, so it behaves exactly like adam.</>,
};

// ── the race ────────────────────────────────────────────────────────────────

export const RACE_GUIDES: Record<string, Guide & { formula: ReactNode; goals: [number, number][] }> = {
  rosenbrock: {
    formula: <code>f(x, y) = (1 − x)² + 100·(y − x²)²</code>,
    goals: [[1, 1]],
    idea: <>A long, curved valley with a nearly flat floor; its lowest point is at (1, 1). Getting into the valley is easy. Following its gentle floor to the lowest point is slow, because the downhill direction points mostly at the steep walls, not along the floor.</>,
    watch: [<><b>sgd</b> drops into the valley and then crawls along the floor; <b>momentum</b> and <b>adam</b> build up speed along it and get much further.</>],
  },
  saddle: {
    formula: <code>f(x, y) = x² − y² + y⁴/4</code>,
    goals: [[0, Math.SQRT2], [0, -Math.SQRT2]],
    idea: <>A saddle, like a mountain pass: at the centre (0, 0) the ground rises to the left and right but falls toward the top and bottom of the picture. The two lowest points are above and below the centre. The race starts almost exactly on the pass’s centre line, where the sideways slope is tiny, so it is hard to notice the way down.</>,
    watch: [<><b>sgd</b> walks to the centre of the pass and lingers there (the flat stretch in its loss line) before it finally slides off. <b>rmsprop</b> and <b>adam</b> scale up small slopes, so they turn off toward a lowest point almost at once.</>],
  },
  bowl: {
    formula: <code>f(x, y) = 0.5·x² + 10·y²</code>,
    goals: [[0, 0]],
    idea: <>A bowl that is 20 times steeper up-and-down than left-to-right, with its lowest point in the middle. A step big enough to make progress left-to-right overshoots in the steep direction.</>,
    watch: [<><b>sgd</b> zig-zags up and down across the bowl; <b>momentum</b> averages the zig-zags out; <b>rmsprop</b> and <b>adam</b> take similar-sized steps in both directions.</>],
  },
};

export const RACE_SCREEN: ReactNode = (
  <>
    Training a network means walking downhill on its <b>loss</b> (how wrong it is). A real network has thousands of adjustable numbers, so its “hills” cannot be drawn; here there are just two, <b>x</b> and <b>y</b>, so the whole landscape fits in one picture. Darker means lower loss. All four <b>optimizers</b> (rules for choosing each step downhill) start at the same point, marked ○, and try to reach the lowest point, marked ★. Each coloured line is the path one of them takes.
  </>
);

export const RACE_HELP = {
  surface: <>The landscape: for every point (x, y), how high the loss is. Darker = lower, lighter = higher. Each coloured line is one optimizer’s path, one dot per step; each runs for at most 3,000 steps.</>,
  loss: <>Each optimizer’s height (loss) as it goes. Lower is better; the line that drops fastest is winning. An optimizer that takes steps so large that the numbers blow up is stopped and marked “diverged”.</>,
};
