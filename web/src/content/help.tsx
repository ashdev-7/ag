// Every explanation the site shows, in one place so it can be read, checked
// against the engine, and kept consistent. Each claim here describes what the
// code does (engine/src, src/graph/presets.ts); if one changes, change both.

import type { ReactNode } from 'react';
import type { OptimConfig, PassKind } from '../engine/protocol.ts';

// ── the basics, for the landing page ────────────────────────────────────────

export const BASICS: { term: string; text: ReactNode }[] = [
  { term: 'Graph', text: <>A network is a chain of small operations (multiply, add, tanh…). Each box is one operation; arrows carry numbers from one to the next.</> },
  { term: 'Forward pass', text: <>Numbers flow left to right: inputs in, every box computes its output, and the last box gives the <b>loss</b>.</> },
  { term: 'Loss', text: <>One number saying how wrong the network is. Training means making it smaller.</> },
  { term: 'Gradient', text: <>For every number in the graph, <code>∂L/∂x</code>: how much the loss would change if that number were nudged up a little. Positive means “increasing x makes the loss worse”.</> },
  { term: 'Backward pass', text: <>Gradients flow right to left, from the loss back to every input, one arrow at a time, using the chain rule. This is what <code>loss.backward()</code> does in PyTorch.</> },
  { term: 'Optimizer', text: <>Uses the gradients to move each parameter a small step downhill, e.g. <code>w ← w − lr · ∂L/∂w</code>. Repeat many times and the network learns.</> },
];

export const MODE_INTRO: Record<'step' | 'train' | 'race', ReactNode> = {
  step: <>Slow motion. Run one forward and one backward pass one micro-step at a time, and read every number and every gradient as it is computed. Start here.</>,
  train: <>Real time. Train a small network on 2-d points and watch the decision boundary, the loss and each layer’s gradients change as it learns.</>,
  race: <>Four optimizers start from the same point on the same 2-parameter loss and race downhill, so their different behaviour is visible as paths.</>,
};

// ── stepping: passes and zoom ───────────────────────────────────────────────

export const PASS_HELP: Record<PassKind, { title: string; body: ReactNode }> = {
  forward: {
    title: 'Forward',
    body: <>Computes every value, inputs first, ending with the loss. No gradients. Each step fills in one box.</>,
  },
  reverse: {
    title: 'Backward (reverse mode)',
    body: <>The forward pass, then the gradient of the loss for every box. It starts at the loss with <code>∂L/∂L = 1</code> and walks back one arrow per step; each step multiplies the gradient arriving at a box by that box’s local derivative and adds the result to the input’s gradient (the chain rule). One pass gives every gradient.</>,
  },
  train: {
    title: 'Train step',
    body: <>One full iteration of a training loop: take the next batch of data, run forward and backward, then let the optimizer update every parameter (purple boxes) using its gradient. Stepping back undoes the update; <i>Next train step</i> keeps it and starts another. Repeat thousands of times and you have training.</>,
  },
  forwardMode: {
    title: 'Forward mode',
    body: <>The other way to get derivatives. Pick one input number (the <i>direction</i>), give it a “nudge” of 1 and every other input 0, and carry that nudge forward through the graph alongside the values. What reaches the loss is ∂L/∂(that one input). One pass per input number, so it is cheap for few inputs and expensive for many; reverse mode is the opposite, which is why training uses reverse mode.</>,
  },
};

export const ZOOM_HELP: Record<'tensor' | 'scalar', ReactNode> = {
  tensor: <><b>Tensors:</b> each box is a whole array (its shape in brackets); a box shows the mean <code>μ</code> and spread <code>σ</code> of its numbers. Click a box to see every number.</>,
  scalar: <><b>Single numbers:</b> every array is opened up so each box holds one number. A matrix multiply becomes its individual multiplies and adds, so you can follow one number all the way.</>,
};

export const GRAPH_LEGEND: ReactNode = (
  <>
    <span className="key-value">blue</span> value ·{' '}
    <span className="key-grad">∂</span> gradient of the loss with respect to that box (<span className="key-tan">ṫ</span> in forward mode: the carried nudge) ·
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

export const TRAIN_SETUP: ReactNode = (
  <>The data are 2-d points in two classes (blue and orange): 80% are used for training and 20% are held out to check the network on points it never trained on. The network ends in two numbers per point, one per class; softmax turns them into probabilities. Each training step takes the next batch of points, runs forward and backward, and updates every weight once.</>
);

export const TRAIN_GUIDES: Record<string, Guide> = {
  moons: {
    idea: <>Two interleaved half circles. A straight line cannot separate them; two tanh layers of 8 units bend the boundary around them.</>,
    watch: [<>Press Run: the boundary starts almost straight, then curls between the moons as the loss drops.</>, <>Tick <i>compare two optimizers</i> to train SGD and Adam from the same starting weights.</>],
  },
  circles: {
    idea: <>One class inside a ring of the other. ReLU units each cut the plane with a straight line; together they enclose the centre.</>,
    watch: [<>The boundary closes into a rough polygon: each corner is where one ReLU switches on.</>],
  },
  spirals: {
    idea: <>Two spiral arms wound around each other, the hardest of the four. It needs a wider network (two ReLU layers of 32) and more steps.</>,
    watch: [<>The loss can sit flat for a while before falling: the network is finding the arms one turn at a time.</>],
  },
  xor: {
    idea: <>Points in opposite corners share a class. A single neuron cannot do this (no straight line works); one hidden layer of 8 tanh units can.</>,
    watch: [<>Trained with momentum: the loss may overshoot and wobble before it settles.</>],
  },
  vanishing: {
    idea: <>Six sigmoid layers trained with plain SGD. A sigmoid’s slope is at most 0.25, and the backward pass multiplies by one slope per layer, so the gradient shrinks layer by layer on its way back.</>,
    watch: [<>In <i>Gradient norm per layer</i>, the first layer (dark) sits about four orders of magnitude below the output (orange).</>, <>The loss stays near 0.69, a coin flip, for thousands of steps: the early layers hardly learn.</>],
  },
  'dead-relu': {
    idea: <>The ReLU biases start at −1.5, so for most inputs the sum before the ReLU is negative and the output is exactly 0. A ReLU at 0 has slope 0, so no gradient flows back through it and it cannot recover.</>,
    watch: [<>In <i>Activations per layer</i>, both hidden layers are 100% zero: every unit is dead.</>, <>So every weight gradient is exactly 0, nothing but the output bias can learn, and the loss stays at 0.69, a coin flip.</>, <>Compare with <i>Circles</i>: the same data and network with ordinary starting weights.</>],
  },
};

export const PANEL_HELP = {
  network: <>The network as it trains. Each box shows the mean <code>μ</code> and spread <code>σ</code> of its values (blue) and gradients (orange) over the current batch. Thicker arrows carry larger gradients back. Click a purple weight box to look inside.</>,
  boundary: <>What the network predicts at every point of the square: orange where it says “orange class”, blue where it says “blue class”, pale where it is unsure; the boundary is where it is 50/50. Dots are the data, coloured by their true class.</>,
  loss: <>Cross-entropy: for each point, <code>−log(probability the network gives the correct class)</code>, averaged. 0 means certain and right; 0.69 (= ln 2) is a coin flip between two classes. <b>Solid:</b> the batch just trained on (jumpy, because every batch differs). <b>Dashed:</b> the held-out points, never trained on; if it rises while solid falls, the network is memorising.</>,
  gradNorm: <>For each layer, the size of the gradient of its weight matrix, <code>√(Σ (∂L/∂w)²)</code>, on the current batch. It shows which layers are actually being pushed to change. Layers are coloured from the first (dark) to the last (orange).</>,
  acts: <>For each layer, a histogram of its outputs over the current batch, lowest to highest value. The share of outputs that are exactly zero is in brackets; turns red above 90% (dead ReLUs).</>,
  inspect: <>The selected array as coloured cells (orange positive, blue negative, white zero): its values, its gradient, and the optimizer’s memory for it. The first moment is the running average of past gradients (momentum, Adam); the second moment is the running average of squared gradients (RMSProp, Adam). Plain SGD keeps neither.</>,
  controls: <><b>Run</b> trains continuously; <b>Step</b> does exactly one training step; <b>Reset</b> goes back to the initial weights. <b>Steps per frame</b> is how many training steps run between screen updates: more is faster. <b>lr</b> is the learning rate, the step size. <b>clip</b>: if the gradient of all weights together is longer than this, it is scaled down to this length before the update (0 = off).</>,
};

export const OPTIM_HELP: Record<OptimConfig['kind'], ReactNode> = {
  sgd: <><b>SGD:</b> step straight down the gradient, <code>w ← w − lr·g</code>.</>,
  momentum: <><b>Momentum:</b> keep a running velocity <code>v ← 0.9·v + g</code> and step along it, <code>w ← w − lr·v</code>. Speeds up along consistent directions, damps zig-zags.</>,
  nesterov: <><b>Nesterov:</b> momentum that looks one step ahead, <code>w ← w − lr·(g + 0.9·v)</code>; usually overshoots less.</>,
  rmsprop: <><b>RMSProp:</b> divide each step by a running average of recent gradient sizes, <code>w ← w − lr·g / √(avg g²)</code>, so steep and shallow directions move at similar speeds.</>,
  adam: <><b>Adam:</b> momentum and RMSProp together: a running average of gradients divided by the square root of a running average of squared gradients, both corrected for starting at zero.</>,
  adamw: <><b>AdamW:</b> Adam plus weight decay applied directly to the weights. Weight decay is 0 here, so it behaves like Adam.</>,
};

// ── the race ────────────────────────────────────────────────────────────────

export const RACE_GUIDES: Record<string, Guide & { formula: ReactNode }> = {
  rosenbrock: {
    formula: <code>f(x, y) = (1 − x)² + 100·(y − x²)²</code>,
    idea: <>A long, curved, flat-bottomed valley. The minimum is at (1, 1), where <code>f = 0</code>. Reaching the valley is easy; following its curve is slow, because the gradient points mostly across the valley, not along it.</>,
    watch: [<>SGD barely moves along the valley floor; momentum and Adam build speed along it.</>],
  },
  saddle: {
    formula: <code>f(x, y) = x² − y² + y⁴/4</code>,
    idea: <>A saddle at (0, 0): downhill in y, uphill in x. The real minima are at (0, ±1.41), where <code>f = −1</code>. Everyone starts with y almost exactly 0, so the push to escape the saddle is tiny at first.</>,
    watch: [<>Plain SGD drifts to the saddle and lingers; methods that rescale the gradient (RMSProp, Adam) amplify the tiny y-gradient and escape fast.</>],
  },
  bowl: {
    formula: <code>f(x, y) = 0.5·x² + 10·y²</code>,
    idea: <>A bowl 20 times steeper in y than in x, minimum at (0, 0). A step size large enough to make progress in x overshoots in y.</>,
    watch: [<>SGD zig-zags across the narrow direction; momentum averages the zig-zag out; RMSProp and Adam take similar-sized steps in both directions.</>],
  },
};

export const RACE_HELP = {
  surface: <>The loss <code>f(x, y)</code> computed at 100 × 100 points: darker is lower (log scale, so the valley floor stays visible). Each line is one optimizer’s path: the point (x, y) after every step. All four start at the same point; each runs at most 3000 steps.</>,
  loss: <>The loss at each optimizer’s current point. The fastest to the bottom wins. A racer marked “diverged” stepped somewhere the loss is not a finite number, and was stopped.</>,
};
