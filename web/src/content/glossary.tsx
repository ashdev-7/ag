// Plain definitions for every term the labs use. Shown when a reader points
// at (or tabs to) an underlined word, so nobody has to look anything up.

import type { ReactNode } from 'react';

export const GLOSSARY = {
  parameter: 'A number inside the network that training adjusts: the weights and biases. The “knobs”.',
  weight: 'A parameter that multiplies an input: how much that input counts, and in which direction.',
  bias: 'A parameter added after the weighted sum; it shifts where the neuron switches on.',
  activation: 'The bending function after each layer (tanh, sigmoid, ReLU). Without it, stacked layers collapse into one straight-line model.',
  layer: 'A group of neurons computed together; each layer’s outputs are the next layer’s inputs.',
  loss: 'One number saying how wrong the network is on the current data. Training makes it smaller; 0 is perfect.',
  'cross-entropy': 'The loss for choosing between classes: −log of the probability the network gave the correct class, averaged. 0.69 means 50/50 guessing.',
  gradient: 'For each number, how much the loss would change if that number were nudged up. Its sign says which way to turn a knob; its size, how much it matters right now.',
  'gradient norm': 'The overall size of a whole array of gradients: √(sum of squares). A big norm means that layer is being pushed hard to change.',
  backward: 'The backward pass: computing every gradient by sending them from the loss back toward the inputs (backpropagation).',
  'learning rate': 'How big a step the optimizer takes against the gradient. Too small: slow. Too large: overshoots and blows up.',
  optimizer: 'The rule that turns gradients into parameter updates, e.g. SGD: w ← w − lr · gradient.',
  batch: 'The small group of examples used for one training step. Different every step, which makes the training loss jumpy.',
  step: 'One training step: take a batch, forward pass, backward pass, update every parameter once.',
  'held-out': 'Examples kept out of training (here 20%). The network never learns from them, so they show whether it learned the pattern or memorised the training points.',
  'decision boundary': 'The line where the network’s prediction switches from one class to the other: 50/50 on the line.',
  probability: 'A number from 0 to 1 saying how sure the network is. The network outputs one per class, adding up to 1.',
  logits: 'The network’s raw scores, one per class, before softmax turns them into probabilities.',
  softmax: 'Turns raw scores into probabilities: exponentiate each, then divide by the total so they add to 1.',
  histogram: 'A bar chart of how many values fall in each range, from the lowest value (left) to the highest (right).',
  'dead unit': 'A ReLU neuron whose input is negative for every example: it always outputs 0 and passes no gradient back, so it can never recover.',
  momentum: 'Keeping part of the previous step’s direction, like a rolling ball, so consistent directions speed up and zig-zags cancel.',
  clip: 'Gradient clipping: if all gradients together are longer than this value, they are scaled down to it before the step. Prevents explosions. 0 means off.',
  tensor: 'An array of numbers with a shape: [] one number, [3] a list, [3, 2] a table.',
  'log scale': 'An axis where each gridline is 10× the one below, so tiny and large values fit on one chart. Equal distances mean equal ratios.',
  'local derivative': 'How much one box’s output changes when one of its inputs changes, ignoring the rest of the graph. For a × b it is b (with respect to a).',
  'chain rule': 'Multiply local derivatives along a path from the loss back to a number; add over all paths.',
  tangent: 'In forward mode: how much each number moves when the one chosen input is nudged by 1.',
} as const;

export type TermName = keyof typeof GLOSSARY;

/** A word with its definition one hover or one Tab away. */
export function Term(props: { t: TermName; children?: ReactNode }) {
  return (
    <span className="term" tabIndex={0} data-def={GLOSSARY[props.t]} aria-label={`${props.t}: ${GLOSSARY[props.t]}`}>
      {props.children ?? props.t}
    </span>
  );
}
