// What the stepping screen says, for someone on their first day. Everything
// here is visible without clicking anything.

import type { ReactNode } from 'react';
import type { PassKind } from '../engine/protocol.ts';

export const STEP_SCREEN: ReactNode = (
  <>
    This screen shows a calculation as <b>boxes joined by arrows</b> (computer scientists call this a <i>graph</i>). Every box does one small piece of arithmetic.
    <span className="key-value"> Blue numbers</span> are the values, flowing left to right.
    <span className="key-grad"> Orange numbers marked ∂</span> are the <b>gradients</b>: for each box, how much the final result would change if that box’s number were nudged up a little. They are worked out right to left.
    Gradients are how a network learns: they say which way to turn each adjustable number to make the result better.
    Press <b>▶</b> (or the → key) to run one piece of arithmetic at a time; the panel on the right explains each one in words.
  </>
);

/** One line per mode, always shown, so the choice makes sense before clicking. */
export const MODE_SHORT: Record<PassKind, { name: string; line: ReactNode }> = {
  forward: { name: 'Forward', line: <>compute the answer, left to right</> },
  reverse: { name: 'Backward', line: <>compute the answer, then work out every gradient, right to left</> },
  train: { name: 'Train step', line: <>backward, then nudge every adjustable number to make the result better: one step of learning</> },
  forwardMode: { name: 'Forward mode', line: <>another way to get the same gradients, one input at a time</> },
};

export const MODE_LONG = (lr: number): Record<PassKind, ReactNode> => ({
  forward: <>Each box computes its value from the boxes feeding into it, until the last box gives the result (for a network, the <b>loss</b>: how wrong it is).</>,
  reverse: <>First the forward pass. Then, starting from the result with ∂ = 1, the computer walks back along the arrows one at a time. At each arrow it multiplies the gradient it has by how much that box’s output moves when its input moves (its <i>local slope</i>) and adds the product to the input’s gradient. This rule is the <b>chain rule</b>, and this whole procedure is <b>backpropagation</b>: how every neural network learns.</>,
  train: <>Everything in Backward, then one more kind of step: each adjustable number (a <b>parameter</b>, the purple boxes) is moved a little against its gradient, <code>w ← w − lr × ∂</code>, where lr, the <i>learning rate</i>, is a small step size (here {lr}), so the result gets better. That is one step of training. Press <i>Next train step</i> at the end to do another and watch the result improve. (Examples whose inputs are all fixed, like <code>d = a·b + a</code>, have nothing to adjust.)</>,
  forwardMode: <>Choose one input (the <i>direction</i>). Give it a nudge of 1 and every other input 0, then carry that nudge forward box by box alongside the values (the numbers marked ṫ). What reaches the end is the gradient for that one input only. One run per input, where Backward gets all of them in one run, which is why training uses Backward.</>,
});

export const ZOOM_SHORT: Record<'tensor' | 'scalar', { name: string; line: ReactNode }> = {
  tensor: { name: 'Tensors', line: <>each box holds a whole table of numbers; it shows their average ± spread</> },
  scalar: { name: 'Single numbers', line: <>every table opened up, one box per number</> },
};
