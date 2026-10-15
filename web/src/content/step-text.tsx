// What the stepping screen says, for someone on their first day. Everything
// here is visible without clicking anything, and kept short.

import type { ReactNode } from 'react';
import type { PassKind } from '../engine/protocol.ts';

/** The screen in one line; the key below the modes names everything drawn. */
export const STEP_SCREEN: ReactNode = (
  <>A calculation as boxes joined by arrows, run one piece of arithmetic at a time. Press <b>▶</b> or the <b>→</b> key; the side panel says what each step did.</>
);

/** One line per mode, always shown, so the choice makes sense before clicking. */
export const MODE_SHORT: Record<PassKind, { name: string; line: ReactNode }> = {
  forward: { name: 'Forward', line: <>Compute the answer, left to right.</> },
  reverse: { name: 'Backward', line: <>Then every gradient, right to left.</> },
  train: { name: 'Train step', line: <>Then nudge each parameter: one step of learning.</> },
  forwardMode: { name: 'Forward mode', line: <>Gradients the other way, one input per run.</> },
};

export const MODE_LONG = (lr: number): Record<PassKind, ReactNode> => ({
  forward: <>Each box computes its value from the boxes feeding it, until the last box gives the result: for a network, the <b>loss</b>.</>,
  reverse: <>From the result (∂ = 1), walk back along the arrows. Each arrow multiplies the gradient so far by the box’s <i>local slope</i> and adds it to the input’s gradient. That is the <b>chain rule</b>; the whole walk is <b>backpropagation</b>.</>,
  train: <>Backward, then each parameter (purple) moves against its gradient: <code>w ← w − {lr} × ∂</code>. Press <i>Next train step</i> at the end to repeat and watch the loss fall.</>,
  forwardMode: <>Nudge one input by 1 and carry the nudge (ṫ) forward with the values. What reaches the end is the gradient for that one input. One run per input, where Backward gets them all at once.</>,
});

export const ZOOM_SHORT: Record<'tensor' | 'scalar', { name: string; line: string }> = {
  tensor: { name: 'Tensors', line: 'Each box holds a table of numbers, shown as average ± spread.' },
  scalar: { name: 'Single numbers', line: 'Every table opened up, one box per number.' },
};
