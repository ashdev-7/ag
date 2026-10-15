// Live captions: what a chart is showing right now, in words. Each function
// reads the same numbers the chart draws and says what they mean, so a
// beginner never has to interpret a curve alone.

import type { Dataset } from '../engine/protocol.ts';
import { fmt } from './draw.ts';

export const COIN_FLIP = Math.LN2;   // cross-entropy of a 50/50 guess between two classes

export interface Domain { x0: number; x1: number; y0: number; y1: number }

/**
 * Share of training and held-out points on the correct side of the boundary,
 * read from the decision grid (probability of class 1 per cell, row 0 at the
 * top, as the engine sweeps it). Approximate within one grid cell.
 */
export function accuracy(grid: { res: number; data: ArrayLike<number> }, d: Domain, data: Dataset): { train: number; val: number } {
  const dx = (d.x1 - d.x0) / grid.res, dy = (d.y1 - d.y0) / grid.res;
  let trainOk = 0, valOk = 0;
  for (let i = 0; i < data.n; ++i) {
    const px = data.x[i * data.xDim], py = data.x[i * data.xDim + 1];
    const col = Math.min(grid.res - 1, Math.max(0, Math.floor((px - d.x0) / dx)));
    const row = Math.min(grid.res - 1, Math.max(0, Math.floor((d.y1 - py) / dy)));
    const predicted = grid.data[row * grid.res + col] > 0.5;
    const actual = data.y[i * data.yDim + 1] > 0.5;
    if (predicted === actual) {
      if (i < data.nTrain) trainOk++;
      else valOk++;
    }
  }
  return { train: trainOk / Math.max(1, data.nTrain), val: valOk / Math.max(1, data.n - data.nTrain) };
}

export type Trend = 'falling' | 'flat' | 'rising' | 'early';

/** Compare the last quarter of a series with the quarter before it, on a log scale. */
export function trend(ys: (number | null)[]): Trend {
  const v = ys.filter((y): y is number => y !== null && Number.isFinite(y) && y > 0);
  if (v.length < 12) return 'early';
  const q = Math.floor(v.length / 4);
  const mean = (a: number[]) => a.reduce((s, x) => s + Math.log(x), 0) / a.length;
  const r = Math.exp(mean(v.slice(-q)) - mean(v.slice(-2 * q, -q)));
  return r < 0.9 ? 'falling' : r > 1.1 ? 'rising' : 'flat';
}

const pct = (x: number) => `${Math.round(x * 100)}%`;

/** `spread`: highest minus lowest predicted probability over the square. */
export function boundaryStory(acc: { train: number; val: number } | null, step: number, spread = 1): string {
  if (step === 0 || !acc) return 'Not trained yet: these colours are the untrained network’s guesses, so they mean nothing yet. Press Run and watch them reshape to fit the dots.';
  const base = `About ${pct(acc.train)} of training points and ${pct(acc.val)} of held-out points are on the correct side.`;
  if (spread < 0.2) return `${base} The square is almost one colour: the network gives nearly the same answer for every point, so it has not learned to tell them apart.`;
  if (acc.train < 0.65) return `${base} That is close to guessing (50%): the network has not found the pattern.`;
  if (acc.train > 0.97) return `${base} The boundary now follows the shape of the data.`;
  return `${base} The boundary is still bending toward the shape of the data.`;
}

export function lossStory(loss: number[], val: (number | null)[], step: number): string {
  if (step === 0 || !loss.length) return `Not trained yet. A network that guesses 50/50 scores ${fmt(COIN_FLIP)}; lower is better, 0 is perfect.`;
  const cur = loss[loss.length - 1], first = loss[0];
  const t = trend(loss);
  const moving = t === 'falling' ? 'It is still falling: the network is still learning.'
    : t === 'rising' ? 'It is rising: the steps may be too large (try a smaller lr).'
    : t === 'flat' ? 'It has levelled off.' : '';
  const level = cur > 0.6 ? `That is still about a coin flip (${fmt(COIN_FLIP)}): the network has learned little so far.`
    : cur < 0.05 ? 'That is low: the network is confident and right on these points.'
    : `That is better than a coin flip (${fmt(COIN_FLIP)}).`;
  let s = `Loss ${fmt(cur)}, from ${fmt(first)} at the start. ${level} ${moving}`;
  const v = val.filter((x): x is number => x !== null).at(-1);
  if (v !== undefined && v > 0.1 && v > 2 * Math.max(cur, 0.01))
    s += ` The held-out loss (${fmt(v)}) is well above the training loss: the network may be memorising its training points.`;
  return s.trim();
}

export function gradStory(layers: { label: string; norm: number }[], history: number[][], lossFalling = false): string {
  if (!layers.length) return '';
  if (layers.every((l) => !(l.norm > 1e-12))) return 'Every layer’s gradient is zero: no weight can change, so the network cannot learn at all.';
  const first = layers[0], last = layers[layers.length - 1];
  const ratio = first.norm / Math.max(last.norm, 1e-30);
  const times = (r: number) => {
    const p = Math.round(Math.log10(r));
    return p >= 3 ? `about ${'1' + '0'.repeat(p)}×`.replace(/(\d)(?=(\d{3})+×)/g, '$1,') : `about ${Math.round(r)}×`;
  };
  let s: string;
  if (ratio < 0.01) s = `The first layer gets ${times(1 / ratio)} less gradient than the last, so it is barely learning: the gradient vanishes on its way back.`;
  else if (ratio > 100) s = `The first layer gets ${times(ratio)} more gradient than the last: the gradient grows on its way back and may explode.`;
  else s = `All layers get gradients of a similar size (within ${times(Math.max(ratio, 1 / ratio))}), so every layer is learning.`;
  const total = history.length ? history[0].map((_, k) => history.reduce((a, h) => a + (h[k] ?? 0), 0)) : [];
  if (trend(total) === 'falling') s += lossFalling ? ' Gradients are shrinking over time: as the loss falls, there is less left to correct.' : ' Gradients are shrinking over time.';
  return s;
}

/** `std`: the spread of a layer's outputs over the batch. */
export function activationStory(layers: { label: string; fracZero: number; std: number }[], relu: boolean): string {
  const dead = layers.filter((l) => l.fracZero > 0.9);
  if (dead.length) return `${dead.map((l) => l.label).join(' and ')}: over 90% of outputs are exactly zero. Those units are “dead”: a zero output passes no gradient back, so they cannot recover.`;
  // measured on the presets: healthy layers spread 0.45 or more, the
  // vanishing-gradient layers 0.06-0.13
  const flat = layers.filter((l) => l.std < 0.2);
  if (flat.length === layers.length) return `Every layer’s outputs stay in a narrow band (spread under 0.2): the units give nearly the same output for every point, so little information about where a point is reaches the end.`;
  if (flat.length) return `${flat.map((l) => l.label).join(', ')}: outputs stay in a narrow band (spread under 0.2), so these units give nearly the same output for every point.`;
  if (relu) return 'Some exact zeros are normal for ReLU: each unit is off for some inputs and on for others, which is how the network carves up the plane.';
  return 'Outputs vary widely from point to point: the units are active and respond to where a point is.';
}

/** Who is ahead in the optimizer race, and who is stuck. */
export function raceStory(racers: { name: string; loss: number[]; diverged: boolean }[]): string {
  const live = racers.filter((r) => r.loss.length > 0 && !r.diverged);
  if (!racers.some((r) => r.loss.length)) return 'Press Race: all four start from the same point and step downhill at the same time.';
  const parts: string[] = [];
  const sorted = [...live].sort((a, b) => a.loss.at(-1)! - b.loss.at(-1)!);
  if (sorted.length) parts.push(`Lowest loss so far: ${sorted[0].name} (${fmt(sorted[0].loss.at(-1)!)}).`);
  if (sorted.length > 1) parts.push(`Furthest behind: ${sorted.at(-1)!.name} (${fmt(sorted.at(-1)!.loss.at(-1)!)}).`);
  // stuck: the loss moved by under 1% over the last 50 recorded steps, while still above the leader
  const stuck = sorted.slice(1).filter((r) => {
    const n = r.loss.length;
    if (n < 60) return false;
    const before = r.loss[n - 51], now = r.loss[n - 1];
    return Math.abs(now - before) < 0.01 * Math.max(Math.abs(before), 1e-9) && now > sorted[0].loss.at(-1)! + 1e-3;
  });
  if (stuck.length) parts.push(`${stuck.map((r) => r.name).join(' and ')} ${stuck.length > 1 ? 'have' : 'has'} almost stopped improving: on a flat stretch, small gradients mean small steps.`);
  const div = racers.filter((r) => r.diverged);
  if (div.length) parts.push(`${div.map((r) => r.name).join(' and ')} diverged: the steps were too large and the loss blew up.`);
  return parts.join(' ');
}

