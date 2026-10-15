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
  if (step === 0 || !acc) return 'Not trained yet: these colours are random guesses. Press Run and watch them reshape.';
  const base = `${pct(acc.train)} of training dots and ${pct(acc.val)} of test dots on the right side.`;
  if (spread < 0.2) return `${base} Almost one colour: it gives the same answer everywhere, so it cannot tell the dots apart.`;
  if (acc.train < 0.65) return `${base} That is close to guessing (50%).`;
  if (acc.train > 0.97) return `${base} The boundary follows the data.`;
  return `${base} The boundary is still bending into shape.`;
}

export function lossStory(loss: number[], val: (number | null)[], step: number): string {
  if (step === 0 || !loss.length) return `Not trained yet. A 50/50 guess scores ${fmt(COIN_FLIP)}; lower is better.`;
  const cur = loss[loss.length - 1], first = loss[0];
  const t = trend(loss);
  const moving = t === 'falling' ? 'Still falling.'
    : t === 'rising' ? 'Rising: the steps may be too big (try a smaller lr).'
    : t === 'flat' ? 'Levelled off.' : '';
  const level = cur > 0.6 ? 'Still about a coin flip.'
    : cur < 0.05 ? 'That is low: confident and right.'
    : 'Better than a coin flip.';
  let s = `${fmt(first)} → ${fmt(cur)}. ${level} ${moving}`;
  const v = val.filter((x): x is number => x !== null).at(-1);
  if (v !== undefined && v > 0.1 && v > 2 * Math.max(cur, 0.01))
    s += ` Test loss ${fmt(v)} is far higher: it may be memorising its training dots.`;
  return s.trim();
}

export function gradStory(layers: { label: string; norm: number }[], history: number[][], lossFalling = false): string {
  if (!layers.length) return '';
  if (layers.every((l) => !(l.norm > 1e-12))) return 'Every gradient is zero: no weight can change, so it cannot learn at all.';
  const first = layers[0], last = layers[layers.length - 1];
  const ratio = first.norm / Math.max(last.norm, 1e-30);
  const times = (r: number) => {
    const p = Math.round(Math.log10(r));
    return p >= 3 ? `about ${'1' + '0'.repeat(p)}×`.replace(/(\d)(?=(\d{3})+×)/g, '$1,') : `about ${Math.round(r)}×`;
  };
  let s: string;
  if (ratio < 0.01) s = `Layer 1 gets ${times(1 / ratio)} less than the last layer: the gradient vanishes on its way back.`;
  else if (ratio > 100) s = `Layer 1 gets ${times(ratio)} more than the last layer: the gradient grows on its way back and may explode.`;
  else s = `Similar sizes (within ${times(Math.max(ratio, 1 / ratio))}): every layer is learning.`;
  const total = history.length ? history[0].map((_, k) => history.reduce((a, h) => a + (h[k] ?? 0), 0)) : [];
  if (trend(total) === 'falling') s += lossFalling ? ' Shrinking as the loss falls: less left to correct.' : ' Shrinking over time.';
  return s;
}

/** `std`: the spread of a layer's outputs over the batch. */
export function activationStory(layers: { label: string; fracZero: number; std: number }[], relu: boolean): string {
  const dead = layers.filter((l) => l.fracZero > 0.9);
  if (dead.length) return `${dead.map((l) => l.label).join(' and ')}: over 90% exactly zero. These neurons are dead: zero passes no gradient back, so they cannot recover.`;
  // measured on the presets: healthy layers spread 0.45 or more, the
  // vanishing-gradient layers 0.06-0.13
  const flat = layers.filter((l) => l.std < 0.2);
  if (flat.length === layers.length) return 'Every layer stays in a narrow band: neurons give nearly the same output for every dot, so little reaches the end.';
  if (flat.length) return `${flat.map((l) => l.label).join(', ')}: a narrow band, nearly the same output for every dot.`;
  if (relu) return 'Some zeros are normal for ReLU: each neuron is off for some dots, on for others.';
  return 'Outputs vary widely from dot to dot: the neurons respond to position.';
}

/** Who is ahead in the optimizer race, who is stuck, and who arrived first. */
export function raceStory(racers: { name: string; loss: number[]; diverged: boolean }[]): string {
  if (!racers.some((r) => r.loss.length)) return 'Press Race: all four start together and step downhill at once.';
  const live = racers.filter((r) => r.loss.length > 0 && !r.diverged);
  const parts: string[] = [];
  const last = (r: { loss: number[] }) => r.loss[r.loss.length - 1];
  const sorted = [...live].sort((a, b) => last(a) - last(b));
  if (sorted.length) {
    // "at the bottom": within 0.1% of the drop from the start to the best so far
    const best = last(sorted[0]), start = Math.max(...live.map((r) => r.loss[0]));
    const near = (v: number) => v - best <= 0.001 * Math.max(start - best, 1e-9);
    const arrived = live.filter((r) => near(last(r)));
    if (arrived.length === live.length && live.length > 1) {
      const firstAt = (r: { loss: number[] }) => r.loss.findIndex(near);
      const first = [...live].sort((a, b) => firstAt(a) - firstAt(b))[0];
      parts.push(`All of them have reached the bottom; the first to get there was ${first.name}.`);
    } else {
      parts.push(`Lowest so far: ${sorted[0].name} (${fmt(best)}).`);
      const behind = sorted.at(-1)!;
      if (sorted.length > 1 && !near(last(behind))) parts.push(`Furthest behind: ${behind.name} (${fmt(last(behind))}).`);
      // stuck: moved under 1% over the last 50 recorded steps while still well above the leader
      const stuck = sorted.slice(1).filter((r) => {
        const n = r.loss.length;
        return n >= 60 && Math.abs(last(r) - r.loss[n - 51]) < 0.01 * Math.max(Math.abs(r.loss[n - 51]), 1e-9) && !near(last(r));
      });
      if (stuck.length) parts.push(`${stuck.map((r) => r.name).join(' and ')} ${stuck.length > 1 ? 'have' : 'has'} almost stopped: flat ground, small slope, small steps.`);
    }
  }
  const div = racers.filter((r) => r.diverged);
  if (div.length) parts.push(`${div.map((r) => r.name).join(' and ')} diverged: steps too large. Try a smaller lr.`);
  return parts.join(' ');
}
