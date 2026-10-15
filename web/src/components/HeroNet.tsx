// The home page's picture: a small network that draws itself in, then runs
// one training step, once: a forward pass sweeps left to right in blue, the
// gradients sweep back in orange, and it settles with every weight tinted by
// its sign. Pure CSS (styles.css, .hero-net); it starts when the splash lifts
// (:root[data-ready]) and shows the settled picture for prefers-reduced-motion.

import type { CSSProperties } from 'react';

const LAYERS = [3, 5, 5, 2];
const W = 440, H = 320, PAD_X = 36, PAD_Y = 34;
const STAGE = 0.32;          // seconds between one layer and the next
const BACK = 1.7;            // when the backward sweep starts

const x = (l: number) => PAD_X + (l * (W - 2 * PAD_X)) / (LAYERS.length - 1);
const y = (l: number, i: number) => {
  const n = LAYERS[l], gap = Math.min(62, (H - 2 * PAD_Y) / (n - 1 || 1));
  return H / 2 + (i - (n - 1) / 2) * gap;
};
// fixed pseudo-random weights: the same picture every load
const weight = (l: number, i: number, j: number) => Math.sin(12.9898 * (l + 1) + 78.233 * (i + 1) + 37.719 * (j + 1));

const last = LAYERS.length - 1;
const delays = (fwd: number, bwd: number) => ({ '--df': `${fwd}s`, '--db': `${bwd}s` }) as CSSProperties;

export function HeroNet() {
  const edges = [], nodes = [];
  for (let l = 0; l < last; ++l)
    for (let i = 0; i < LAYERS[l]; ++i)
      for (let j = 0; j < LAYERS[l + 1]; ++j) {
        const w = weight(l, i, j);   // -1 .. 1
        const d = { x1: x(l), y1: y(l, i), x2: x(l + 1), y2: y(l + 1, j) };
        // an edge lights between its two layers: forward after layer l, backward after layer l + 1
        const style = { ...delays((l + 0.5) * STAGE, BACK + (last - l - 0.5) * STAGE), '--in': `${0.15 + l * 0.12}s`, strokeWidth: 0.6 + 1.8 * Math.abs(w) } as CSSProperties;
        edges.push(
          <g key={`${l}-${i}-${j}`} className="hn-edge" style={style}>
            <line {...d} className="hn-base" />
            <line {...d} className="hn-f" />
            <line {...d} className="hn-b" />
            <line {...d} className={`hn-w ${w < 0 ? 'neg' : 'pos'}`} />
          </g>,
        );
      }
  for (let l = 0; l <= last; ++l)
    for (let i = 0; i < LAYERS[l]; ++i) {
      const style = { ...delays(l * STAGE, BACK + (last - l) * STAGE), '--in': `${l * 0.12 + i * 0.03}s` } as CSSProperties;
      const r = l === 0 || l === last ? 9 : 10;
      nodes.push(
        <g key={`${l}-${i}`} className="hn-node" style={style} transform={`translate(${x(l)} ${y(l, i)})`}>
          <circle r={r} className="hn-base" />
          <circle r={r} className="hn-f" />
          <circle r={r} className="hn-b" />
        </g>,
      );
    }
  return (
    <svg className="hero-net" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="A small neural network: values flow forward in blue, gradients flow back in orange">
      <g>{edges}</g>
      <g>{nodes}</g>
      <g className="hn-labels">
        <text x={x(0)} y={H - 4}>input</text>
        <text x={x(last)} y={H - 4}>output</text>
      </g>
    </svg>
  );
}
