// Small interactive figures for the lessons. Plain canvas, no engine: each
// shows one idea with one or two sliders.

import { useState } from 'react';
import { fmt } from '../views/draw.ts';
import { Canvas } from './Canvas.tsx';

const BLUE = '#2166ac', ORANGE = '#e66101', INK = '#1d1d1f', GRID = '#e4e4ea';

interface Box { x0: number; x1: number; y0: number; y1: number }

/** Map a math box onto the canvas, draw the axes, and return the mappings. */
function frame(ctx: CanvasRenderingContext2D, w: number, h: number, b: Box) {
  const X = (x: number) => ((x - b.x0) / (b.x1 - b.x0)) * w;
  const Y = (y: number) => h - ((y - b.y0) / (b.y1 - b.y0)) * h;
  ctx.strokeStyle = GRID;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, Y(0)); ctx.lineTo(w, Y(0));
  ctx.moveTo(X(0), 0); ctx.lineTo(X(0), h);
  ctx.stroke();
  return { X, Y };
}

function curve(ctx: CanvasRenderingContext2D, w: number, X: (x: number) => number, Y: (y: number) => number, b: Box,
  f: (x: number) => number, color: string, dash: number[] = []) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.setLineDash(dash);
  ctx.beginPath();
  for (let i = 0; i <= w; i += 2) {
    const x = b.x0 + (i / w) * (b.x1 - b.x0);
    const y = f(x);
    if (i === 0) ctx.moveTo(X(x), Y(y)); else ctx.lineTo(X(x), Y(y));
  }
  ctx.stroke();
  ctx.setLineDash([]);
}

function dot(ctx: CanvasRenderingContext2D, x: number, y: number, color: string, r = 5) {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fill();
}

// ── slope ───────────────────────────────────────────────────────────────────

/** f(x) = x²: a point, a nudge h, the measured slope and the exact one. */
export function SlopeWidget() {
  const [x, setX] = useState(1);
  const [h, setH] = useState(0.5);
  const f = (t: number) => t * t;
  const measured = (f(x + h) - f(x)) / h, exact = 2 * x;
  const b: Box = { x0: -2.5, x1: 2.5, y0: -1, y1: 6 };
  return (
    <figure className="widget">
      <Canvas height={220} draw={(ctx, w, hh) => {
        const { X, Y } = frame(ctx, w, hh, b);
        curve(ctx, w, X, Y, b, f, INK);
        // the measured slope: a line through (x, f(x)) and (x + h, f(x + h))
        curve(ctx, w, X, Y, b, (t) => f(x) + measured * (t - x), ORANGE, [5, 4]);
        curve(ctx, w, X, Y, b, (t) => f(x) + exact * (t - x), BLUE);
        dot(ctx, X(x), Y(f(x)), INK);
        dot(ctx, X(x + h), Y(f(x + h)), ORANGE, 4);
      }} />
      <div className="widget-controls">
        <label>x = {fmt(x)} <input type="range" min={-2} max={2} step={0.05} value={x} onChange={(e) => setX(Number(e.target.value))} /></label>
        <label>nudge h = {fmt(h)} <input type="range" min={0.01} max={1} step={0.01} value={h} onChange={(e) => setH(Number(e.target.value))} /></label>
      </div>
      <figcaption>
        The curve is <code>f(x) = x²</code>. Dashed orange: the slope measured by nudging, <code>(f(x + h) − f(x)) / h = {fmt(measured)}</code>.
        Blue: the exact slope, the derivative <code>2x = {fmt(exact)}</code>. Shrink <code>h</code> and the two meet.
      </figcaption>
    </figure>
  );
}

// ── gradient descent ────────────────────────────────────────────────────────

/** Gradient descent on f(x) = x² from x = 1.8, with a learning-rate slider. */
export function DescentWidget() {
  const [lr, setLr] = useState(0.1);
  const steps: number[] = [1.8];
  for (let k = 0; k < 15; ++k) steps.push(steps[k] - lr * 2 * steps[k]);
  const last = Math.abs(steps[steps.length - 1]);
  // each step multiplies x by (1 − 2·lr): that factor decides everything
  const factor = 1 - 2 * lr;
  const verdict = Math.abs(factor) >= 1 ? 'diverges: every step lands further up the other side'
    : factor < 0 ? 'overshoots the bottom each step, but each bounce is smaller, so it converges'
    : lr < 0.05 ? 'creeps: correct but very slow' : 'converges smoothly';
  const b: Box = { x0: -3, x1: 3, y0: -0.5, y1: 9 };
  return (
    <figure className="widget">
      <Canvas height={220} draw={(ctx, w, hh) => {
        const { X, Y } = frame(ctx, w, hh, b);
        curve(ctx, w, X, Y, b, (t) => t * t, INK);
        ctx.strokeStyle = ORANGE;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        steps.forEach((s, i) => {
          const c = Math.max(b.x0, Math.min(b.x1, s));
          if (i === 0) ctx.moveTo(X(c), Y(c * c)); else ctx.lineTo(X(c), Y(Math.min(b.y1, c * c)));
        });
        ctx.stroke();
        steps.forEach((s, i) => {
          if (Math.abs(s) <= b.x1) dot(ctx, X(s), Y(s * s), i === 0 ? INK : ORANGE, i === 0 ? 5 : 3.5);
        });
      }} />
      <div className="widget-controls">
        <label>learning rate = {fmt(lr)} <input type="range" min={0.01} max={1.1} step={0.01} value={lr} onChange={(e) => setLr(Number(e.target.value))} /></label>
      </div>
      <figcaption>
        Loss <code>f(x) = x²</code>, slope <code>2x</code>, start at <code>x = 1.8</code>, 15 steps of <code>x ← x − lr · 2x</code>.
        With lr = {fmt(lr)} each step multiplies x by {fmt(factor)}, so it <b>{verdict}</b> (after 15 steps x = {fmt(steps[15])}{last > 1e6 ? ', blowing up' : ''}).
        Try 0.02, 0.3, 0.8 and 1.05.
      </figcaption>
    </figure>
  );
}

// ── activations ─────────────────────────────────────────────────────────────

const ACTS = {
  sigmoid: { f: (x: number) => 1 / (1 + Math.exp(-x)), d: (x: number) => { const s = 1 / (1 + Math.exp(-x)); return s * (1 - s); }, max: 'at most 0.25' },
  tanh: { f: Math.tanh, d: (x: number) => 1 - Math.tanh(x) ** 2, max: 'at most 1, near 0 for large |x|' },
  relu: { f: (x: number) => Math.max(0, x), d: (x: number) => (x > 0 ? 1 : 0), max: 'exactly 1 for positive inputs, exactly 0 for negative' },
  leaky_relu: { f: (x: number) => (x > 0 ? x : 0.01 * x), d: (x: number) => (x > 0 ? 1 : 0.01), max: '1 for positive inputs, 0.01 for negative: never fully off' },
} as const;
type ActName = keyof typeof ACTS;

/** An activation function (solid) and its slope (dashed). */
export function ActivationWidget(props: { initial?: ActName }) {
  const [name, setName] = useState<ActName>(props.initial ?? 'tanh');
  const a = ACTS[name];
  const b: Box = { x0: -5, x1: 5, y0: -1.2, y1: 1.6 };
  return (
    <figure className="widget">
      <Canvas height={200} draw={(ctx, w, hh) => {
        const { X, Y } = frame(ctx, w, hh, b);
        curve(ctx, w, X, Y, b, (t) => Math.min(b.y1, a.f(t)), BLUE);
        curve(ctx, w, X, Y, b, a.d, ORANGE, [5, 4]);
      }} />
      <div className="widget-controls segmented">
        {(Object.keys(ACTS) as ActName[]).map((k) => (
          <button key={k} className={k === name ? 'on' : ''} aria-pressed={k === name} onClick={() => setName(k)}>{k}</button>
        ))}
      </div>
      <figcaption>Blue: the function. Dashed orange: its slope, the factor every gradient is multiplied by when it passes back through this function: {a.max}.</figcaption>
    </figure>
  );
}
