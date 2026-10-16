// Every chart is a plain function onto a canvas context. No chart library:
// these are a few dozen lines each and draw exactly what the page needs.

import type { Stats } from '../engine/protocol.ts';

import { AXIS, BLUE as NEG, GRID, INK, INK_2, LAYER_RAMP, MID, MUTED, ORANGE as POS, SURFACE_HIGH, SURFACE_LOW } from './theme.ts';
import type { RGB } from './theme.ts';

export type { RGB };
export type ColorScale = (value: number) => RGB;

// blue - dark - orange (theme.ts): safe for red-green colour blindness

const mix = (a: RGB, b: RGB, t: number): RGB => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];

/** Diverging scale centred on zero. Non-finite values are drawn magenta. */
export function divergingScale(maxAbs: number): ColorScale {
  const m = maxAbs > 0 && Number.isFinite(maxAbs) ? maxAbs : 1;
  return (v) => {
    if (!Number.isFinite(v)) return [200, 0, 200];
    const t = Math.max(-1, Math.min(1, v / m));
    return t < 0 ? mix(MID, NEG, -t) : mix(MID, POS, t);
  };
}

/**
 * For loss surfaces: dark at the minimum, light where the loss is high, on a
 * log scale so a valley stays visible next to walls a thousand times higher.
 * Contour bands every tenth of the range make the shape readable.
 */
export function surfaceScale(lo: number, hi: number): ColorScale {
  const span = Math.log1p(Math.max(hi - lo, 1e-9));
  return (v) => {
    if (!Number.isFinite(v)) return SURFACE_HIGH;
    const t = Math.max(0, Math.min(1, Math.log1p(Math.max(0, v - lo)) / span));
    const band = Math.floor(t * 10) % 2 === 0 ? 0 : 0.07;
    return mix(SURFACE_LOW, SURFACE_HIGH, Math.min(1, Math.sqrt(t) + band));
  };
}

const css = (c: RGB, a = 1) => `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},${a})`;

export function maxAbs(data: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < data.length; ++i) if (Number.isFinite(data[i])) m = Math.max(m, Math.abs(data[i]));
  return m;
}

/** Size the canvas backing store to its CSS size times the pixel ratio. Returns CSS width and height. */
export function fit(canvas: HTMLCanvasElement): { w: number; h: number; ctx: CanvasRenderingContext2D } {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext('2d')!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { w, h, ctx };
}

export function fmt(x: number): string {
  if (!Number.isFinite(x)) return Number.isNaN(x) ? 'nan' : x > 0 ? 'inf' : '-inf';
  const a = Math.abs(x);
  if (a !== 0 && (a < 1e-3 || a >= 1e4)) return x.toExponential(2);
  return x.toFixed(a < 1 ? 3 : 2);
}

/**
 * A tensor of rank 0 to 2 as a grid of cells; higher ranks are flattened to
 * [first dim, rest]. Numbers are written in the cells when they fit.
 */
export function drawTensor(ctx: CanvasRenderingContext2D, w: number, h: number, data: Float32Array, shape: number[], scale: ColorScale) {
  const rows = shape.length >= 2 ? shape[0] : 1;
  const cols = shape.length === 0 ? 1 : shape.length === 1 ? shape[0] : data.length / shape[0];
  const cw = w / cols, ch = Math.min(h / rows, cw * 1.2 + 8);
  ctx.font = `${Math.max(8, Math.min(13, ch * 0.4))}px 'JetBrains Mono', ui-monospace, monospace`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (let r = 0; r < rows; ++r)
    for (let c = 0; c < cols; ++c) {
      const v = data[r * cols + c];
      ctx.fillStyle = css(scale(v));
      ctx.fillRect(c * cw + 0.5, r * ch + 0.5, cw - 1, ch - 1);
      if (cw > 34 && ch > 14) {
        // light numbers on dark cells, dark on bright ones
        const [r0, g0, b0] = scale(v);
        ctx.fillStyle = 0.299 * r0 + 0.587 * g0 + 0.114 * b0 > 150 ? '#0b0f1a' : INK;
        ctx.fillText(fmt(v), c * cw + cw / 2, r * ch + ch / 2);
      }
    }
}

/** `res` x `res` values, row-major with y descending, one cell per value. */
export function drawField(ctx: CanvasRenderingContext2D, w: number, h: number, data: ArrayLike<number | null>, res: number, scale: ColorScale) {
  const img = ctx.createImageData(res, res);
  for (let i = 0; i < res * res; ++i) {
    const c = scale(data[i] ?? NaN);
    img.data.set([c[0], c[1], c[2], 255], i * 4);
  }
  // draw at native size on a scratch canvas, then let drawImage scale it
  const tmp = document.createElement('canvas');
  tmp.width = tmp.height = res;
  tmp.getContext('2d')!.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(tmp, 0, 0, w, h);
}

export interface Domain { x0: number; x1: number; y0: number; y1: number }

/** Labelled points over a field: class 0 blue, class 1 orange. */
export function drawPoints(ctx: CanvasRenderingContext2D, w: number, h: number, x: ArrayLike<number>, labels: number[], d: Domain, highlight = -1) {
  for (let i = 0; i < labels.length; ++i) {
    const px = ((x[2 * i] - d.x0) / (d.x1 - d.x0)) * w;
    const py = (1 - (x[2 * i + 1] - d.y0) / (d.y1 - d.y0)) * h;
    ctx.beginPath();
    ctx.arc(px, py, i === highlight ? 5 : 3, 0, Math.PI * 2);
    ctx.fillStyle = css(labels[i] ? POS : NEG);
    ctx.fill();
    ctx.strokeStyle = 'rgba(232,238,249,0.9)';
    ctx.lineWidth = 1;
    ctx.stroke();
  }
}

export function drawHistogram(ctx: CanvasRenderingContext2D, w: number, h: number, stats: Stats, color: string) {
  const hist = stats.hist;
  let peak = 1;
  hist.forEach((v) => (peak = Math.max(peak, v)));
  const bw = w / hist.length;
  ctx.fillStyle = color;
  hist.forEach((v, i) => {
    const bh = (v / peak) * (h - 12);
    ctx.fillRect(i * bw + 0.5, h - 12 - bh, bw - 1, bh);
  });
  ctx.fillStyle = TICK;
  ctx.font = CHART_FONT;
  ctx.textBaseline = 'bottom';
  ctx.textAlign = 'left';
  ctx.fillText(fmt(stats.histLo), 0, h);
  ctx.textAlign = 'right';
  ctx.fillText(fmt(stats.histHi), w, h);
}

/** Chart text: the page's own font, small and quiet. */
export const CHART_FONT = '500 10.5px Inter, system-ui, sans-serif';
const TICK = MUTED;

export interface Series { label: string; color: string; x: number[]; y: (number | null)[]; dash?: boolean }

/** Line chart with optional log y. Gaps (null, non-finite) break the line. */
export interface LineOpts {
  logY?: boolean;
  /** Axis names, written on the chart: a reader should not have to guess them. */
  xLabel?: string;
  yLabel?: string;
  /** Labelled horizontal reference lines, e.g. the loss of random guessing. */
  refs?: { y: number; label: string }[];
}

export function drawLines(ctx: CanvasRenderingContext2D, w: number, h: number, series: Series[], opts: LineOpts = {}) {
  const pad = { l: 50, r: 8, t: opts.yLabel ? 20 : 8, b: 20 };
  const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
  // a log scale only when every value is above zero: a loss that goes
  // negative (the saddle's minimum is −1) would otherwise be pinned at the floor
  const logY = !!opts.logY && series.every((s) => s.y.every((y) => y === null || !Number.isFinite(y) || y > 0));
  const ty = (v: number) => (logY ? Math.log10(v) : v);
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const s of series)
    s.x.forEach((x, i) => {
      const y = s.y[i];
      if (y === null || !Number.isFinite(y)) return;
      x0 = Math.min(x0, x); x1 = Math.max(x1, x);
      y0 = Math.min(y0, ty(y)); y1 = Math.max(y1, ty(y));
    });
  // reference lines stay in view
  if (Number.isFinite(x0)) for (const r of opts.refs ?? []) if (!logY || r.y > 0) { y0 = Math.min(y0, ty(r.y)); y1 = Math.max(y1, ty(r.y)); }
  ctx.strokeStyle = AXIS;
  ctx.beginPath(); ctx.moveTo(pad.l, pad.t + ph + 0.5); ctx.lineTo(pad.l + pw, pad.t + ph + 0.5); ctx.stroke();
  if (!Number.isFinite(x0)) return;
  if (x1 === x0) x1 = x0 + 1;
  if (y1 - y0 < 1e-9) { y0 -= 0.5; y1 += 0.5; }
  const X = (x: number) => pad.l + ((x - x0) / (x1 - x0)) * pw;
  const Y = (y: number) => pad.t + (1 - (ty(y) - y0) / (y1 - y0)) * ph;

  ctx.fillStyle = TICK;
  ctx.font = CHART_FONT;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let k = 0; k <= 3; ++k) {
    const v = y0 + ((y1 - y0) * k) / 3;
    const yy = Math.round(pad.t + (1 - k / 3) * ph) + 0.5;
    ctx.fillText(fmt(logY ? 10 ** v : v), pad.l - 6, yy);
    if (k === 0) continue;
    ctx.strokeStyle = GRID;
    ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(pad.l + pw, yy); ctx.stroke();
  }
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillText(String(Math.round(x0)), pad.l, h - pad.b + 4);
  ctx.textAlign = 'right';
  ctx.fillText(String(Math.round(x1)), pad.l + pw, h - pad.b + 4);
  ctx.textAlign = 'center';
  ctx.fillText(opts.xLabel ?? 'training steps →', pad.l + pw / 2, h - pad.b + 4);
  if (opts.yLabel) {
    // the axis title, written level above the axis rather than sideways
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(`↑ ${opts.yLabel}${logY ? ' (log scale)' : ''}`, 2, 0);
  }
  for (const r of opts.refs ?? []) {
    if (logY && r.y <= 0) continue;
    const yy = Y(r.y);
    ctx.strokeStyle = AXIS;
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(pad.l + pw, yy); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = INK_2;
    ctx.textAlign = 'right';
    // above the line, or below it when the line is near the top
    const above = yy - pad.t > 14;
    ctx.textBaseline = above ? 'bottom' : 'top';
    ctx.fillText(r.label, pad.l + pw - 4, above ? yy - 2 : yy + 2);
    ctx.fillStyle = TICK;
  }

  for (const s of series) {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 1.5;
    ctx.setLineDash(s.dash ? [4, 3] : []);
    ctx.beginPath();
    let pen = false;
    s.x.forEach((x, i) => {
      const y = s.y[i];
      if (y === null || !Number.isFinite(y)) { pen = false; return; }
      if (pen) ctx.lineTo(X(x), Y(y)); else ctx.moveTo(X(x), Y(y));
      pen = true;
    });
    ctx.stroke();
  }
  ctx.setLineDash([]);
}

/** Optimizer paths over a loss surface drawn with `drawField`. */
export function drawPaths(ctx: CanvasRenderingContext2D, w: number, h: number,
  paths: { label: string; color: string; points: [number, number][] }[], d: Domain) {
  const X = (x: number) => ((x - d.x0) / (d.x1 - d.x0)) * w;
  const Y = (y: number) => (1 - (y - d.y0) / (d.y1 - d.y0)) * h;
  for (const p of paths) {
    if (!p.points.length) continue;
    ctx.beginPath();
    p.points.forEach(([x, y], i) => (i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y))));
    // a dark halo first, so each path stands off the landscape
    ctx.lineJoin = 'round';
    ctx.strokeStyle = 'rgba(5,7,13,0.85)';
    ctx.lineWidth = 4.5;
    ctx.stroke();
    ctx.strokeStyle = p.color;
    ctx.lineWidth = 2;
    ctx.stroke();
    const [ex, ey] = p.points[p.points.length - 1];
    ctx.beginPath();
    ctx.arc(X(ex), Y(ey), 5, 0, Math.PI * 2);
    ctx.fillStyle = p.color;
    ctx.fill();
    ctx.strokeStyle = INK;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
}

// colours for things that are told apart by name (racers, compared optimizers).
// chosen so every pair stays far apart (CIELAB distance > 25) with normal
// vision and simulated protanopia, deuteranopia and tritanopia; checked in
// colors.test.ts. no view shows more than four such series at once
export { PALETTE } from './theme.ts';

// layers are ordered, so they get a ramp instead: violet for the first layer
// to orange for the last (theme.ts). lightness rises along it under every
// kind of colour vision, so depth reads the same for everyone
const RAMP = LAYER_RAMP;

/** Colour of layer `i` of `n`. */
export function layerColor(i: number, n: number): string {
  const t = n > 1 ? Math.max(0, Math.min(1, i / (n - 1))) : 0;
  const s = t * (RAMP.length - 1), k = Math.min(RAMP.length - 2, Math.floor(s));
  const c = mix(RAMP[k], RAMP[k + 1], s - k);
  return '#' + c.map((v) => Math.round(v).toString(16).padStart(2, '0')).join('');
}
