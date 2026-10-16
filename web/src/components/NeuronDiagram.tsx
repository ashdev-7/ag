// The classic picture of a neural network: neurons in columns (input,
// hidden layers, output), every weight a line between them. Orange lines
// are positive weights, blue negative, thicker is larger. Inside each neuron,
// a small picture of what it responds to: its output at every point of the
// input plane. Pointing at a neuron or a line says what it is, in words.

import { useEffect, useRef, useState } from 'react';
import type { Session } from '../engine/client.ts';
import type { Preset } from '../graph/presets.ts';
import { divergingScale, drawField, fit, fmt } from '../views/draw.ts';
import { Canvas } from './Canvas.tsx';
import { BLUE, INK_2, MUTED, ORANGE } from '../views/theme.ts';
import type { ColorScale } from '../views/draw.ts';

export interface NeuronState {
  res: number;
  /** Per layer after the input (hidden layers, then output), per neuron: its output over the plane. */
  grids: Float32Array[][];
  /** Per layer: weights [in, out] row-major, and biases [out]. */
  weights: { inDim: number; outDim: number; data: Float32Array }[];
  biases: Float32Array[];
}

const RES = 24;

/** Read every weight and sweep every neuron over the plane. Null if the engine refused. */
export async function fetchNeurons(s: Session, preset: Preset, domain: { x0: number; x1: number; y0: number; y1: number }): Promise<NeuronState | null> {
  const grids: Float32Array[][] = [], weights: NeuronState['weights'] = [], biases: Float32Array[] = [];
  for (const [k, l] of preset.layers.entries()) {
    const w = await s.read(l.weight, 'value');
    if (!w.ok || w.value.shape.length !== 2) return null;
    const [inDim, outDim] = w.value.shape;
    const b = await s.read(l.weight.replace(/^W/, 'b'), 'value');
    weights.push({ inDim, outDim, data: w.value.data });
    biases.push(b.ok ? b.value.data : new Float32Array(outDim));
    // the output layer is shown as probabilities, the softmax of the logits
    const node = k === preset.layers.length - 1 ? (preset.boundary?.nodeId ?? l.act) : l.act;
    // the whole layer from one sweep of the grid: one request, not one per neuron
    const g = await s.evalGrid({ ...domain, res: RES, nodeId: node, element: 'all' });
    const n = RES * RES;
    grids.push(Array.from({ length: outDim }, (_, j) =>
      g.ok && g.value.data.length >= (j + 1) * n ? g.value.data.subarray(j * n, (j + 1) * n) : new Float32Array(n)));
  }
  return { res: RES, grids, weights, biases };
}

interface Box { x: number; y: number; s: number; layer: number; index: number }
interface Line { x1: number; y1: number; x2: number; y2: number; layer: number; from: number; to: number; w: number }

const POS = ORANGE.join(','), NEG = BLUE.join(',');

function drawCell(ctx: CanvasRenderingContext2D, data: ArrayLike<number>, res: number, scale: ColorScale, b: Box) {
  const tmp = document.createElement('canvas');
  tmp.width = tmp.height = res;
  const tctx = tmp.getContext('2d')!;
  const img = tctx.createImageData(res, res);
  for (let i = 0; i < res * res; ++i) {
    const c = scale(data[i]);
    img.data.set([c[0], c[1], c[2], 255], i * 4);
  }
  tctx.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(tmp, b.x, b.y, b.s, b.s);
}

/** Squash an activation's outputs to [-1, 1] for colour: probabilities and sigmoid via 2p − 1, ReLU by its peak. */
function normalised(act: string, g: ArrayLike<number>): number[] {
  if (act === 'softmax' || act === 'sigmoid') return Array.from(g, (v) => 2 * v - 1);
  if (act === 'relu' || act === 'leaky_relu') {
    let peak = 0;
    for (let i = 0; i < g.length; ++i) peak = Math.max(peak, Math.abs(g[i]));
    return Array.from(g, (v) => (peak > 0 ? v / peak : 0));
  }
  return Array.from(g);
}

export function NeuronDiagram(props: { preset: Preset; state: NeuronState | null; height?: number }) {
  const { preset, state } = props;
  const ref = useRef<HTMLCanvasElement>(null);
  const layout = useRef<{ boxes: Box[]; lines: Line[] }>({ boxes: [], lines: [] });
  const [hover, setHoverState] = useState<{ text: string; picture?: number[] }>({ text: '' });
  const setHover = (text: string, picture?: number[]) => setHoverState({ text, picture });
  const acts = preset.layers.map((l, k) => (k === preset.layers.length - 1 ? 'softmax' : preset.spec.nodes.find((n) => n.id === l.act)?.op ?? ''));
  const sizes = state ? [state.weights[0].inDim, ...state.weights.map((w) => w.outDim)] : [];
  // each label from longest to shortest; the longest that fits its column is drawn
  const titles: string[][] = [['input'], ...preset.layers.map((l, k) => (k === preset.layers.length - 1
    ? ['output (softmax)', 'output'] : [`${l.label} (${acts[k]})`, l.label, `L${k + 1}`]))];

  useEffect(() => {
    const c = ref.current;
    if (!c || !state) return;
    const draw = () => {
      const { ctx, w, h } = fit(c);
      const top = 22, bottom = 26, cols = sizes.length;
      // room on the left for x₁/x₂ and on the right for the class names
      const colX = (k: number) => 24 + (k * (w - 24 - 64 - 34)) / Math.max(1, cols - 1);
      const maxN = Math.max(...sizes);
      // wide layers (64 neurons) get small squares rather than a page-tall diagram
      const s = Math.max(3, Math.min(34, ((h - top - bottom) / maxN) * 0.78));
      const boxes: Box[] = [];
      sizes.forEach((n, k) => {
        const gap = (h - top - bottom) / n;
        for (let j = 0; j < n; ++j) boxes.push({ x: colX(k), y: top + gap * (j + 0.5) - s / 2, s, layer: k, index: j });
      });
      const at = (k: number, j: number) => boxes.find((b) => b.layer === k && b.index === j)!;
      // lines: one per weight, width and opacity by size relative to the layer's largest
      const lines: Line[] = [];
      state.weights.forEach((wt, k) => {
        let max = 0;
        for (const v of wt.data) max = Math.max(max, Math.abs(v));
        for (let i = 0; i < wt.inDim; ++i)
          for (let j = 0; j < wt.outDim; ++j) {
            const v = wt.data[i * wt.outDim + j], a = at(k, i), b = at(k + 1, j);
            const r = max > 0 ? Math.abs(v) / max : 0;
            const line = { x1: a.x + a.s, y1: a.y + a.s / 2, x2: b.x, y2: b.y + b.s / 2, layer: k, from: i, to: j, w: v };
            lines.push(line);
            ctx.strokeStyle = `rgba(${v >= 0 ? POS : NEG},${0.15 + 0.75 * r})`;
            ctx.lineWidth = 0.4 + 3.6 * r;
            ctx.beginPath();
            ctx.moveTo(line.x1, line.y1);
            ctx.lineTo(line.x2, line.y2);
            ctx.stroke();
          }
      });
      // neurons: inputs show their own coordinate; the rest what they output over the plane
      boxes.forEach((b) => {
        if (b.layer === 0) {
          const d = new Float32Array(state.res * state.res);
          for (let i = 0; i < d.length; ++i) {
            const col = i % state.res, row = Math.floor(i / state.res);
            d[i] = b.index === 0 ? (2 * col) / (state.res - 1) - 1 : 1 - (2 * row) / (state.res - 1);
          }
          drawCell(ctx, d, state.res, divergingScale(1), b);
        } else {
          drawCell(ctx, normalised(acts[b.layer - 1], state.grids[b.layer - 1][b.index]), state.res, divergingScale(1), b);
        }
        ctx.strokeStyle = 'rgba(148,170,220,0.45)';
        ctx.lineWidth = 1;
        ctx.strokeRect(b.x, b.y, b.s, b.s);
      });
      const room = cols > 1 ? colX(1) - colX(0) - 8 : w;
      const fitting = (options: string[]) => options.find((o) => ctx.measureText(o).width <= room) ?? '';
      // centred on its column, kept inside the canvas at both edges
      const label = (text: string, k: number, y: number) => {
        const half = ctx.measureText(text).width / 2;
        ctx.fillText(text, Math.min(w - half - 2, Math.max(half + 2, colX(k) + s / 2)), y);
      };
      ctx.fillStyle = MUTED;
      ctx.font = '500 11px Inter, system-ui, sans-serif';
      ctx.textAlign = 'center';
      titles.forEach((t, k) => label(fitting(t), k, 12));
      // what each column does, under it
      ctx.fillStyle = INK_2;
      ctx.font = '500 11.5px Inter, system-ui, sans-serif';
      sizes.forEach((_, k) => {
        const what = k === 0 ? ['the dot’s position', 'position'] : k === cols - 1 ? ['the answer', 'answer']
          : k === 1 ? ['learn simple shapes', 'simple shapes', 'shapes'] : ['combine them', 'combine'];
        label(fitting(what), k, h - 8);
      });
      ctx.fillStyle = MUTED;
      ctx.font = '500 11px Inter, system-ui, sans-serif';
      // name the inputs and outputs
      ctx.textAlign = 'left';
      boxes.filter((b) => b.layer === 0).forEach((b) => ctx.fillText(b.index === 0 ? 'x₁' : 'x₂', b.x - 18, b.y + b.s / 2 + 4));
      boxes.filter((b) => b.layer === sizes.length - 1).forEach((b) => ctx.fillText(b.index === 0 ? 'blue' : 'orange', b.x + b.s + 4, b.y + b.s / 2 + 4));
      layout.current = { boxes, lines };
    };
    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(c);
    return () => ro.disconnect();
  }, [state, sizes.join(','), titles.join(',')]); // eslint-disable-line react-hooks/exhaustive-deps

  const onMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    const { boxes, lines } = layout.current;
    const b = boxes.find((q) => x >= q.x - 2 && x <= q.x + q.s + 2 && y >= q.y - 2 && y <= q.y + q.s + 2);
    if (b && state) {
      if (b.layer === 0) return setHover(`Input ${b.index === 0 ? 'x₁' : 'x₂'}: the point’s ${b.index === 0 ? 'horizontal' : 'vertical'} position. Its picture is just that coordinate: orange where it is large, blue where it is small.`);
      const last = b.layer === sizes.length - 1;
      const bias = state.biases[b.layer - 1][b.index];
      const picture = normalised(acts[b.layer - 1], state.grids[b.layer - 1][b.index]);
      return setHover(last
        ? `Answer neuron “${b.index === 0 ? 'blue' : 'orange'}”: how sure the network is that a dot is ${b.index === 0 ? 'blue' : 'orange'}, for every position on the plane (orange = sure, blue = not). Bias ${fmt(bias)}.`
        : `${titles[b.layer][0]}, neuron ${b.index + 1}: it adds up its weighted inputs plus a bias of ${fmt(bias)}, then applies ${acts[b.layer - 1]}. Its picture is its output at every point of the plane (orange high, blue low): the feature it has learned to detect.`, picture);
    }
    // nearest line within 4 pixels
    let best: Line | null = null, bestD = 4;
    for (const l of lines) {
      const dx = l.x2 - l.x1, dy = l.y2 - l.y1;
      const t = Math.max(0, Math.min(1, ((x - l.x1) * dx + (y - l.y1) * dy) / (dx * dx + dy * dy)));
      const d = Math.hypot(x - (l.x1 + t * dx), y - (l.y1 + t * dy));
      if (d < bestD) { bestD = d; best = l; }
    }
    if (best) {
      const from = best.layer === 0 ? (best.from === 0 ? 'input x₁' : 'input x₂') : `${titles[best.layer][0]} neuron ${best.from + 1}`;
      const to = best.layer === sizes.length - 2 ? `output “${best.to === 0 ? 'blue' : 'orange'}”` : `${titles[best.layer + 1][0]} neuron ${best.to + 1}`;
      return setHover(`Weight from ${from} to ${to}: ${fmt(best.w)}. ${best.w >= 0 ? 'Positive: when the first goes up, it pushes the second up.' : 'Negative: when the first goes up, it pushes the second down.'}`);
    }
    setHover('');
  };

  return (
    <div className="neurons">
      {/* wider layers get a taller diagram, so every neuron stays at least ~11 px */}
      <canvas ref={ref} style={{ width: '100%', height: props.height ?? Math.min(520, Math.max(360, 40 + Math.max(0, ...sizes) * 14)), display: 'block' }}
        onPointerMove={onMove} onPointerDown={onMove} onPointerLeave={(e) => { if (e.pointerType === 'mouse') setHover(''); }} role="img" aria-label={`Network diagram: ${sizes.join(' → ')} neurons`} />
      <div className="story neuron-hover">
        {hover.picture && state && (
          <Canvas className="neuron-preview" height={84} draw={(ctx, w, h) => drawField(ctx, w, h, hover.picture!, state.res, divergingScale(1))} />
        )}
        <span>{hover.text || (state ? 'Point at or tap a neuron or a line to see what it is.' : 'Reading the network…')}</span>
      </div>
    </div>
  );
}
