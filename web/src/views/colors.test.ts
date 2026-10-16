// Colour checks: everything the page tells apart by colour must stay apart
// for viewers with the common colour vision deficiencies, not only with
// normal vision (PRD §9: no reliance on red against green).
//
// Simulation: Machado, Oliveira & Fernandes (2009), full severity, applied
// to linear RGB. Distance: CIE76 ΔE in CIELAB. As a rough guide, ΔE below
// about 10 is hard to tell apart on thin lines and small marks.

import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { layerColor, PALETTE } from './draw.ts';

type Vision = 'normal' | 'protan' | 'deutan' | 'tritan';
const VISIONS: Vision[] = ['normal', 'protan', 'deutan', 'tritan'];
const MACHADO: Record<Exclude<Vision, 'normal'>, number[][]> = {
  protan: [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
  deutan: [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.01182, 0.04294, 0.968881]],
  tritan: [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.3039]],
};

const rgb = (hex: string) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
const linear = (c: number) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);

function lab(hex: string, v: Vision): number[] {
  let l = rgb(hex).map(linear);
  if (v !== 'normal') l = MACHADO[v].map((row) => Math.min(1, Math.max(0, row[0] * l[0] + row[1] * l[1] + row[2] * l[2])));
  const [r, g, b] = l;
  const x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047, y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  const z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883;
  const f = (t: number) => (t > 216 / 24389 ? Math.cbrt(t) : (24389 / 27 * t + 16) / 116);
  return [116 * f(y) - 16, 500 * (f(x) - f(y)), 200 * (f(y) - f(z))];
}

const distance = (a: string, b: string, v: Vision) => {
  const p = lab(a, v), q = lab(b, v);
  return Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
};

/** The closest pair under any kind of vision. */
function closest(colors: Record<string, string>) {
  let worst = { d: Infinity, pair: '' };
  const names = Object.keys(colors);
  for (let i = 0; i < names.length; ++i)
    for (let j = i + 1; j < names.length; ++j)
      for (const v of VISIONS) {
        const d = distance(colors[names[i]], colors[names[j]], v);
        if (d < worst.d) worst = { d, pair: `${names[i]} / ${names[j]} (${v})` };
      }
  return worst;
}

const cssVar = (name: string) => {
  const m = new RegExp(String.raw`--${name}:\s*(#[0-9a-fA-F]{6})`).exec(readFileSync(new URL('../styles.css', import.meta.url), 'utf8'));
  if (!m) throw new Error(`--${name} not found in styles.css`);
  return m[1];
};

const luminance = (hex: string) => {
  const [r, g, b] = rgb(hex).map(linear);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};
/** WCAG contrast of a line colour against the page's background (--bg, dark) */
const contrastOnPage = (hex: string) => (luminance(hex) + 0.05) / (luminance(cssVar('bg')) + 0.05);

describe('colours', () => {
  it('series colours stay far apart for every kind of colour vision', () => {
    const w = closest(Object.fromEntries(PALETTE.map((c, i) => [`series ${i}`, c])));
    expect(w.d, w.pair).toBeGreaterThan(20);
  });

  it('the four step colours (forward, backward, update, tangent) stay apart', () => {
    const w = closest(Object.fromEntries(['forward', 'backward', 'update', 'tangent'].map((n) => [n, cssVar(n)])));
    expect(w.d, w.pair).toBeGreaterThan(20);
  });

  it('layer colours get lighter with depth for every kind of colour vision', () => {
    for (const n of [2, 3, 7]) {
      const colors = Array.from({ length: n }, (_, i) => layerColor(i, n));
      for (const v of VISIONS) {
        const L = colors.map((c) => lab(c, v)[0]);
        L.slice(1).forEach((l, i) => expect(l, `${n} layers, ${v}, layer ${i + 1}`).toBeGreaterThan(L[i] + 2));
      }
    }
  });

  it('every line colour is visible on the page (contrast at least 3:1)', () => {
    for (const c of [...PALETTE, layerColor(0, 2), layerColor(1, 2), cssVar('forward'), cssVar('backward'), cssVar('update'), cssVar('tangent')])
      expect(contrastOnPage(c), c).toBeGreaterThanOrEqual(3);
  });
});
