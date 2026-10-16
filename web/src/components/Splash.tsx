// The opening screen, once per visit, about two and a half seconds: the two
// moons' dots glide into place, then the field behind them "trains", from
// unsure white to a straight guess that bends into the right boundary, in
// the labs' own blue-white-orange. Then it lifts away and the home page's
// network plays (it waits for :root[data-ready]). A click or any key skips
// it; with prefers-reduced-motion it never shows.

import { useEffect, useRef, useState } from 'react';
import { divergingScale } from '../views/draw.ts';
import { BLUE, hex, ORANGE } from '../views/theme.ts';

const KEY = 'ag-splash-seen';
const EXIT_AT = 2300, EXIT_MS = 450;
const D = { x0: -1.5, x1: 2.5, y0: -1.0, y1: 1.5 };   // 1.6 : 1, like the canvas
const RES_X = 96, RES_Y = 60;

const ready = () => document.documentElement.setAttribute('data-ready', '');

function shouldShow() {
  if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) return false;
  try { return sessionStorage.getItem(KEY) === null; } catch { return true; }
}

// distance from (x, y) to a half circle of radius 1 around (cx, cy): the
// upper half (up = 1, the blue moon) or the lower half (up = -1, orange)
function distArc(x: number, y: number, cx: number, cy: number, up: number) {
  const dx = x - cx, dy = y - cy;
  if (dy * up >= 0) return Math.abs(Math.hypot(dx, dy) - 1);
  return Math.min(Math.hypot(dx - 1, dy), Math.hypot(dx + 1, dy));   // nearest tip
}
const distA = (x: number, y: number) => distArc(x, y, 0, 0, 1);
const distB = (x: number, y: number) => distArc(x, y, 1, 0.5, -1);

// the dots: a fixed pseudo-random draw, the same every visit
function points() {
  let s = 7;
  const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647);
  const pts: { x: number; y: number; sx: number; sy: number; c: number; at: number }[] = [];
  for (let c = 0; c < 2; ++c)
    for (let i = 0; i < 44; ++i) {
      const t = Math.PI * (i + rnd() * 0.6) / 44;
      const nx = (rnd() - 0.5) * 0.22, ny = (rnd() - 0.5) * 0.22;
      const x = c ? 1 - Math.cos(t) + nx : Math.cos(t) + nx;
      const y = c ? 0.5 - Math.sin(t) + ny : Math.sin(t) + ny;
      pts.push({ x, y, sx: D.x0 + rnd() * (D.x1 - D.x0), sy: D.y0 + rnd() * (D.y1 - D.y0), c, at: rnd() * 260 });
    }
  return pts;
}

const ease = (t: number) => 1 - Math.pow(1 - Math.min(1, Math.max(0, t)), 3);
const phase = (ms: number, from: number, to: number) => ease((ms - from) / (to - from));

export function Splash() {
  const [show, setShow] = useState(shouldShow);
  const [leaving, setLeaving] = useState(false);
  const canvas = useRef<HTMLCanvasElement>(null);
  const skipRef = useRef(() => {});

  useEffect(() => {
    if (!show) { ready(); return; }
    try { sessionStorage.setItem(KEY, '1'); } catch { /* fine: it shows again next load */ }
    const cv = canvas.current!;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = cv.clientWidth, h = cv.clientHeight;
    cv.width = w * dpr; cv.height = h * dpr;
    const ctx = cv.getContext('2d')!;
    ctx.scale(dpr, dpr);
    const tmp = document.createElement('canvas');
    tmp.width = RES_X; tmp.height = RES_Y;
    const tctx = tmp.getContext('2d')!;
    const img = tctx.createImageData(RES_X, RES_Y);
    const scale = divergingScale(1);
    const pts = points();
    const px = (x: number) => ((x - D.x0) / (D.x1 - D.x0)) * w;
    const py = (y: number) => (1 - (y - D.y0) / (D.y1 - D.y0)) * h;

    let raf = 0, done = false;
    const t0 = performance.now();
    const leave = () => {
      if (done) return;
      done = true;
      setLeaving(true);
      ready();
      setTimeout(() => setShow(false), EXIT_MS);
    };
    const frame = (now: number) => {
      const ms = now - t0;
      // the field: sharpness grows (k), and a straight first guess bends into the moons' boundary (b)
      const k = 5 * phase(ms, 450, 1150), b = phase(ms, 900, 1750);
      for (let j = 0; j < RES_Y; ++j)
        for (let i = 0; i < RES_X; ++i) {
          const x = D.x0 + ((i + 0.5) / RES_X) * (D.x1 - D.x0);
          const y = D.y1 - ((j + 0.5) / RES_Y) * (D.y1 - D.y0);
          const guess = 0.25 - y, curve = distA(x, y) - distB(x, y);
          const c = scale(0.55 * Math.tanh(k * ((1 - b) * guess + b * curve)));   // pale, so the dots stand out
          img.data.set([c[0], c[1], c[2], 255], (j * RES_X + i) * 4);
        }
      tctx.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(tmp, 0, 0, w, h);
      // the dots glide in from scattered starts
      const r = Math.max(2.5, w / 150);
      for (const p of pts) {
        const t = phase(ms, p.at, p.at + 650);
        if (t <= 0) continue;
        ctx.globalAlpha = Math.min(1, t * 2);
        ctx.beginPath();
        ctx.arc(px(p.sx + (p.x - p.sx) * t), py(p.sy + (p.y - p.sy) * t), r, 0, Math.PI * 2);
        ctx.fillStyle = p.c ? hex(ORANGE) : hex(BLUE);
        ctx.fill();
        ctx.strokeStyle = 'rgba(232,238,249,0.9)';
        ctx.lineWidth = 1.25;
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
      if (ms >= EXIT_AT) leave();
      else raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    const skip = () => { cancelAnimationFrame(raf); leave(); };
    skipRef.current = skip;
    window.addEventListener('keydown', skip);
    return () => { cancelAnimationFrame(raf); window.removeEventListener('keydown', skip); };
  }, [show]);

  if (!show) return null;
  return (
    <div className={`splash${leaving ? ' leaving' : ''}`} onClick={() => skipRef.current()} aria-hidden="true">
      <div className="splash-inner">
        <canvas ref={canvas} className="splash-field" />
        <div className="splash-brand">ag<span>·glass</span></div>
        <div className="splash-line">See inside a neural network</div>
      </div>
    </div>
  );
}
