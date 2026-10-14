// Shown when the engine cannot run in this browser: what the preset does,
// recorded beforehand by scripts/record.ts with the same engine.

import { useEffect, useState } from 'react';
import type { Preset } from '../graph/presets.ts';
import { divergingScale, drawField, drawLines, drawPaths, drawPoints, PALETTE, surfaceScale } from '../views/draw.ts';
import { Canvas } from './Canvas.tsx';
import { Legend } from './TrainView.tsx';

interface TrainRecording {
  every: number; loss: (number | null)[]; val: (number | null)[]; gradNorms: Record<string, (number | null)[]>;
  grid: { res: number; data: (number | null)[] }; points: { x: number[]; y: number[]; yDim: number };
}
interface RaceRecording { surface: { res: number; data: (number | null)[] }; paths: { kind: string; points: [number, number][] }[] }

const DOMAIN = { x0: -1.5, x1: 1.5, y0: -1.5, y1: 1.5 };

export function Recorded({ preset }: { preset: Preset }) {
  const [rec, setRec] = useState<TrainRecording | RaceRecording | null | 'missing'>(null);
  useEffect(() => {
    if (preset.mode === 'step') return setRec('missing');
    fetch(`${import.meta.env.BASE_URL}recordings/${preset.id}.json`)
      .then((r) => (r.ok ? r.json() : 'missing'))
      .then(setRec, () => setRec('missing'));
  }, [preset]);

  if (rec === null) return <div className="loading">Loading the recording…</div>;
  if (rec === 'missing')
    return <div className="loading">Stepping needs the live engine, which could not start in this browser. Try a recent Chrome, Firefox or Safari.</div>;

  if ('surface' in rec) {
    const race = preset.race!;
    const vals = rec.surface.data.filter((v): v is number => v !== null);
    const lo = Math.min(...vals), hi = Math.max(...vals);
    return (
      <div className="recorded">
        <p className="muted">A recorded race. Live controls need the engine.</p>
        <div className="square big">
          <Canvas height={460} draw={(ctx, w, h) => {
            drawField(ctx, w, h, rec.surface.data, rec.surface.res, surfaceScale(lo, hi));
            drawPaths(ctx, w, h, rec.paths.map((p, i) => ({ label: p.kind, color: PALETTE[i], points: p.points })), race.domain);
          }} />
        </div>
        <Legend items={rec.paths.map((p, i) => ({ label: p.kind, color: PALETTE[i] }))} />
      </div>
    );
  }

  const steps = rec.loss.map((_, i) => (i + 1) * rec.every);
  const labels = Array.from({ length: rec.points.y.length / rec.points.yDim }, (_, i) => rec.points.y[i * rec.points.yDim + 1]);
  return (
    <div className="recorded train-grid">
      <p className="muted span">A recorded run of {steps.at(-1)} steps. Live training needs the engine.</p>
      <section className="panel">
        <h4>Decision boundary at the end</h4>
        <div className="square">
          <Canvas height={260} draw={(ctx, w, h) => {
            drawField(ctx, w, h, rec.grid.data.map((p) => (p === null ? null : 2 * p - 1)), rec.grid.res, divergingScale(1));
            drawPoints(ctx, w, h, rec.points.x, labels, DOMAIN);
          }} />
        </div>
      </section>
      <section className="panel">
        <h4>Loss</h4>
        <Canvas height={180} draw={(ctx, w, h) => drawLines(ctx, w, h, [
          { label: 'train', color: PALETTE[0], x: steps, y: rec.loss },
          { label: 'validation', color: PALETTE[0], x: steps, y: rec.val, dash: true },
        ], { logY: true })} />
      </section>
      <section className="panel">
        <h4>Gradient norm per layer</h4>
        <Canvas height={180} draw={(ctx, w, h) => drawLines(ctx, w, h,
          Object.entries(rec.gradNorms).map(([k, y], i) => ({ label: k, color: PALETTE[(i + 2) % PALETTE.length], x: steps, y })), { logY: true })} />
        <Legend items={Object.keys(rec.gradNorms).map((k, i) => ({ label: k, color: PALETTE[(i + 2) % PALETTE.length] }))} />
      </section>
    </div>
  );
}
