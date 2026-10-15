// The optimizer race: one session per optimizer on the same two-parameter
// loss, all starting at the same point. The background is the loss surface
// itself, swept by the engine; each frame adds a point to each path.

import { useCallback, useEffect, useRef, useState } from 'react';
import type { EngineClient, Session } from '../engine/client.ts';
import type { OptimConfig } from '../engine/protocol.ts';
import type { Preset } from '../graph/presets.ts';
import { drawField, drawLines, drawPaths, fmt, PALETTE, surfaceScale } from '../views/draw.ts';
import { Canvas } from './Canvas.tsx';
import { describe } from './ErrorBanner.tsx';
import type { Banner } from './ErrorBanner.tsx';
import { Legend, OptimControls } from './TrainView.tsx';
import { GuideCard, Help } from './Guide.tsx';
import { OPTIM_HELP, RACE_GUIDES, RACE_HELP } from '../content/help.tsx';

interface Racer { optim: OptimConfig; path: [number, number][]; steps: number[]; loss: number[]; done?: string }

export function RaceView(props: { client: EngineClient; preset: Preset; seed: number; onBanner: (b?: Banner) => void }) {
  const { client, preset, onBanner } = props;
  const race = preset.race!;
  const [optims, setOptims] = useState<OptimConfig[]>(race.optimizers);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [racers, setRacers] = useState<Racer[]>([]);
  const [surface, setSurface] = useState<{ res: number; data: Float32Array } | null>(null);
  const [running, setRunning] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const optimsRef = useRef(optims);
  optimsRef.current = optims;

  useEffect(() => {
    let dead = false;
    const made: Session[] = [];
    (async () => {
      for (const optim of optimsRef.current) {
        const r = await client.createSession(preset.spec, { ...preset.config, optim });
        if (dead) { if (r.ok) r.value.destroy(); return; }
        if (!r.ok) return onBanner({ kind: 'error', text: describe(r.error), error: r.error });
        made.push(r.value);
      }
      const start = await made[0].exportParams();
      const s = await made[0].lossSurface({ nodeId: race.x, element: 0 }, { nodeId: race.y, element: 0 }, { ...race.domain, res: 100 });
      if (dead) return;
      if (s.ok) setSurface(s.value);
      const p0: [number, number] = start.ok ? [start.value[0], start.value[1]] : [0, 0];
      setRacers(optimsRef.current.map((optim) => ({ optim, path: [p0], steps: [], loss: [] })));
      setSessions(made);
    })();
    return () => {
      dead = true;
      made.forEach((s) => s.destroy());
      setSessions([]);
      setRunning(false);
    };
  }, [client, preset, race, epoch, onBanner]);

  const run = () => {
    setRunning(true);
    let stopped = 0;
    sessions.forEach((s, i) =>
      s.run({
        watch: { stats: [], tensors: [{ nodeId: race.x, which: 'value' }, { nodeId: race.y, which: 'value' }] },
        stepsPerFrame: 2,
        maxSteps: 3000,
        onFrame: (f) => {
          const x = f.tensors[0].data[0], y = f.tensors[1].data[0];
          setRacers((rs) => rs.map((r, k) => (k !== i ? r : {
            ...r, path: [...r.path, [x, y]], steps: [...r.steps, f.step], loss: [...r.loss, f.loss],
          })));
        },
        onStopped: (st) => {
          // the button goes back to "Race" once every racer has stopped
          if (++stopped === sessions.length) setRunning(false);
          setRacers((rs) => rs.map((r, k) => (k !== i ? r : { ...r, done: st.reason === 'fault' ? 'diverged' : 'finished' })));
          if (st.error && st.error.code !== 'NonFinite') onBanner({ kind: 'error', text: describe(st.error), error: st.error });
        },
      }),
    );
  };

  const pause = useCallback(() => {
    setRunning(false);
    sessions.forEach((s) => s.pause());
  }, [sessions]);

  // a hidden tab stops the race, as it stops training
  useEffect(() => {
    const onHide = () => document.hidden && pause();
    document.addEventListener('visibilitychange', onHide);
    return () => document.removeEventListener('visibilitychange', onHide);
  }, [pause]);

  const setOptim = (i: number, o: OptimConfig) => {
    setOptims((os) => os.map((x, k) => (k === i ? o : x)));
    sessions[i]?.setOptimizer(o);
    setRacers((rs) => rs.map((r, k) => (k === i ? { ...r, optim: o } : r)));
  };

  if (!surface || !racers.length) return <div className="loading">Sweeping the loss surface…</div>;
  let lo = Infinity, hi = -Infinity;
  surface.data.forEach((v) => { if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } });

  return (
    <div className="race-view">
      <GuideCard title={preset.title} guide={RACE_GUIDES[preset.id]}>
        {RACE_GUIDES[preset.id] && <p>The loss: {RACE_GUIDES[preset.id].formula}</p>}
      </GuideCard>
      <div className="toolbar">
        <button className="primary" onClick={running ? pause : run}>{running ? 'Pause' : 'Race'}</button>
        <button onClick={() => { pause(); setEpoch((e) => e + 1); }}>Reset</button>
        <span className="muted">Each optimizer starts at ({fmt(racers[0].path[0][0])}, {fmt(racers[0].path[0][1])}). Change a learning rate and reset to try again.</span>
      </div>
      <div className="optim-row">
        {optims.map((o, i) => (
          <OptimControls key={i} color={PALETTE[i]} optim={o} onChange={(n) => setOptim(i, n)}
            step={racers[i]?.steps.at(-1) ?? 0} loss={racers[i]?.loss.at(-1)} />
        ))}
      </div>
      {[...new Set(optims.map((o) => o.kind))].map((k) => <Help key={k}>{OPTIM_HELP[k]}</Help>)}
      <div className="race-grid">
        <section className="panel">
          <h4>Loss surface <span className="muted">x horizontal, y vertical</span></h4>
          <Help>{RACE_HELP.surface}</Help>
          <div className="square big">
            <Canvas height={460} draw={(ctx, w, h) => {
              drawField(ctx, w, h, surface.data, surface.res, surfaceScale(lo, hi));
              drawPaths(ctx, w, h, racers.map((r, i) => ({ label: r.optim.kind, color: PALETTE[i], points: r.path })), race.domain);
            }} />
          </div>
          <Legend items={racers.map((r, i) => ({ label: `${r.optim.kind}${r.done === 'diverged' ? ' (diverged)' : ''}`, color: PALETTE[i] }))} />
        </section>
        <section className="panel">
          <h4>Loss over steps</h4>
          <Help>{RACE_HELP.loss}</Help>
          <Canvas height={220} draw={(ctx, w, h) => drawLines(ctx, w, h,
            racers.map((r, i) => ({ label: r.optim.kind, color: PALETTE[i], x: r.steps, y: r.loss })), { logY: true })} />
        </section>
      </div>
    </div>
  );
}
