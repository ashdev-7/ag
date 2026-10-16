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
import { RACE_GUIDES, RACE_HELP, RACE_SCREEN } from '../content/help.tsx';
import { Panel, Story } from './Panel.tsx';
import { SpeedControl } from './Speed.tsx';
import { Dock } from './Dock.tsx';
import { raceStory } from '../views/narrate.ts';

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
  const [speed, setSpeed] = useState(preset.speed ?? 20);   // steps per second
  const optimsRef = useRef(optims);
  // per race, cleared by Reset: each racer's first loss, and who has diverged
  const start = useRef<number[]>([]);
  const gone = useRef(new Set<number>());
  optimsRef.current = optims;

  useEffect(() => {
    let dead = false;
    const made: Session[] = [];
    start.current = [];
    gone.current = new Set();
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
    // a racer that diverged stays stopped when the race resumes
    const going = sessions.filter((_, i) => !gone.current.has(i));
    if (!going.length) return;
    setRunning(true);
    let stopped = 0;
    sessions.forEach((s, i) => !gone.current.has(i) &&
      s.run({
        watch: { stats: [], tensors: [{ nodeId: race.x, which: 'value' }, { nodeId: race.y, which: 'value' }] },
        stepsPerSecond: speed,
        maxSteps: race.maxSteps,
        onFrame: (f) => {
          const x = f.tensors[0].data[0], y = f.tensors[1].data[0];
          // diverged: the loss has grown a thousandfold past where it started.
          // Waiting for the numbers to overflow could outlast the race, and
          // the point is already made
          const s0 = (start.current[i] ??= f.loss);
          const diverged = f.loss > s0 + 1000 * Math.max(1, Math.abs(s0));
          setRacers((rs) => rs.map((r, k) => (k !== i ? r : {
            ...r, path: [...r.path, [x, y]], steps: [...r.steps, f.step], loss: [...r.loss, f.loss],
            done: diverged ? 'diverged' : r.done,
          })));
          if (diverged) {
            gone.current.add(i);
            s.pause();
            if (++stopped === going.length) setRunning(false);
          }
        },
        onStopped: (st) => {
          // the button goes back to "Race" once every racer has stopped
          if (++stopped === going.length) setRunning(false);
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

  // a new speed applies at once: restart the running race at it
  useEffect(() => {
    if (running) run();
  }, [speed]); // eslint-disable-line react-hooks/exhaustive-deps

  const setOptim = (i: number, o: OptimConfig) => {
    setOptims((os) => os.map((x, k) => (k === i ? o : x)));
    sessions[i]?.setOptimizer(o);
    setRacers((rs) => rs.map((r, k) => (k === i ? { ...r, optim: o } : r)));
  };

  if (!surface || !racers.length) return <div className="loading" role="status" aria-label="Sweeping the loss surface"><span /><span /><span /></div>;
  let lo = Infinity, hi = -Infinity;
  surface.data.forEach((v) => { if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } });

  const guide = RACE_GUIDES[preset.id];
  return (
    <div className="race-view">
      <header className="lab-head">
        <div>
          <span className="eyebrow">Race optimizers</span>
          <h1>{preset.title}</h1>
          <p className="lede">{RACE_SCREEN} {guide?.idea}</p>
          {guide && <p className="formula-line">{guide.formula}</p>}
        </div>
        <div className="try">
          <span className="eyebrow">Try this</span>
          <ol>
            <li>Press <b>Race</b>: all four start at ○ and head for the lowest point ★.</li>
            {guide?.watch.map((w, k) => <li key={k}>{w}</li>)}
            <li>Change an <b>lr</b> (step size), press Reset, race again.</li>
          </ol>
        </div>
      </header>
      <div className="controls toolbar">
        <button className="primary" onClick={running ? pause : run}>{running ? 'Pause' : 'Race'}</button>
        <button onClick={() => { pause(); setEpoch((e) => e + 1); }}>Reset</button>
        <SpeedControl value={speed} onChange={setSpeed} />
        <span className="counter step-count">step <b>{racers[0].steps.at(-1) ?? 0}</b> of {race.maxSteps}</span>
        <span className="hint">All four start at ({fmt(racers[0].path[0][0])}, {fmt(racers[0].path[0][1])}).</span>
      </div>
      <Dock>
        <button className="primary" onClick={running ? pause : run}>{running ? 'Pause' : 'Race'}</button>
        <button onClick={() => { pause(); setEpoch((e) => e + 1); }}>Reset</button>
        <span className="counter">step <b>{racers[0].steps.at(-1) ?? 0}</b> of {race.maxSteps}</span>
      </Dock>
      <div className="race-optims">
        {optims.map((o, i) => (
          <OptimControls key={i} color={PALETTE[i]} optim={o} onChange={(n) => setOptim(i, n)}
            step={racers[i]?.steps.at(-1) ?? 0} loss={racers[i]?.loss.at(-1)} />
        ))}
      </div>
      <div className="race-grid">
        <Panel title="The landscape" sub={RACE_HELP.surface}>
          <div className="square big">
            <Canvas height={480} draw={(ctx, w, h) => {
              drawField(ctx, w, h, surface.data, surface.res, surfaceScale(lo, hi));
              drawPaths(ctx, w, h, racers.map((r, i) => ({ label: r.optim.kind, color: PALETTE[i], points: r.path })), race.domain);
              drawMarkers(ctx, w, h, race.domain, racers[0].path[0], guide?.goals ?? []);
            }} />
          </div>
          <Legend items={racers.map((r, i) => ({ label: `${r.optim.kind}${r.done === 'diverged' ? ' (diverged)' : ''}`, color: PALETTE[i] }))} />
        </Panel>
        <Panel title="How low each one has got" sub={RACE_HELP.loss}>
          <Canvas height={240} draw={(ctx, w, h) => drawLines(ctx, w, h,
            racers.map((r, i) => ({ label: r.optim.kind, color: PALETTE[i], x: r.steps, y: r.loss })), { logY: true, yLabel: 'loss (height)', xLabel: 'steps →' })} />
          <Story>{raceStory(racers.map((r) => ({ name: r.optim.kind, loss: r.loss, diverged: r.done === 'diverged' })))}</Story>
        </Panel>
      </div>
    </div>
  );
}

/** ○ at the start and ★ at each lowest point, labelled, so the goal is never a guess. */
function drawMarkers(ctx: CanvasRenderingContext2D, w: number, h: number, d: { x0: number; x1: number; y0: number; y1: number },
  start: [number, number], goals: [number, number][]) {
  const X = (x: number) => ((x - d.x0) / (d.x1 - d.x0)) * w, Y = (y: number) => (1 - (y - d.y0) / (d.y1 - d.y0)) * h;
  const label = (text: string, x: number, y: number) => {
    ctx.font = '500 12px Inter, system-ui, sans-serif';
    ctx.lineWidth = 3;
    ctx.strokeStyle = 'rgba(255,255,255,0.9)';
    ctx.strokeText(text, x + 9, y + 4);
    ctx.fillStyle = '#1d1d1f';
    ctx.fillText(text, x + 9, y + 4);
  };
  ctx.lineWidth = 2;
  ctx.strokeStyle = '#1d1d1f';
  ctx.fillStyle = '#fff';
  ctx.beginPath();
  ctx.arc(X(start[0]), Y(start[1]), 6, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  label('start', X(start[0]), Y(start[1]));
  for (const [gx, gy] of goals) {
    ctx.font = '18px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.lineWidth = 3;
    ctx.strokeStyle = 'rgba(255,255,255,0.9)';
    ctx.strokeText('★', X(gx), Y(gy) + 6);
    ctx.fillStyle = '#c8a400';
    ctx.fillText('★', X(gx), Y(gy) + 6);
    ctx.textAlign = 'left';
    label('lowest point', X(gx) + 2, Y(gy));
  }
}

