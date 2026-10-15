// Live training. The worker trains and sends one frame at a time; each frame
// is a consistent snapshot (loss, per-node statistics, the decision boundary),
// drawn here and then acknowledged. A second run with another optimizer can
// train beside the first from identical starting weights.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { EngineClient, Session } from '../engine/client.ts';
import type { Dataset, EngineError, Frame, Grid, OptimConfig, Stats } from '../engine/protocol.ts';
import type { Preset } from '../graph/presets.ts';
import { divergingScale, drawField, drawHistogram, drawLines, drawPoints, drawTensor, fmt, layerColor, maxAbs, PALETTE } from '../views/draw.ts';
import type { Series } from '../views/draw.ts';
import type { Highlight, UrlState } from '../views/graph.ts';
import { Canvas } from './Canvas.tsx';
import { describe } from './ErrorBanner.tsx';
import type { Banner } from './ErrorBanner.tsx';
import { GraphCanvas } from './GraphCanvas.tsx';
import { GuideCard, Help } from './Guide.tsx';
import { OPTIM_HELP, PANEL_HELP, TRAIN_GUIDES, TRAIN_SETUP } from '../content/help.tsx';

export const OPTIMIZERS: OptimConfig['kind'][] = ['sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'adamw'];
const DOMAIN = { x0: -1.5, x1: 1.5, y0: -1.5, y1: 1.5 };
const MAX_POINTS = 1500;

interface History { step: number[]; loss: number[]; val: (number | null)[]; norms: Record<string, number[]> }
const emptyHistory = (): History => ({ step: [], loss: [], val: [], norms: {} });

/** Keep charts bounded on long runs: past the cap, drop every other point. */
function thin(h: History): History {
  if (h.step.length <= MAX_POINTS) return h;
  const keep = (_: unknown, i: number) => i % 2 === 0;
  const norms: Record<string, number[]> = {};
  for (const k in h.norms) norms[k] = h.norms[k].filter(keep);
  return { step: h.step.filter(keep), loss: h.loss.filter(keep), val: h.val.filter(keep), norms };
}

interface RunView {
  optim: OptimConfig;
  history: History;
  frame?: Frame;
  grid?: { res: number; data: Float32Array };
  values: Record<string, Stats>;
  grads: Record<string, Stats>;
  step: number;
  sps: number;
  fault?: EngineError;
}

/**
 * `value`, passed on at most every `ms`; the latest value always arrives.
 * The network diagram is the most expensive thing on the page to redraw, and
 * ten updates a second read as smoothly as sixty for numbers on a graph.
 */
function useThrottled<T>(value: T, ms: number): T {
  const [shown, setShown] = useState(value);
  const last = useRef(0);
  useEffect(() => {
    const show = () => {
      last.current = performance.now();
      setShown(value);
    };
    const wait = last.current + ms - performance.now();
    if (wait <= 0) return show();
    const t = setTimeout(show, wait);
    return () => clearTimeout(t);
  }, [value, ms]);
  return shown;
}

const newRun = (optim: OptimConfig): RunView => ({ optim, history: emptyHistory(), values: {}, grads: {}, step: 0, sps: 0 });

function defaultOther(kind: OptimConfig['kind']): OptimConfig {
  return kind === 'sgd' ? { kind: 'adam', lr: 0.02 } : { kind: 'sgd', lr: 0.3 };
}

export function TrainView(props: {
  client: EngineClient;
  preset: Preset;
  seed: number;
  url: UrlState;
  onUrl: (u: Partial<UrlState>) => void;
  onBanner: (b?: Banner) => void;
}) {
  const { client, preset, seed, onBanner } = props;
  const initial: OptimConfig = props.url.opt && OPTIMIZERS.includes(props.url.opt as OptimConfig['kind'])
    ? { ...preset.config.optim, kind: props.url.opt as OptimConfig['kind'], lr: props.url.lr ?? preset.config.optim.lr }
    : preset.config.optim;
  const [optims, setOptims] = useState<OptimConfig[]>(() => {
    const cmp = props.url.cmp && OPTIMIZERS.includes(props.url.cmp as OptimConfig['kind'])
      ? { ...defaultOther(initial.kind), kind: props.url.cmp as OptimConfig['kind'] } : null;
    return cmp ? [initial, cmp] : [initial];
  });
  const [sessions, setSessions] = useState<Session[]>([]);
  const [runs, setRuns] = useState<RunView[]>([]);
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(10);
  const [data, setData] = useState<Dataset | null>(null);
  const [selected, setSelected] = useState<string | undefined>();
  const [inspect, setInspect] = useState<{ value?: Float32Array; grad?: Float32Array; m?: Float32Array; v?: Float32Array; shape: number[] } | null>(null);
  const [epoch, setEpoch] = useState(0);   // bumps on reset, so sessions are rebuilt
  const comparing = optims.length > 1;

  const grid: Grid | undefined = preset.boundary ? { ...DOMAIN, res: 60, nodeId: preset.boundary.nodeId, element: preset.boundary.element } : undefined;
  const allIds = useMemo(() => preset.spec.nodes.map((n) => n.id), [preset]);

  // build one session per optimizer; the second imports the first's weights
  // so both start from exactly the same place
  useEffect(() => {
    let dead = false;
    const made: Session[] = [];
    (async () => {
      for (const optim of optims) {
        const r = await client.createSession(preset.spec, { ...preset.config, seed, optim });
        if (dead) { if (r.ok) r.value.destroy(); return; }
        if (!r.ok) return onBanner({ kind: 'error', text: describe(r.error), error: r.error });
        made.push(r.value);
      }
      if (made.length > 1) {
        const p = await made[0].exportParams();
        if (p.ok) await made[1].importParams(p.value);
      }
      const d = await made[0].dataset();
      if (dead) return;
      if (d.ok) setData(d.value);
      const views = await Promise.all(made.map(async (s, i) => {
        const v = newRun(optims[i]);
        const [vs, gs] = await Promise.all([s.stats(allIds, 'value'), s.stats(allIds, 'grad')]);
        if (vs.ok) v.values = vs.value;
        if (gs.ok) v.grads = gs.value;
        if (grid) {
          const g = await s.evalGrid(grid);
          if (g.ok) v.grid = g.value;
        }
        return v;
      }));
      if (dead) return;
      setSessions(made);
      setRuns(views);
    })();
    return () => {
      dead = true;
      made.forEach((s) => s.destroy());
      setSessions([]);
      setRunning(false);
    };
    // optimizer changes within a run go through setOptimizer, not here
  }, [client, preset, seed, comparing, epoch]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => props.onUrl({ opt: optims[0].kind, lr: optims[0].lr, cmp: optims[1]?.kind }), [optims]); // eslint-disable-line react-hooks/exhaustive-deps

  const onFrame = useCallback((i: number, f: Frame) => {
    setRuns((rs) => rs.map((r, k) => {
      if (k !== i) return r;
      const h = r.history;
      const norms = { ...h.norms };
      for (const l of preset.layers) norms[l.label] = [...(norms[l.label] ?? []), f.gradStats[l.weight]?.l2 ?? 0];
      const history = thin({
        step: [...h.step, f.step], loss: [...h.loss, f.loss],
        val: [...h.val, f.valLoss ?? (h.val.length ? h.val[h.val.length - 1] : null)], norms,
      });
      return { ...r, history, frame: f, grid: f.grid ?? r.grid, values: f.valueStats, grads: f.gradStats, step: f.step, sps: f.stepsPerSecond };
    }));
  }, [preset]);

  const start = useCallback(() => {
    if (!sessions.length) return;
    setRunning(true);
    onBanner(undefined);
    sessions.forEach((s, i) => {
      s.run({
        watch: { stats: allIds, tensors: [], grid, gridEvery: 2, valEvery: 5 },
        stepsPerFrame: speed,
        onFrame: (f) => onFrame(i, f),
        onStopped: (st) => {
          setRunning(false);
          sessions.forEach((o) => o.pause());
          if (st.error) {
            setRuns((rs) => rs.map((r, k) => (k === i ? { ...r, fault: st.error } : r)));
            onBanner({ kind: 'error', text: `Training paused. ${describe(st.error)}`, error: st.error });
          }
        },
      });
    });
  }, [sessions, allIds, grid, speed, onFrame, onBanner]);

  const pause = useCallback(() => {
    setRunning(false);
    sessions.forEach((s) => s.pause());
  }, [sessions]);

  // a hidden tab stops training; it resumes when asked
  useEffect(() => {
    const onHide = () => document.hidden && pause();
    document.addEventListener('visibilitychange', onHide);
    return () => document.removeEventListener('visibilitychange', onHide);
  }, [pause]);

  // speed changes apply to the next frame
  useEffect(() => {
    if (running) start();
  }, [speed]); // eslint-disable-line react-hooks/exhaustive-deps

  const stepOnce = async () => {
    pause();
    for (let i = 0; i < sessions.length; ++i) {
      const s = sessions[i];
      const r = await s.train(1);
      if (!r.ok) return onBanner({ kind: 'error', text: describe(r.error), error: r.error });
      if (r.value.fault) onBanner({ kind: 'error', text: describe(r.value.fault), error: r.value.fault });
      const [vs, gs, g] = await Promise.all([s.stats(allIds, 'value'), s.stats(allIds, 'grad'), grid ? s.evalGrid(grid) : Promise.resolve(null)]);
      setRuns((rs) => rs.map((run, k) => k !== i ? run : {
        ...run, step: r.value.step, values: vs.ok ? vs.value : run.values, grads: gs.ok ? gs.value : run.grads,
        grid: g && g.ok ? g.value : run.grid,
        history: { ...run.history, step: [...run.history.step, r.value.step], loss: [...run.history.loss, r.value.loss], val: [...run.history.val, run.history.val.at(-1) ?? null] },
      }));
    }
  };

  const reset = () => {
    pause();
    onBanner(undefined);
    setEpoch((e) => e + 1);
  };

  const setOptim = (i: number, o: OptimConfig) => {
    setOptims((os) => os.map((x, k) => (k === i ? o : x)));
    sessions[i]?.setOptimizer(o).then((r) => !r.ok && onBanner({ kind: 'error', text: r.error.message, error: r.error }));
  };

  const toggleCompare = () => {
    pause();
    setOptims((os) => (os.length > 1 ? [os[0]] : [os[0], defaultOther(os[0].kind)]));
  };

  // the selected node's tensors, and for a parameter its optimizer state
  const lastStep = runs[0]?.step ?? 0;
  useEffect(() => {
    const s = sessions[0];
    if (!s || !selected) return setInspect(null);
    if (running && lastStep % 5 !== 0) return;
    let dead = false;
    const isParam = preset.spec.nodes.find((n) => n.id === selected)?.op === 'param';
    Promise.all([s.read(selected, 'value'), s.read(selected, 'grad'),
      isParam ? s.readOptim(selected, 0) : null, isParam ? s.readOptim(selected, 1) : null]).then(([v, g, m, vv]) => {
      if (dead || !v.ok) return;
      setInspect({ shape: v.value.shape, value: v.value.data, grad: g.ok ? g.value.data : undefined,
        m: m && m.ok ? m.value.data : undefined, v: vv && vv.ok ? vv.value.data : undefined });
    });
    return () => { dead = true; };
  }, [sessions, selected, lastStep, running, preset]);

  const highlight: Highlight = useMemo(() => ({ valueDone: new Set(allIds), gradDone: new Set(allIds), direction: 'idle' }), [allIds]);
  const shapes = useMemo(() => Object.fromEntries((sessions[0]?.info.nodes ?? []).map((n) => [n.id, n.shape])), [sessions]);
  const labels = useMemo(() => (data ? Array.from({ length: data.n }, (_, i) => (data.y[i * data.yDim + 1] > 0.5 ? 1 : 0)) : []), [data]);

  const nodeStats = useMemo(() => ({ values: runs[0]?.values ?? {}, grads: runs[0]?.grads ?? {} }), [runs]);
  const graphStats = useThrottled(nodeStats, running ? 100 : 0);

  if (!runs.length) return <div className="loading">Building the network…</div>;
  const a = runs[0];

  const lossSeries: Series[] = runs.flatMap((r, i) => [
    { label: `${r.optim.kind} train`, color: PALETTE[i], x: r.history.step, y: r.history.loss },
    { label: `${r.optim.kind} validation`, color: PALETTE[i], x: r.history.step, y: r.history.val, dash: true },
  ]);
  const normSeries: Series[] = preset.layers.map((l, i) => ({
    label: l.label, color: layerColor(i, preset.layers.length), x: a.history.step, y: a.history.norms[l.label] ?? [],
  }));

  return (
    <div className="train-view">
      <GuideCard title={preset.title} presetId={preset.id} guide={TRAIN_GUIDES[preset.id]}>
        <p>{TRAIN_SETUP}</p>
      </GuideCard>
      <div className="toolbar">
        <button className="primary" onClick={running ? pause : start}>{running ? 'Pause' : 'Run'}</button>
        <button onClick={stepOnce} disabled={running}>Step</button>
        <button onClick={reset}>Reset</button>
        <label className="field">steps per frame
          <input type="range" min={1} max={100} value={speed} onChange={(e) => setSpeed(Number(e.target.value))} />
          <span className="counter">{speed}</span>
        </label>
        <span className="counter">step {a.step}{a.sps > 0 ? ` · ${Math.round(a.sps)}/s` : ''}</span>
        <label className="field"><input type="checkbox" checked={comparing} onChange={toggleCompare} /> compare two optimizers</label>
      </div>

      <Help>{PANEL_HELP.controls}</Help>
      <div className="optim-row">
        {optims.map((o, i) => (
          <OptimControls key={i} color={PALETTE[i]} optim={o} onChange={(n) => setOptim(i, n)}
            step={runs[i]?.step ?? 0} loss={runs[i]?.history.loss.at(-1)} fault={runs[i]?.fault} />
        ))}
      </div>
      {[...new Set(optims.map((o) => o.kind))].map((k) => <Help key={k}>{OPTIM_HELP[k]}</Help>)}

      <div className="train-grid">
        <section className="panel graph-panel">
          <h4>Network</h4>
          <Help>{PANEL_HELP.network}</Help>
          <div className="graph-wrap small">
            <GraphCanvas spec={preset.spec} shapes={shapes} highlight={highlight} values={graphStats.values} grads={graphStats.grads}
              selected={selected} onSelect={setSelected} edgeWeights faultNode={a.fault?.nodeId} />
          </div>
        </section>

        {grid && (
          <section className="panel boundary-panel">
            <h4>Decision boundary</h4>
            <Help>{PANEL_HELP.boundary}</Help>
            <div className="boundaries">
              {runs.map((r, i) => (
                <div key={i} className="boundary">
                  {comparing && <div className="muted" style={{ color: PALETTE[i] }}>{r.optim.kind}</div>}
                  <div className="square">
                    <Canvas height={260} draw={(ctx, w, h) => {
                      if (r.grid) drawField(ctx, w, h, r.grid.data.map((p) => 2 * p - 1), r.grid.res, divergingScale(1));
                      if (data) drawPoints(ctx, w, h, data.x, labels, DOMAIN);
                    }} />
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        <section className="panel">
          <h4>Loss</h4>
          <Help>{PANEL_HELP.loss}</Help>
          <Canvas height={180} draw={(ctx, w, h) => drawLines(ctx, w, h, lossSeries, { logY: true })} />
          <Legend items={runs.map((r, i) => ({ label: r.optim.kind, color: PALETTE[i] }))} />
        </section>

        <section className="panel">
          <h4>Gradient norm per layer <span className="muted">{comparing ? runs[0].optim.kind : ''}</span></h4>
          <Help>{PANEL_HELP.gradNorm}</Help>
          <Canvas height={180} draw={(ctx, w, h) => drawLines(ctx, w, h, normSeries, { logY: true })} />
          <Legend items={preset.layers.map((l, i) => ({ label: l.label, color: layerColor(i, preset.layers.length) }))} />
        </section>

        <section className="panel">
          <h4>Activations per layer</h4>
          <Help>{PANEL_HELP.acts}</Help>
          <div className="hists">
            {preset.layers.map((l) => {
              const s = a.values[l.act];
              return (
                <div key={l.label} className="hist">
                  <div className="muted">{l.label} {s ? `(${Math.round(s.fracZero * 100)}% zero)` : ''}</div>
                  {s && <Canvas height={60} draw={(ctx, w, h) => drawHistogram(ctx, w, h, s, s.fracZero > 0.9 ? '#c0392b' : '#5e3c99')} />}
                </div>
              );
            })}
          </div>
        </section>

        <section className="panel">
          <h4>{selected ? `Inside ${selected}` : 'Inspect'}</h4>
          <Help>{PANEL_HELP.inspect}</Help>
          {!selected && <p className="muted">Click a purple weight box in the network to look inside it.</p>}
          {selected && inspect && (
            <div className="inspect-grid">
              {([['value', inspect.value], ['gradient', inspect.grad], ['first moment (velocity)', inspect.m], ['second moment', inspect.v]] as const)
                .filter(([, d]) => d)
                .map(([label, d]) => (
                  <div key={label}>
                    <div className="muted">{label} <span>max |x| {fmt(maxAbs(d!))}</span></div>
                    <Canvas height={Math.min(140, Math.max(24, (inspect.shape.length >= 2 ? inspect.shape[0] : 1) * 14))}
                      draw={(ctx, w, h) => drawTensor(ctx, w, h, d!, inspect.shape, divergingScale(maxAbs(d!)))} />
                  </div>
                ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

export function Legend({ items }: { items: { label: string; color: string }[] }) {
  return (
    <div className="legend">
      {items.map((it) => (
        <span key={it.label}><i style={{ background: it.color }} />{it.label}</span>
      ))}
    </div>
  );
}

export function OptimControls(props: {
  optim: OptimConfig; color: string; onChange: (o: OptimConfig) => void; step: number; loss?: number; fault?: EngineError;
}) {
  const o = props.optim;
  return (
    <div className="optim" style={{ borderColor: props.color }}>
      <select value={o.kind} onChange={(e) => props.onChange({ ...o, kind: e.target.value as OptimConfig['kind'] })} aria-label="Optimizer">
        {OPTIMIZERS.map((k) => <option key={k} value={k}>{k}</option>)}
      </select>
      <label className="field">lr
        <input type="number" step="any" min={0.0001} max={10} value={o.lr}
          onChange={(e) => { const lr = Number(e.target.value); if (lr > 0 && lr <= 10) props.onChange({ ...o, lr }); }} />
      </label>
      <label className="field">clip
        <input type="number" step="any" min={0} value={o.clipNorm ?? 0}
          onChange={(e) => { const c = Number(e.target.value); if (c >= 0) props.onChange({ ...o, clipNorm: c }); }} />
      </label>
      {props.loss !== undefined && <span className="muted">loss {fmt(props.loss)}</span>}
      {props.fault && <span className="bad">stopped: nan</span>}
    </div>
  );
}
