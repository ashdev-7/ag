// Live training. The worker trains and sends one frame at a time; each frame
// is a consistent snapshot (loss, per-node statistics, the decision boundary),
// drawn here and then acknowledged. A second run with another optimizer can
// train beside the first from identical starting weights.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { EngineClient, Session } from '../engine/client.ts';
import { SpeedControl } from './Speed.tsx';
import { Dock } from './Dock.tsx';
import type { Dataset, EngineError, Frame, Grid, OptimConfig, Stats } from '../engine/protocol.ts';
import type { Preset } from '../graph/presets.ts';
import { divergingScale, drawField, drawHistogram, drawLines, drawPoints, drawTensor, fmt, layerColor, maxAbs, PALETTE } from '../views/draw.ts';
import type { Series } from '../views/draw.ts';
import type { Highlight, UrlState } from '../views/graph.ts';
import { Canvas } from './Canvas.tsx';
import { describe } from './ErrorBanner.tsx';
import type { Banner } from './ErrorBanner.tsx';
import { GraphCanvas } from './GraphCanvas.tsx';
import { fetchNeurons, NeuronDiagram } from './NeuronDiagram.tsx';
import type { NeuronState } from './NeuronDiagram.tsx';
import { OPTIM_HELP } from '../content/help.tsx';
import { OPTIMIZER_KEY, PANEL, TRAIN_TASK, TRAINING_LOOP } from '../content/train-text.tsx';
import { Panel, Story } from './Panel.tsx';
import { accuracy, activationStory, boundaryStory, gradStory, lossStory, trend } from '../views/narrate.ts';

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

/** Steps at which the boundary is photographed for the filmstrip. */
const FILM_STEPS = [0, 25, 100, 400, 1600];

/** Exponential moving average, for reading a trend through batch noise. */
function smooth(ys: number[], a = 0.15): number[] {
  let m = ys[0] ?? 0;
  return ys.map((y) => (m = Number.isFinite(y) ? a * y + (1 - a) * m : m));
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
  const [speed, setSpeed] = useState(preset.speed ?? 10);   // steps per second
  const [data, setData] = useState<Dataset | null>(null);
  const [selected, setSelected] = useState<string | undefined>();
  const [inspect, setInspect] = useState<{ value?: Float32Array; grad?: Float32Array; m?: Float32Array; v?: Float32Array; shape: number[] } | null>(null);
  const [epoch, setEpoch] = useState(0);   // bumps on reset, so sessions are rebuilt
  const [netView, setNetView] = useState<'neurons' | 'ops'>('neurons');
  const [neurons, setNeurons] = useState<NeuronState | null>(null);
  const [film, setFilm] = useState<{ step: number; grid: { res: number; data: Float32Array } }[]>([]);
  const comparing = optims.length > 1;

  const grid: Grid | undefined = useMemo(() => (preset.boundary
    ? { ...DOMAIN, res: 60, nodeId: preset.boundary.nodeId, element: preset.boundary.element } : undefined), [preset]);
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
      setFilm(views[0].grid ? [{ step: 0, grid: views[0].grid }] : []);
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
    if (i === 0 && f.grid) {
      const g = f.grid;
      setFilm((fs) => {
        const next = FILM_STEPS.find((s) => s > (fs.at(-1)?.step ?? -1));
        return next !== undefined && f.step >= next ? [...fs, { step: f.step, grid: { res: g.res, data: g.data.slice() } }] : fs;
      });
    }
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
      // the boundary up to 30 and the test loss up to 10 times a second; the
      // worker spaces them out further when they are expensive
      s.run({
        watch: { stats: allIds, tensors: [], grid, gridEveryMs: 33, valEveryMs: 100 },
        stepsPerSecond: speed,
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

  const current = useRef(sessions);
  current.current = sessions;
  const pause = useCallback(() => {
    setRunning(false);
    sessions.forEach((s, i) => {
      void s.pause().then(async () => {
        // the boundary and test loss are refreshed on a time budget, so they
        // can trail the last step a little: bring them up to date
        const [g, v] = await Promise.all([grid ? s.evalGrid(grid) : null, s.evalLoss('val')]);
        if (current.current[i] !== s) return;   // reset meanwhile
        setRuns((rs) => rs.map((r, k) => (k !== i ? r : {
          ...r,
          grid: g && g.ok ? g.value : r.grid,
          history: v.ok && r.history.val.length ? { ...r.history, val: [...r.history.val.slice(0, -1), v.value] } : r.history,
        })));
      });
    });
  }, [sessions, grid]);

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

  // the neuron diagram: re-read twice a second while training, and after a
  // Step or Reset. One read at a time; a slow engine just updates less often
  useEffect(() => {
    const s = sessions[0];
    if (!s || netView !== 'neurons') return;
    let dead = false, busy = false;
    const pull = async () => {
      if (busy) return;
      busy = true;
      const st = await fetchNeurons(s, preset, DOMAIN);
      busy = false;
      if (!dead && st) setNeurons(st);
    };
    void pull();
    const timer = running ? setInterval(pull, 500) : undefined;
    return () => {
      dead = true;
      if (timer) clearInterval(timer);
    };
  }, [sessions, netView, running, preset, running ? -1 : lastStep]); // eslint-disable-line react-hooks/exhaustive-deps

  const highlight: Highlight = useMemo(() => ({ valueDone: new Set(allIds), gradDone: new Set(allIds), direction: 'idle' }), [allIds]);
  const shapes = useMemo(() => Object.fromEntries((sessions[0]?.info.nodes ?? []).map((n) => [n.id, n.shape])), [sessions]);
  const labels = useMemo(() => (data ? Array.from({ length: data.n }, (_, i) => (data.y[i * data.yDim + 1] > 0.5 ? 1 : 0)) : []), [data]);

  const nodeStats = useMemo(() => ({ values: runs[0]?.values ?? {}, grads: runs[0]?.grads ?? {} }), [runs]);
  const graphStats = useThrottled(nodeStats, running ? 100 : 0);

  if (!runs.length) return <div className="loading" role="status" aria-label="Building the network"><span /><span /><span /></div>;
  const a = runs[0];
  const acc = a.grid && data ? accuracy(a.grid, DOMAIN, data) : null;
  const nowNorms = preset.layers.map((l) => ({ label: l.label, norm: a.grads[l.weight]?.l2 ?? 0 }));
  const relu = preset.spec.nodes.some((n) => n.op === 'relu');

  const lossSeries: Series[] = runs.flatMap((r, i) => [
    { label: `${r.optim.kind} train`, color: PALETTE[i], x: r.history.step, y: r.history.loss },
    { label: `${r.optim.kind} validation`, color: PALETTE[i], x: r.history.step, y: r.history.val, dash: true },
  ]);
  const normSeries: Series[] = preset.layers.map((l, i) => ({
    label: l.label, color: layerColor(i, preset.layers.length), x: a.history.step, y: smooth(a.history.norms[l.label] ?? []),
  }));

  const P = PANEL(preset.config.batchSize);
  const task = TRAIN_TASK[preset.id];
  const weightIds = preset.layers.map((l) => l.weight);
  // every photo has its place from the start: the strip fills in as training
  // reaches each step, without pushing the page down when it appears
  const filmstrip = film.length > 0 && (
    <div className="film">
      {FILM_STEPS.map((step, k) => {
        const f = film[k];   // photos are taken in order, one per slot
        return (
          <figure key={step}>
            {f ? <Canvas height={58} draw={(ctx, w, h) => drawField(ctx, w, h, f.grid.data.map((p) => 2 * p - 1), f.grid.res, divergingScale(1))} />
              : <div className="film-empty" />}
            <figcaption>step {f ? f.step : step}</figcaption>
          </figure>
        );
      })}
    </div>
  );
  return (
    <div className="train-view">
      <header className="lab-head">
        <div>
          <span className="eyebrow">Train · {preset.title}</span>
          <h1>{task?.headline ?? preset.title}</h1>
          {task && <p className="lede">{task.task}</p>}
          {task && <p className="watch-for"><span><b>Watch for: </b>{task.lesson}</span></p>}
          {preset.variants && (
            <div className="variants">
              <span>{preset.variants.label}</span>
              <span className="segmented">
                {preset.variants.options.map((v) => (
                  <a key={v.id} href={`#preset=${v.id}`} className={v.id === preset.id ? 'on' : ''} aria-current={v.id === preset.id ? 'page' : undefined}>{v.label}</a>
                ))}
              </span>
            </div>
          )}
        </div>
        <div>
          <span className="eyebrow">How training works</span>
          <ol className="loop">
            {TRAINING_LOOP(preset.config.batchSize).map((s) => <li key={s.name}><b>{s.name}</b><span>{s.line}</span></li>)}
          </ol>
        </div>
      </header>

      <div className="controls toolbar">
        <button className="primary" onClick={running ? pause : start} disabled={!running && !sessions.length}>{running ? 'Pause' : 'Run'}</button>
        <button onClick={stepOnce} disabled={running || !sessions.length} title="Exactly one training step">Step</button>
        <button onClick={reset} title="Start over from the same random weights">Reset</button>
        <SpeedControl value={speed} onChange={setSpeed} />
        <span className="counter step-count">step <b>{a.step}</b></span>
        <span className="sep" />
        <div className="optim-row">
          {optims.map((o, i) => (
            <OptimControls key={i} color={PALETTE[i]} optim={o} onChange={(n) => setOptim(i, n)}
              step={runs[i]?.step ?? 0} loss={runs[i]?.history.loss.at(-1)} fault={runs[i]?.fault} />
          ))}
        </div>
        <span className="grow" />
        <label className="field"><input type="checkbox" checked={comparing} onChange={toggleCompare} /> compare two optimizers</label>
      </div>
      <Dock>
        <button className="primary" onClick={running ? pause : start} disabled={!running && !sessions.length}>{running ? 'Pause' : 'Run'}</button>
        <button onClick={stepOnce} disabled={running || !sessions.length}>Step</button>
        <button onClick={reset}>Reset</button>
        <span className="counter">step <b>{a.step}</b></span>
      </Dock>
      <p className="controls-foot">
        <span><b>Run</b> trains until paused · <b>Step</b> makes one nudge · <b>speed</b>: training steps per second</span>
        <span>{OPTIMIZER_KEY}</span>
      </p>

      <div className="train-grid">
        <Panel n="①" className="span-7" title={netView === 'neurons' ? P.network.title : P.ops.title}
          sub={netView === 'neurons' ? P.network.explain : P.ops.explain}
          aside={
            <span className="segmented small-seg" aria-label="Network view">
              <button className={netView === 'neurons' ? 'on' : ''} aria-pressed={netView === 'neurons'} onClick={() => setNetView('neurons')}>Neurons</button>
              <button className={netView === 'ops' ? 'on' : ''} aria-pressed={netView === 'ops'} onClick={() => setNetView('ops')}>Calculations</button>
            </span>
          }>
          {netView === 'neurons' ? (
            <NeuronDiagram preset={preset} state={neurons} />
          ) : (
            <div className="graph-wrap small">
              <GraphCanvas spec={preset.spec} shapes={shapes} highlight={highlight} values={graphStats.values} grads={graphStats.grads}
                selected={selected} onSelect={setSelected} edgeWeights faultNode={a.fault?.nodeId} />
            </div>
          )}
        </Panel>

        {grid && (
          <Panel n="②" className="span-5 boundary-panel" title={P.boundary.title} sub={P.boundary.explain}>
            <div className="boundaries">
              {runs.map((r, i) => (
                <div key={i} className="boundary">
                  {comparing && <div className="muted" style={{ color: PALETTE[i] }}>{r.optim.kind}</div>}
                  <div className="square">
                    <Canvas height={280} draw={(ctx, w, h) => {
                      if (r.grid) drawField(ctx, w, h, r.grid.data.map((p) => 2 * p - 1), r.grid.res, divergingScale(1));
                      if (data) drawPoints(ctx, w, h, data.x, labels, DOMAIN);
                    }} />
                  </div>
                </div>
              ))}
            </div>
            <div className="key">
              <span><i style={{ background: '#e66101' }} />says orange</span>
              <span><i style={{ background: '#2166ac' }} />says blue</span>
              <span><i style={{ background: '#f7f7f7', border: '1px solid #ccc' }} />unsure</span>
              <span><i className="dot" />a real dot</span>
            </div>
            {filmstrip}
            <Story>{boundaryStory(acc, a.step, a.grid ? Math.max(...a.grid.data) - Math.min(...a.grid.data) : 1)}</Story>
          </Panel>
        )}

        <Panel n="③" className="span-4" title={P.loss.title} sub={P.loss.explain}>
          <Canvas height={170} draw={(ctx, w, h) => drawLines(ctx, w, h, lossSeries, {
            logY: true, yLabel: 'loss', refs: [{ y: Math.LN2, label: 'coin flip 0.69' }],
          })} />
          <div className="key">
            {runs.map((r, i) => (
              <span key={i}>
                <i className="line" style={{ background: PALETTE[i] }} />{comparing ? `${r.optim.kind} ` : ''}training dots
                <i className="dash" style={{ marginLeft: 12, color: PALETTE[i] }} />test dots
              </span>
            ))}
          </div>
          <Story>{lossStory(a.history.loss, a.history.val, a.step, a.history.step, preset.config.data.noise <= 0.1 && preset.config.data.n >= 400)}</Story>
        </Panel>

        <Panel n="④" className="span-4" title={<>{P.grads.title}{comparing && <span className="muted"> · {runs[0].optim.kind}</span>}</>} sub={P.grads.explain}>
          <LayerBars layers={nowNorms} />
          <div className="chart">
            <Canvas height={120} draw={(ctx, w, h) => drawLines(ctx, w, h, normSeries, { logY: true, yLabel: 'gradient size' })} />
          </div>
          <Story>{a.step === 0 ? 'Not trained yet: press Run or Step to compute the first corrections.' : gradStory(nowNorms, preset.layers.map((l) => a.history.norms[l.label] ?? []), trend(a.history.loss) === 'falling')}</Story>
        </Panel>

        <Panel n="⑤" className="span-4" title={P.acts.title} sub={P.acts.explain}>
          <div className="hists">
            {preset.layers.slice(0, -1).map((l) => {
              // before the first step nothing has been computed: the outputs read as zero, not as dead
              const s = a.step > 0 ? a.values[l.act] : undefined;
              return (
                <div key={l.label} className="hist">
                  <div className="muted">{l.label}{s ? ` · ${Math.round(s.fracZero * 100)}% zero` : ''}</div>
                  {s ? <Canvas height={84} draw={(ctx, w, h) => drawHistogram(ctx, w, h, s, s.fracZero > 0.9 ? '#c0392b' : '#7a5bb5')} /> : <div style={{ height: 84 }} />}
                </div>
              );
            })}
          </div>
          <Story>{a.step === 0 ? 'Not trained yet: press Run or Step, and each layer’s outputs over the dots appear here.'
            : activationStory(preset.layers.slice(0, -1).map((l) => ({ label: l.label, fracZero: a.values[l.act]?.fracZero ?? 0, std: a.values[l.act]?.std ?? 1 })), relu)}</Story>
        </Panel>

        <Panel n="⑥" className="span-12" title={P.inspect.title} sub={P.inspect.explain}
          aside={
            <select value={selected && weightIds.includes(selected) ? selected : ''} onChange={(e) => setSelected(e.target.value || undefined)} aria-label="Layer to inspect">
              <option value="">choose a layer…</option>
              {preset.layers.map((l) => <option key={l.weight} value={l.weight}>{l.label} ({l.weight})</option>)}
            </select>
          }>
          {selected && inspect ? (
            <div className="inspect-grid">
              {([['weights', '', inspect.value], ['their corrections (gradients)', '', inspect.grad],
                ['the optimizer’s average correction', 'm', inspect.m], ['the optimizer’s average squared correction', 'v', inspect.v]] as const)
                .filter(([, , d]) => d && maxAbs(d) > 0)
                .map(([label, name, d]) => (
                  <div key={label}>
                    <div className="muted">{name && <><code>{name}</code> </>}{label} · largest {fmt(maxAbs(d!))}</div>
                    <Canvas height={Math.min(140, Math.max(24, (inspect.shape.length >= 2 ? inspect.shape[0] : 1) * 14))}
                      draw={(ctx, w, h) => drawTensor(ctx, w, h, d!, inspect.shape, divergingScale(maxAbs(d!)))} />
                  </div>
                ))}
            </div>
          ) : <p className="hint">Choose a layer to open it up.</p>}
        </Panel>
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

/** One optimizer: its rule, step size and cap, and one line saying what the rule does. */
export function OptimControls(props: {
  optim: OptimConfig; color: string; onChange: (o: OptimConfig) => void; step: number; loss?: number; fault?: EngineError;
}) {
  const o = props.optim;
  return (
    <div className="optim" style={{ ['--c' as string]: props.color }}>
      <div className="optim-top">
        <select value={o.kind} onChange={(e) => props.onChange({ ...o, kind: e.target.value as OptimConfig['kind'] })} aria-label="Optimizer">
          {OPTIMIZERS.map((k) => <option key={k} value={k}>{k}</option>)}
        </select>
        <label className="field" title="learning rate: the size of each nudge">lr
          <input type="number" step="any" min={0.0001} max={10} value={o.lr}
            onChange={(e) => { const lr = Number(e.target.value); if (lr > 0 && lr <= 10) props.onChange({ ...o, lr }); }} />
        </label>
        <label className="field" title="the largest nudge allowed; 0 = no limit">clip
          <input type="number" step="any" min={0} value={o.clipNorm ?? 0}
            onChange={(e) => { const c = Number(e.target.value); if (c >= 0) props.onChange({ ...o, clipNorm: c }); }} />
        </label>
        {props.loss !== undefined && <span className="optim-loss">loss {fmt(props.loss)}</span>}
        {props.fault && <span className="bad">stopped: nan</span>}
      </div>
      <p className="optim-help">{OPTIM_HELP[o.kind]}</p>
    </div>
  );
}

/** Gradient size per layer right now, as bars on a log scale (first layer on top). */
function LayerBars({ layers }: { layers: { label: string; norm: number }[] }) {
  const logs = layers.map((l) => (l.norm > 0 ? Math.log10(l.norm) : -Infinity));
  const hi = Math.max(...logs.filter(Number.isFinite), -6);
  const lo = Math.min(hi - 1, ...logs.filter(Number.isFinite));
  return (
    <div className="layer-bars">
      {layers.map((l, i) => {
        const w = Number.isFinite(logs[i]) ? 6 + 94 * (logs[i] - lo) / Math.max(1e-9, hi - lo) : 0;
        return (
          <div key={l.label} className="layer-bar">
            <span className="muted">{l.label}</span>
            <span className="bar-track"><span className="bar" style={{ width: `${w}%`, background: layerColor(i, layers.length) }} /></span>
            <span className="counter">{l.norm > 0 ? fmt(l.norm) : '0'}</span>
          </div>
        );
      })}
    </div>
  );
}
