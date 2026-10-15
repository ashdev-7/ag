// Step through one pass, one micro-step at a time: forward values, the seed,
// every gradient contribution along every edge, parameter updates, or the
// tangents of forward mode. Any step can be revisited: seeking replays the
// pass from the start, so going back is exact.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { EngineClient, Session } from '../engine/client.ts';
import type { Cursor, EngineError, OptimConfig, PassKind, ScalarView, Stats, Step, TangentSeed } from '../engine/protocol.ts';
import { GRAPH_KEY, MODES_HELP, OPTIM_HELP, STEP_GUIDES } from '../content/help.tsx';
import { MODE_LONG, MODE_SHORT, STEP_SCREEN, ZOOM_SHORT } from '../content/step-text.tsx';
import type { Preset } from '../graph/presets.ts';
import type { GraphSpec, NodeSpec } from '../graph/spec.ts';
import { divergingScale, drawTensor, fmt, maxAbs } from '../views/draw.ts';
import { backwardQueue, gradientStops, highlightAt, localDerivative, localDerivativeValue } from '../views/graph.ts';
import type { QueueEntry, UrlState } from '../views/graph.ts';
import { Canvas } from './Canvas.tsx';
import { describe } from './ErrorBanner.tsx';
import type { Banner } from './ErrorBanner.tsx';
import { GraphCanvas } from './GraphCanvas.tsx';

const PASSES: PassKind[] = ['forward', 'reverse', 'train', 'forwardMode'];

const leaves = (spec: GraphSpec) => spec.nodes.filter((n) => n.op === 'param' || (n.op === 'input' && n.requiresGrad));

interface Detail { value?: Float32Array; grad?: Float32Array; shape?: number[] }

interface ModeRow { label: string; reverse: number; forward: number }
interface ModeResult { rows: ModeRow[]; passes: number; worst: number; truncated: boolean }

export function StepView(props: {
  client: EngineClient;
  preset: Preset;
  seed: number;
  url: UrlState;
  onUrl: (u: Partial<UrlState>) => void;
  onBanner: (b?: Banner) => void;
}) {
  const { client, preset, seed, onBanner } = props;
  const [tensor, setTensor] = useState<Session | null>(null);
  const [scalar, setScalar] = useState<{ session: Session; view: ScalarView } | null>(null);
  const [zoom, setZoom] = useState<'tensor' | 'scalar'>(props.url.zoom ?? 'tensor');
  const [pass, setPass] = useState<PassKind>(PASSES.find((p) => p === props.url.pass) ?? 'reverse');
  const [cursor, setCursor] = useState<Cursor>({ index: 0, length: 0 });
  const [values, setValues] = useState<Record<string, Stats>>({});
  const [grads, setGrads] = useState<Record<string, Stats>>({});
  const [selected, setSelected] = useState<string | undefined>();
  const [detail, setDetail] = useState<Detail>({});
  const [playing, setPlaying] = useState(false);
  const [delay, setDelay] = useState(450);
  const [tangentSeed, setTangentSeed] = useState<TangentSeed | null>(null);
  const [modes, setModes] = useState<ModeResult | 'running' | null>(null);
  const [trainLosses, setTrainLosses] = useState<number[]>([]);
  const firstSeek = useRef(props.url.i ?? 0);
  const generation = useRef(0);

  // the tensor session lives as long as the preset and seed
  useEffect(() => {
    let dead = false;
    let made: Session | null = null;
    client.createSession(preset.spec, { ...preset.config, seed }).then((r) => {
      if (dead) return void (r.ok && r.value.destroy());
      if (!r.ok) return onBanner({ kind: 'error', text: describe(r.error), error: r.error });
      made = r.value;
      setTensor(r.value);
    });
    return () => {
      dead = true;
      made?.destroy();
      setTensor(null);
    };
  }, [client, preset, seed, onBanner]);

  // the scalar zoom is a second session, made from the tensor session's current values
  useEffect(() => {
    if (zoom !== 'scalar' || !tensor) return;
    let dead = false;
    let made: Session | null = null;
    tensor.scalarize().then((r) => {
      if (dead) return void (r.ok && r.value.session.destroy());
      if (!r.ok) {
        setZoom('tensor');
        return onBanner({ kind: 'info', text: `The single-number view is not available here: ${r.error.message}` });
      }
      made = r.value.session;
      setScalar(r.value);
    });
    return () => {
      dead = true;
      made?.destroy();
      setScalar(null);
    };
  }, [zoom, tensor, onBanner]);

  const active = zoom === 'scalar' ? scalar?.session ?? null : tensor;
  const spec: GraphSpec | null = zoom === 'scalar' ? scalar?.view.spec ?? null : tensor ? preset.spec : null;
  const schedule: Step[] = active?.info.schedules[pass] ?? [];
  const shapes = useMemo(() => Object.fromEntries((active?.info.nodes ?? []).map((n) => [n.id, n.shape])), [active]);
  const byId = useMemo(() => new Map((spec?.nodes ?? []).map((n) => [n.id, n])), [spec]);
  const stops = useMemo(() => (spec && active
    ? gradientStops(spec, Object.fromEntries(active.info.nodes.map((n) => [n.id, n.requiresGrad])))
    : undefined), [spec, active]);
  const gradWhich = pass === 'forwardMode' ? 'tangent' : 'grad';

  // default forward-mode direction: the first element of the first leaf
  useEffect(() => {
    if (!spec) return;
    const first = leaves(spec)[0];
    setTangentSeed(first ? { nodeId: first.id, element: 0 } : null);
  }, [spec]);

  const refresh = useCallback(async (s: Session, c: Cursor, gen: number) => {
    const ids = s.info.nodes.map((n) => n.id);
    const [v, g] = await Promise.all([s.stats(ids, 'value'), s.stats(ids, gradWhich)]);
    if (gen !== generation.current) return;
    setCursor(c);
    if (v.ok) setValues(v.value);
    if (g.ok) setGrads(g.value);
    if (c.fault) onBanner({ kind: 'error', text: describe(c.fault), error: c.fault });
  }, [gradWhich, onBanner]);

  // the direction matters only in forward mode; elsewhere a new default
  // direction must not restart the pass (it used to, and on a fresh load that
  // could land after a linked step was reached and start over at 0)
  const direction = pass === 'forwardMode' ? tangentSeed : null;

  // (re)start the pass whenever the session, pass or direction changes
  useEffect(() => {
    if (!active) return;
    // after a zoom change the old seed names a node of the other graph; wait
    // for the default seed of this graph instead of starting a doomed pass
    if (pass === 'forwardMode' && (!tangentSeed || !active.info.nodes.some((n) => n.id === tangentSeed.nodeId))) return;
    const gen = ++generation.current;
    setPlaying(false);
    (async () => {
      const b = await active.begin(pass, pass === 'forwardMode' ? tangentSeed! : undefined);
      if (!b.ok) return onBanner({ kind: 'error', text: describe(b.error), error: b.error });
      let c = b.value;
      if (firstSeek.current > 0) {
        const s = await active.seek(Math.min(firstSeek.current, c.length));
        if (s.ok) c = s.value;
      }
      // a restart may have superseded this run; the linked step then stays for
      // that run, and counts as reached only once a run has shown it
      if (gen !== generation.current) return;
      await refresh(active, c, gen);
      if (gen === generation.current) firstSeek.current = 0;
    })();
  }, [active, pass, direction, refresh, onBanner]); // eslint-disable-line react-hooks/exhaustive-deps

  const go = useCallback(async (index: number) => {
    if (!active) return;
    const gen = ++generation.current;
    const target = Math.max(0, Math.min(index, cursor.length));
    const r = target === cursor.index + 1 ? await active.advance(1) : await active.seek(target);
    if (!r.ok) return onBanner({ kind: 'error', text: describe(r.error), error: r.error });
    await refresh(active, r.value, gen);
  }, [active, cursor, refresh, onBanner]);

  // a link's step stays in the address until the page has got there; writing
  // the starting step 0 first would lose it on a quick reload
  useEffect(() => {
    if (firstSeek.current === 0) props.onUrl({ pass, i: cursor.index, zoom });
  }, [pass, cursor.index, zoom]); // eslint-disable-line react-hooks/exhaustive-deps

  // play: one step per tick until the end
  useEffect(() => {
    if (!playing) return;
    if (cursor.index >= cursor.length) return setPlaying(false);
    const t = setTimeout(() => go(cursor.index + 1), delay);
    return () => clearTimeout(t);
  }, [playing, cursor, delay, go]);

  // the selected node's full tensors
  useEffect(() => {
    if (!active || !selected || !shapes[selected]) return setDetail({});
    let dead = false;
    Promise.all([active.read(selected, 'value'), active.read(selected, gradWhich)]).then(([v, g]) => {
      if (dead) return;
      setDetail({ shape: shapes[selected], value: v.ok ? v.value.data : undefined, grad: g.ok ? g.value.data : undefined });
    });
    return () => {
      dead = true;
    };
  }, [active, selected, cursor, shapes, gradWhich]);

  // keyboard: arrows step, space plays, home and end jump. keys typed into a
  // field, or pressed on a focused button, keep their usual meaning
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.altKey || e.ctrlKey || e.metaKey) return;
      const t = e.target as HTMLElement | null;
      if (t && (t.isContentEditable || ['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON'].includes(t.tagName))) return;
      const target = { ArrowRight: cursor.index + 1, ArrowLeft: cursor.index - 1, Home: 0, End: cursor.length }[e.key];
      if (target === undefined && e.key !== ' ') return;
      e.preventDefault();
      if (target === undefined) return setPlaying((p) => !p && cursor.index < cursor.length);
      setPlaying(false);
      go(target);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [go, cursor]);

  const highlight = useMemo(
    () => (spec ? highlightAt(schedule, cursor.index, spec, tangentSeed?.nodeId) : null),
    [spec, schedule, cursor.index, tangentSeed],
  );

  // keep this train step's update and begin the next one on a fresh batch
  const nextTrainStep = async () => {
    if (!active || !spec) return;
    const loss = values[spec.loss]?.mean;
    const gen = ++generation.current;
    setPlaying(false);
    const b = await active.begin('train');
    if (!b.ok) return onBanner({ kind: 'error', text: describe(b.error), error: b.error });
    if (loss !== undefined) setTrainLosses((l) => [...l, loss]);
    await refresh(active, b.value, gen);
  };

  // reverse mode once against forward mode once per leaf element
  const compareModes = async () => {
    if (!active || !spec) return;
    setPlaying(false);
    setModes('running');
    const gen = ++generation.current;
    const loss = spec.loss;
    const rows: ModeRow[] = [];
    const fail = (e: EngineError) => {
      setModes(null);
      onBanner({ kind: 'error', text: describe(e), error: e });
    };
    let b = await active.begin('reverse');
    if (!b.ok) return fail(b.error);
    await active.advance(b.value.length);
    const reverse: Record<string, Float32Array> = {};
    for (const leaf of leaves(spec)) {
      const r = await active.read(leaf.id, 'grad');
      if (r.ok) reverse[leaf.id] = r.value.data;
    }
    let passes = 0, truncated = false;
    outer: for (const leaf of leaves(spec)) {
      const n = (shapes[leaf.id] ?? []).reduce((a, x) => a * x, 1);
      for (let e = 0; e < n; ++e) {
        if (passes >= 200) { truncated = true; break outer; }
        b = await active.begin('forwardMode', { nodeId: leaf.id, element: e });
        if (!b.ok) return fail(b.error);
        await active.advance(b.value.length);
        const t = await active.read(loss, 'tangent');
        passes++;
        rows.push({ label: n > 1 ? `${leaf.name ?? leaf.id}[${e}]` : leaf.name ?? leaf.id, reverse: reverse[leaf.id]?.[e] ?? NaN, forward: t.ok ? t.value.data[0] : NaN });
      }
    }
    const worst = rows.reduce((m, r) => Math.max(m, Math.abs(r.reverse - r.forward) / Math.max(1, Math.abs(r.reverse))), 0);
    // put the pass back where it was
    const back = await active.begin(pass, pass === 'forwardMode' ? tangentSeed ?? undefined : undefined);
    if (back.ok) {
      const s = await active.seek(cursor.index);
      await refresh(active, s.ok ? s.value : back.value, gen);
    }
    setModes({ rows, passes, worst, truncated });
  };

  if (!spec || !active || !highlight) return <div className="loading">Building the graph…</div>;

  const step = highlight.step;
  const selNode = selected ? byId.get(selected) : undefined;
  const guide = STEP_GUIDES[preset.id];

  return (
    <div className="step-view">
      <header className="lab-head">
        <div>
          <span className="eyebrow">Step through</span>
          <h1>{preset.title}</h1>
          {guide && <p className="lede">{guide.idea}</p>}
          <p className="hint">{STEP_SCREEN}</p>
        </div>
        {guide && (
          <div className="try">
            <span className="eyebrow">Try this</span>
            <ol>{guide.watch.map((w, i) => <li key={i}>{w}</li>)}</ol>
          </div>
        )}
      </header>

      <div className="modes-pick" role="group" aria-label="Pass">
        {PASSES.map((p) => (
          <button key={p} className={`mode-card${pass === p ? ' on' : ''}`} aria-pressed={pass === p} aria-label={MODE_SHORT[p].name}
            onClick={() => setPass(p)} disabled={active.info.schedules[p].length === 0}>
            <b>{MODE_SHORT[p].name}</b>
            <span>{MODE_SHORT[p].line}</span>
          </button>
        ))}
      </div>
      <p className="mode-long"><b>{MODE_SHORT[pass].name}:</b> {MODE_LONG(preset.config.optim.lr)[pass]}</p>

      <div className="step-bar toolbar">
        <div className="segmented" aria-label="Zoom">
          <button className={zoom === 'tensor' ? 'on' : ''} aria-pressed={zoom === 'tensor'} onClick={() => setZoom('tensor')} title={ZOOM_SHORT.tensor.line}>Tensors</button>
          <button className={zoom === 'scalar' ? 'on' : ''} aria-pressed={zoom === 'scalar'} onClick={() => setZoom('scalar')} title={ZOOM_SHORT.scalar.line}>Single numbers</button>
        </div>
        <span className="hint">{ZOOM_SHORT[zoom].line}</span>
        {pass === 'forwardMode' && spec && (
          <label className="field">direction
            <select value={tangentSeed ? `${tangentSeed.nodeId}:${tangentSeed.element}` : ''}
              onChange={(e) => { const [id, el] = e.target.value.split(':'); setTangentSeed({ nodeId: id, element: Number(el) }); }}>
              {leaves(spec).flatMap((l) => {
                const n = (shapes[l.id] ?? []).reduce((a, x) => a * x, 1);
                return Array.from({ length: Math.min(n, 64) }, (_, e) => (
                  <option key={`${l.id}:${e}`} value={`${l.id}:${e}`}>{l.name ?? l.id}{n > 1 ? `[${e}]` : ''}</option>
                ));
              })}
            </select>
          </label>
        )}
        <button className="ghost" onClick={compareModes} disabled={modes === 'running'} title="Compute the gradients both ways and check they agree">Reverse vs forward mode</button>
      </div>
      <div className="graph-key" aria-label="Key">
        {GRAPH_KEY.map((k, i) => <span key={i}><i className={k.swatch} />{k.label}</span>)}
      </div>

      <div className="step-main">
        <div className="graph-wrap">
          <GraphCanvas spec={spec} shapes={shapes} highlight={highlight} values={values} grads={grads}
            gradLabel={pass === 'forwardMode' ? 'ṫ' : '∂'} selected={selected} onSelect={setSelected}
            faultNode={cursor.fault?.nodeId} stops={stops} />
        </div>
        <aside className="inspector">
          <StepExplain step={step} spec={spec} byId={byId} values={values} grads={grads} shapes={shapes} schedule={schedule}
            index={cursor.index} pass={pass} seed={tangentSeed} optim={preset.config.optim} />
          {pass === 'train' && (cursor.index >= cursor.length || trainLosses.length > 0) && (
            <div className="train-steps">
              {cursor.index >= cursor.length && <button className="primary" onClick={nextTrainStep}>Next train step</button>}
              {trainLosses.length > 0 && <p className="muted">Loss of each finished train step: {trainLosses.map(fmt).join(' → ')}</p>}
            </div>
          )}
          {selNode ? (
            <NodeDetail node={selNode} detail={detail} gradLabel={pass === 'forwardMode' ? 'tangent' : 'gradient'}
              done={highlight.valueDone.has(selNode.id)} gradDone={highlight.gradDone.has(selNode.id)} />
          ) : (
            <p className="muted">Click a node to see its numbers.</p>
          )}
          {(pass === 'reverse' || pass === 'train') && (
            <BackwardOrder queue={backwardQueue(schedule, cursor.index, spec)} byId={byId} onSelect={setSelected} />
          )}
          {modes && <ModesPanel modes={modes} onClose={() => setModes(null)} />}
        </aside>
      </div>

      <div className="transport">
        <button onClick={() => go(0)} title="Back to the start (Home)" aria-label="Back to the start">⏮</button>
        <button onClick={() => go(cursor.index - 1)} disabled={cursor.index === 0} title="One step back (←)" aria-label="One step back">◀</button>
        <button className="primary" onClick={() => setPlaying(!playing)} disabled={cursor.index >= cursor.length && !playing} title="Play or pause (space)">
          {playing ? 'Pause' : 'Play'}
        </button>
        <button onClick={() => go(cursor.index + 1)} disabled={cursor.index >= cursor.length} title="One step forward (→)" aria-label="One step forward">▶</button>
        <button onClick={() => go(cursor.length)} title="To the end (End)" aria-label="To the end">⏭</button>
        <input type="range" min={0} max={cursor.length} value={cursor.index} onChange={(e) => go(Number(e.target.value))} aria-label="Step" />
        <span className="counter" aria-live="polite">{cursor.index} / {cursor.length}</span>
        <label className="field">speed
          <input type="range" min={60} max={1200} step={30} value={1260 - delay} onChange={(e) => setDelay(1260 - Number(e.target.value))} />
        </label>
      </div>
    </div>
  );
}

// ── the step that just ran ──────────────────────────────────────────────────

function StepExplain(props: {
  step?: Step; spec: GraphSpec; byId: Map<string, NodeSpec>; values: Record<string, Stats>; grads: Record<string, Stats>;
  shapes: Record<string, number[]>; schedule: Step[]; index: number; pass: PassKind; seed: TangentSeed | null;
  optim: OptimConfig;
}) {
  const { step, byId, values, grads, shapes } = props;
  const name = (id: string) => byId.get(id)?.name ?? id;
  // a box holds a single number: show the arithmetic with real numbers
  const single = (id: string) => (shapes[id] ?? []).reduce((a, b) => a * b, 1) === 1;
  const v = (id: string) => values[id]?.mean ?? NaN;
  const g = (id: string) => grads[id]?.mean ?? NaN;
  if (!step)
    return (
      <div className="explain">
        <h3>Nothing computed yet</h3>
        <p>Press ▶ (or the → key) for one step, or Play to run them all. The box that just ran is outlined, and this panel says what it did.</p>
      </div>
    );
  const n = byId.get(step.nodeId);

  switch (step.kind) {
    case 'forward': {
      if (!n || n.inputs.length === 0)
        return (
          <div className="explain forward">
            <h3>Forward · {name(step.nodeId)}</h3>
            <p>{n?.op === 'param' ? <>A <b>parameter</b>: a number the network learns. Its current value is read in.</>
              : n?.op === 'const' ? <>A <b>constant</b>: a fixed number that is part of the formula.</>
              : <>A <b>fixed input</b>: its value is given, not learned{n?.role ? ' (here, the current batch of data)' : ''}.</>}</p>
            {single(step.nodeId) && <p className="big">{name(step.nodeId)} = {fmt(v(step.nodeId))}</p>}
          </div>
        );
      const allSingle = single(step.nodeId) && n.inputs.every(single);
      return (
        <div className="explain forward">
          <h3>Forward · {name(step.nodeId)}</h3>
          <p>Computes <code>{name(step.nodeId)} = {formula(n, n.inputs.map(name))}</code> from its inputs, which are already known.</p>
          {allSingle && <p className="big">{formula(n, n.inputs.map((i) => fmt(v(i))))} = {fmt(v(step.nodeId))}</p>}
        </div>
      );
    }
    case 'seedGrad':
      return (
        <div className="explain backward">
          <h3>Backward starts at the loss</h3>
          <p>The gradient of the loss with respect to itself is 1: <code>∂L/∂L = 1</code>. Every other gradient is built from this one, going back along the arrows.</p>
        </div>
      );
    case 'vjp': {
      const target = n?.inputs[step.slot] ?? '';
      const uses = (k: Step) => k.kind === 'vjp' && byId.get(k.nodeId)?.inputs[k.slot] === target;
      const total = props.schedule.filter(uses).length;
      const sofar = props.schedule.slice(0, props.index).filter(uses).length;
      const local = n ? localDerivative(n.op, step.slot, n.inputs.map(name), n.scalar) : '';
      const nums = !!n && single(step.nodeId) && single(target) && n.inputs.every(single);
      const localValue = nums ? localDerivativeValue(n!.op, step.slot, n!.inputs.map(v), v(step.nodeId), n!.scalar) : undefined;
      return (
        <div className="explain backward">
          <h3>Backward · {name(step.nodeId)} → {name(target)}</h3>
          <p><b>{name(step.nodeId)}</b> passes gradient back to its input <b>{name(target)}</b>. Chain rule: the gradient at {name(step.nodeId)} times how much {name(step.nodeId)} moves when {name(target)} moves.</p>
          <p><code>∂L/∂{name(target)} += ∂L/∂{name(step.nodeId)} × ∂{name(step.nodeId)}/∂{name(target)}</code></p>
          <p>How much {name(step.nodeId)} moves when {name(target)} moves (its local slope, <code>∂{name(step.nodeId)}/∂{name(target)}</code>): <code>{local}</code></p>
          {localValue !== undefined && (
            <p className="big">{fmt(g(step.nodeId))} × {fmt(localValue)} = {fmt(g(step.nodeId) * localValue)}</p>
          )}
          {total > 1 && <p className="muted">{name(target)} is used {total} times, so its gradient is the sum of {total} pieces; this is piece {sofar}.</p>}
          {single(target) && <p>∂L/∂{name(target)} is now <b>{fmt(g(target))}</b>{total > 1 && sofar < total ? ' so far' : ''}.</p>}
        </div>
      );
    }
    case 'update':
      return (
        <div className="explain update">
          <h3>Update · {name(step.nodeId)}</h3>
          <p>The optimizer moves <b>{name(step.nodeId)}</b> a small step against its gradient, so the loss goes down. Learning rate <code>lr = {props.optim.lr}</code>.</p>
          <p><b>{props.optim.kind}</b>: {OPTIM_HELP[props.optim.kind]}</p>
          <p className="muted">The values in the boxes are still from before the update; the next train step’s forward pass uses the new ones.</p>
        </div>
      );
    case 'seedTangent':
      return (
        <div className="explain tangent">
          <h3>Forward mode starts</h3>
          <p>The chosen input <code>{props.seed ? name(props.seed.nodeId) : ''}[{props.seed?.element}]</code> gets a nudge <code>ṫ = 1</code>; every other input gets 0. The next steps carry this nudge forward.</p>
        </div>
      );
    case 'jvp':
      return (
        <div className="explain tangent">
          <h3>Tangent · {name(step.nodeId)}</h3>
          <p>How much <b>{name(step.nodeId)}</b> moves when the chosen input is nudged: each input’s nudge times the local derivative, summed.</p>
          <p><code>ṫ{name(step.nodeId)} = Σ ∂{name(step.nodeId)}/∂input × ṫinput</code></p>
          {single(step.nodeId) && <p className="big">ṫ{name(step.nodeId)} = {fmt(g(step.nodeId))}</p>}
          {step.nodeId === props.spec.loss && <p>This is the loss, so this number is <b>∂L/∂{props.seed ? name(props.seed.nodeId) : ''}[{props.seed?.element}]</b>.</p>}
        </div>
      );
  }
}

/** The op as a formula of its inputs, e.g. "a × b". */
function formula(n: NodeSpec, ins: string[]): string {
  const [a, b] = ins;
  switch (n.op) {
    case 'add': return `${a} + ${b}`;
    case 'sub': return `${a} − ${b}`;
    case 'mul': return `${a} × ${b}`;
    case 'div': return `${a} / ${b}`;
    case 'neg': return `−${a}`;
    case 'pow': return `${a}^${n.scalar ?? 0}`;
    case 'exp': return `exp(${a})`;
    case 'log': return `log(${a})`;
    case 'matmul': return `${a} @ ${b}`;
    case 'sum': return `sum(${a})`;
    case 'mean': return `mean(${a})`;
    case 'reshape': return `reshape(${a})`;
    case 'transpose': return `${a}ᵀ`;
    case 'relu': return `max(0, ${a})`;
    case 'leaky_relu': return `leaky_relu(${a})`;
    case 'tanh': return `tanh(${a})`;
    case 'sigmoid': return `sigmoid(${a})`;
    case 'softmax': return `softmax(${a})`;
    case 'mse_loss': return `mean((${a} − ${b})²)`;
    case 'softmax_cross_entropy': return `cross_entropy(${a}, ${b})`;
    case 'detach': return `detach(${a})`;
    default: return `${n.op}(${ins.join(', ')})`;
  }
}

function NodeDetail(props: { node: NodeSpec; detail: Detail; gradLabel: string; done: boolean; gradDone: boolean }) {
  const { node, detail } = props;
  const shape = detail.shape ?? [];
  const small = shape.reduce((a, b) => a * b, 1) <= 64;
  const height = Math.max(36, Math.min(200, (shape.length >= 2 ? shape[0] : 1) * (small ? 28 : 8)));
  return (
    <div className="node-detail">
      <h3>{node.name ?? node.id} <span className="muted">{node.op} [{shape.join(', ')}]</span></h3>
      <div className="label">value</div>
      {props.done && detail.value ? (
        <Canvas height={height} draw={(ctx, w, hh) => drawTensor(ctx, w, hh, detail.value!, shape, divergingScale(maxAbs(detail.value!)))} />
      ) : <p className="muted">not computed yet</p>}
      <div className="label">{props.gradLabel}</div>
      {detail.grad && props.gradDone ? (
        <Canvas height={height} draw={(ctx, w, hh) => drawTensor(ctx, w, hh, detail.grad!, shape, divergingScale(maxAbs(detail.grad!)))} />
      ) : <p className="muted">{detail.grad ? 'nothing has arrived yet' : 'none: nothing it feeds needs one'}</p>}
    </div>
  );
}

// ── the order of the backward pass ──────────────────────────────────────────

const QUEUE_LABEL: Record<QueueEntry['state'], string> = { waiting: 'waiting', ready: 'ready', running: 'passing back', done: 'done' };
const QUEUE_LIST_MAX = 40;

function BackwardOrder(props: { queue: QueueEntry[]; byId: Map<string, NodeSpec>; onSelect: (id: string) => void }) {
  const { queue, byId } = props;
  const name = (id: string) => byId.get(id)?.name ?? id;
  const ready = queue.filter((q) => q.state === 'ready');
  return (
    <div className="queue">
      <h3>Backward order</h3>
      <p className="muted">A node passes its gradient back only once every use of it has sent its share.</p>
      <p>Ready: {ready.length ? ready.map((q) => <b key={q.nodeId}>{name(q.nodeId)} </b>) : <span className="muted">none</span>}</p>
      {queue.length <= QUEUE_LIST_MAX ? (
        <ol>
          {queue.map((q) => (
            <li key={q.nodeId} className={`q-${q.state}`}>
              <button onClick={() => props.onSelect(q.nodeId)}>
                <span>{name(q.nodeId)}</span>
                <span className="q-state">{QUEUE_LABEL[q.state]}{q.total > 1 && q.state === 'waiting' ? ` ${q.got} of ${q.total}` : ''}</span>
              </button>
            </li>
          ))}
        </ol>
      ) : (
        <p className="muted">
          {(['done', 'running', 'ready', 'waiting'] as const).map((s) => `${queue.filter((q) => q.state === s).length} ${QUEUE_LABEL[s]}`).join(' · ')}
        </p>
      )}
    </div>
  );
}

function ModesPanel({ modes, onClose }: { modes: ModeResult | 'running'; onClose: () => void }) {
  if (modes === 'running') return <div className="modes"><p>Running forward mode once per input number…</p></div>;
  return (
    <div className="modes">
      <h3>Reverse vs forward mode <button className="ghost small" onClick={onClose}>×</button></h3>
      <p className="muted">{MODES_HELP}</p>
      <p><b>Reverse mode:</b> 1 pass gave all {modes.rows.length} derivatives of the loss.</p>
      <p><b>Forward mode:</b> {modes.passes} passes{modes.truncated ? ' (stopped at 200)' : ''}, one per input number.</p>
      <p className={modes.worst < 1e-4 ? 'ok' : 'bad'}>
        {modes.worst < 1e-4 ? '✓ They agree.' : '✗ They differ.'} Largest relative difference: {modes.worst.toExponential(1)}
      </p>
      <table>
        <thead><tr><th>input</th><th>reverse</th><th>forward</th></tr></thead>
        <tbody>
          {modes.rows.slice(0, 40).map((r) => (
            <tr key={r.label}><td>{r.label}</td><td>{fmt(r.reverse)}</td><td>{fmt(r.forward)}</td></tr>
          ))}
        </tbody>
      </table>
      {modes.rows.length > 40 && <p className="muted">and {modes.rows.length - 40} more</p>}
    </div>
  );
}
