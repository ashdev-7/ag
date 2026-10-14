// Step through one pass, one micro-step at a time: forward values, the seed,
// every gradient contribution along every edge, parameter updates, or the
// tangents of forward mode. Any step can be revisited: seeking replays the
// pass from the start, so going back is exact.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { EngineClient, Session } from '../engine/client.ts';
import type { Cursor, EngineError, PassKind, ScalarView, Stats, Step, TangentSeed } from '../engine/protocol.ts';
import type { Preset } from '../graph/presets.ts';
import type { GraphSpec, NodeSpec } from '../graph/spec.ts';
import { divergingScale, drawTensor, fmt, maxAbs } from '../views/draw.ts';
import { gradientStops, highlightAt, localDerivative, localDerivativeValue } from '../views/graph.ts';
import type { UrlState } from '../views/graph.ts';
import { Canvas } from './Canvas.tsx';
import { describe } from './ErrorBanner.tsx';
import type { Banner } from './ErrorBanner.tsx';
import { GraphCanvas } from './GraphCanvas.tsx';

const PASSES: { pass: PassKind; label: string; hint: string }[] = [
  { pass: 'forward', label: 'Forward', hint: 'values only' },
  { pass: 'reverse', label: 'Backward', hint: 'forward, then gradients from the loss back' },
  { pass: 'train', label: 'Train step', hint: 'backward, then every parameter updated' },
  { pass: 'forwardMode', label: 'Forward mode', hint: 'derivatives carried forward along one direction' },
];

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
  const [pass, setPass] = useState<PassKind>((PASSES.find((p) => p.pass === props.url.pass)?.pass) ?? 'reverse');
  const [cursor, setCursor] = useState<Cursor>({ index: 0, length: 0 });
  const [values, setValues] = useState<Record<string, Stats>>({});
  const [grads, setGrads] = useState<Record<string, Stats>>({});
  const [selected, setSelected] = useState<string | undefined>();
  const [detail, setDetail] = useState<Detail>({});
  const [playing, setPlaying] = useState(false);
  const [delay, setDelay] = useState(450);
  const [tangentSeed, setTangentSeed] = useState<TangentSeed | null>(null);
  const [modes, setModes] = useState<ModeResult | 'running' | null>(null);
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
        firstSeek.current = 0;
        if (s.ok) c = s.value;
      }
      if (gen === generation.current) await refresh(active, c, gen);
    })();
  }, [active, pass, tangentSeed, refresh, onBanner]);

  const go = useCallback(async (index: number) => {
    if (!active) return;
    const gen = ++generation.current;
    const target = Math.max(0, Math.min(index, cursor.length));
    const r = target === cursor.index + 1 ? await active.advance(1) : await active.seek(target);
    if (!r.ok) return onBanner({ kind: 'error', text: describe(r.error), error: r.error });
    await refresh(active, r.value, gen);
  }, [active, cursor, refresh, onBanner]);

  useEffect(() => props.onUrl({ pass, i: cursor.index, zoom }), [pass, cursor.index, zoom]); // eslint-disable-line react-hooks/exhaustive-deps

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

  const highlight = useMemo(
    () => (spec ? highlightAt(schedule, cursor.index, spec, tangentSeed?.nodeId) : null),
    [spec, schedule, cursor.index, tangentSeed],
  );

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

  return (
    <div className="step-view">
      <div className="toolbar">
        <div className="segmented" role="tablist" aria-label="Pass">
          {PASSES.map((p) => (
            <button key={p.pass} className={pass === p.pass ? 'on' : ''} title={p.hint} onClick={() => setPass(p.pass)}
              disabled={active.info.schedules[p.pass].length === 0}>{p.label}</button>
          ))}
        </div>
        <div className="segmented" aria-label="Zoom">
          <button className={zoom === 'tensor' ? 'on' : ''} onClick={() => setZoom('tensor')}>Tensors</button>
          <button className={zoom === 'scalar' ? 'on' : ''} onClick={() => setZoom('scalar')} title="Every element as its own node">Single numbers</button>
        </div>
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
        <button className="ghost" onClick={compareModes} disabled={modes === 'running'}>Reverse vs forward mode</button>
      </div>

      <div className="step-main">
        <div className="graph-wrap">
          <GraphCanvas spec={spec} shapes={shapes} highlight={highlight} values={values} grads={grads}
            gradLabel={pass === 'forwardMode' ? 'ṫ' : '∂'} selected={selected} onSelect={setSelected}
            faultNode={cursor.fault?.nodeId} stops={stops} />
        </div>
        <aside className="inspector">
          <StepExplain step={step} spec={spec} byId={byId} values={values} grads={grads} schedule={schedule}
            index={cursor.index} pass={pass} seed={tangentSeed} scalarZoom={zoom === 'scalar'} />
          {selNode ? (
            <NodeDetail node={selNode} detail={detail} gradLabel={pass === 'forwardMode' ? 'tangent' : 'gradient'}
              done={highlight.valueDone.has(selNode.id)} gradDone={highlight.gradDone.has(selNode.id)} />
          ) : (
            <p className="muted">Click a node to see its numbers.</p>
          )}
          {modes && <ModesPanel modes={modes} onClose={() => setModes(null)} />}
        </aside>
      </div>

      <div className="transport">
        <button onClick={() => go(0)} title="Back to the start">⏮</button>
        <button onClick={() => go(cursor.index - 1)} disabled={cursor.index === 0} title="One step back">◀</button>
        <button className="primary" onClick={() => setPlaying(!playing)} disabled={cursor.index >= cursor.length && !playing}>
          {playing ? 'Pause' : 'Play'}
        </button>
        <button onClick={() => go(cursor.index + 1)} disabled={cursor.index >= cursor.length} title="One step forward">▶</button>
        <button onClick={() => go(cursor.length)} title="To the end">⏭</button>
        <input type="range" min={0} max={cursor.length} value={cursor.index} onChange={(e) => go(Number(e.target.value))} aria-label="Step" />
        <span className="counter">{cursor.index} / {cursor.length}</span>
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
  schedule: Step[]; index: number; pass: PassKind; seed: TangentSeed | null; scalarZoom: boolean;
}) {
  const { step, byId, values, grads } = props;
  const name = (id: string) => byId.get(id)?.name ?? id;
  if (!step) return <div className="explain"><h3>Ready</h3><p>Press play or step forward. Nothing has been computed yet.</p></div>;
  const n = byId.get(step.nodeId);
  const v = (id: string) => values[id]?.mean ?? NaN;
  const g = (id: string) => grads[id]?.mean ?? NaN;

  switch (step.kind) {
    case 'forward':
      return (
        <div className="explain forward">
          <h3>Forward · {name(step.nodeId)}</h3>
          {n && n.inputs.length > 0 ? (
            <p><code>{name(step.nodeId)} = {n.op}({n.inputs.map(name).join(', ')})</code></p>
          ) : (
            <p>{n?.op === 'param' ? 'A parameter: its current value is read.' : 'A leaf: its value is given.'}</p>
          )}
          {props.scalarZoom && <p className="big">{fmt(v(step.nodeId))}</p>}
        </div>
      );
    case 'seedGrad':
      return (
        <div className="explain backward">
          <h3>Seed</h3>
          <p><code>∂L/∂{name(step.nodeId)} = 1</code></p>
          <p>The backward pass starts at the loss.</p>
        </div>
      );
    case 'vjp': {
      const target = n?.inputs[step.slot] ?? '';
      const total = props.schedule.filter((s) => s.kind === 'vjp' && byId.get(s.nodeId)?.inputs[s.slot] === target).length;
      const sofar = props.schedule.slice(0, props.index).filter((s) => s.kind === 'vjp' && byId.get(s.nodeId)?.inputs[s.slot] === target).length;
      const local = n ? localDerivative(n.op, step.slot, n.inputs.map(name), n.scalar) : '';
      const localValue = n && props.scalarZoom ? localDerivativeValue(n.op, step.slot, n.inputs.map(v), v(step.nodeId), n.scalar) : undefined;
      return (
        <div className="explain backward">
          <h3>Backward · {name(step.nodeId)} → {name(target)}</h3>
          <p><code>∂L/∂{name(target)} += ∂L/∂{name(step.nodeId)} · ∂{name(step.nodeId)}/∂{name(target)}</code></p>
          <p>Local derivative: <code>{local}</code></p>
          {localValue !== undefined && (
            <p className="big">{fmt(g(step.nodeId))} × {fmt(localValue)} = {fmt(g(step.nodeId) * localValue)}</p>
          )}
          {total > 1 && <p className="muted">Contribution {sofar} of {total}: {name(target)} is used {total} times, and its gradient is the sum.</p>}
          {props.scalarZoom && <p>∂L/∂{name(target)} is now <b>{fmt(g(target))}</b></p>}
        </div>
      );
    }
    case 'update':
      return (
        <div className="explain update">
          <h3>Update · {name(step.nodeId)}</h3>
          <p>The optimizer moves {name(step.nodeId)} against its gradient.</p>
        </div>
      );
    case 'seedTangent':
      return (
        <div className="explain tangent">
          <h3>Seed direction</h3>
          <p><code>ṫ {props.seed ? name(props.seed.nodeId) : ''}[{props.seed?.element}] = 1</code>, every other input 0.</p>
          <p>Forward mode follows this one direction through the graph.</p>
        </div>
      );
    case 'jvp':
      return (
        <div className="explain tangent">
          <h3>Tangent · {name(step.nodeId)}</h3>
          <p><code>ṫ{name(step.nodeId)} = Σ ∂{name(step.nodeId)}/∂input · ṫinput</code></p>
          {props.scalarZoom && <p className="big">{fmt(g(step.nodeId))}</p>}
        </div>
      );
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

function ModesPanel({ modes, onClose }: { modes: ModeResult | 'running'; onClose: () => void }) {
  if (modes === 'running') return <div className="modes"><p>Running forward mode once per input number…</p></div>;
  return (
    <div className="modes">
      <h3>Reverse vs forward mode <button className="ghost small" onClick={onClose}>×</button></h3>
      <p><b>Reverse mode:</b> 1 pass gave all {modes.rows.length} derivatives of the loss.</p>
      <p><b>Forward mode:</b> {modes.passes} passes{modes.truncated ? ' (stopped at 200)' : ''}, one per input number.</p>
      <p className={modes.worst < 1e-4 ? 'ok' : 'bad'}>Largest relative difference: {modes.worst.toExponential(1)}</p>
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
