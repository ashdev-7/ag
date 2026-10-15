// The network as a data-flow diagram. Each node shows its op, shape, value
// and gradient; the step that just ran is outlined, and during the backward
// pass the edge carrying gradient is drawn moving against the arrow.

import { memo, useEffect, useMemo, useRef, useState } from 'react';
import { Background, Controls, Handle, Position, ReactFlow, ReactFlowProvider } from '@xyflow/react';
import type { Edge, Node, NodeProps } from '@xyflow/react';
import type { GraphSpec, NodeSpec } from '../graph/spec.ts';
import type { Stats } from '../engine/protocol.ts';
import { fmt } from '../views/draw.ts';
import { layoutGraph, OP_SYMBOL } from '../views/graph.ts';
import { PHONE, TOUCH, useMedia } from './useMedia.ts';
import type { Highlight } from '../views/graph.ts';

interface OpData extends Record<string, unknown> {
  spec: NodeSpec;
  shape?: number[];
  value?: Stats;
  grad?: Stats;
  gradLabel: string;
  done: boolean;
  gradDone: boolean;
  active?: Highlight['direction'];
  selected: boolean;
  fault: boolean;
  compact: boolean;
  vertical: boolean;
  /** The gradient stops here (detach or no_grad). */
  stop: boolean;
}

function summary(s: Stats | undefined, shape: number[] | undefined): string {
  if (!s) return '';
  const n = shape ? shape.reduce((a, b) => a * b, 1) : 1;
  if (s.nonFinite) return 'nan';
  // average ± spread: the two numbers that summarise an array
  return n === 1 ? fmt(s.mean) : `${fmt(s.mean)} ± ${fmt(s.std)}`;
}

const OpNode = memo(function OpNode({ data }: NodeProps<Node<OpData>>) {
  const { spec, shape } = data;
  const kind = spec.op === 'param' ? 'param' : spec.op === 'input' || spec.op === 'const' ? 'leaf' : 'op';
  const cls = ['gnode', `gnode-${kind}`, data.done ? '' : 'gnode-pending', data.active ? `gnode-active-${data.active}` : '',
    data.selected ? 'gnode-selected' : '', data.fault ? 'gnode-fault' : '', data.compact ? 'gnode-compact' : ''].join(' ');
  const label = spec.name ?? spec.id;
  const showOp = kind === 'op' && OP_SYMBOL[spec.op] !== label;
  return (
    <div className={cls}>
      <Handle type="target" position={data.vertical ? Position.Top : Position.Left} isConnectable={false} />
      <div className="gnode-head">
        {showOp && <span className="gnode-op">{OP_SYMBOL[spec.op] ?? spec.op}</span>}
        <span className="gnode-name">{label}</span>
      </div>
      {!data.compact && shape && shape.length > 0 && <div className="gnode-shape">[{shape.join(', ')}]</div>}
      <div className="gnode-value">{data.done ? summary(data.value, shape) : ' '}</div>
      {data.stop
        ? <div className="gnode-grad gnode-stop" title="No gradient flows back through this node">no {data.gradLabel}</div>
        : <div className="gnode-grad">{data.gradDone && data.grad ? `${data.gradLabel} ${summary(data.grad, shape)}` : ' '}</div>}
      <Handle type="source" position={data.vertical ? Position.Bottom : Position.Right} isConnectable={false} />
    </div>
  );
});

const nodeTypes = { op: OpNode };

export interface GraphCanvasProps {
  spec: GraphSpec;
  shapes: Record<string, number[]>;
  highlight: Highlight;
  values: Record<string, Stats>;
  grads: Record<string, Stats>;
  gradLabel?: string;
  selected?: string;
  faultNode?: string;
  onSelect: (id?: string) => void;
  /** Edge thickness by the gradient norm of the receiving node (training view). */
  edgeWeights?: boolean;
  /** Nodes that stop the gradient; edges into them are dashed. */
  stops?: Set<string>;
}

function Inner(props: GraphCanvasProps & { vertical: boolean; touch: boolean }) {
  const { spec, highlight } = props;
  const compact = spec.nodes.length > 40;
  const { vertical, touch } = props;
  const layout = useMemo(
    () => layoutGraph(spec, () => (compact ? { width: 92, height: 46 } : { width: 138, height: 70 }), vertical),
    [spec, compact, vertical],
  );
  const byId = useMemo(() => new Map(spec.nodes.map((n) => [n.id, n])), [spec]);
  // sizes are fixed by the layout, so React Flow never has to measure a node
  // before showing it or fitting the view. `measured` matters too: without it
  // React Flow drops the handle positions every time it receives a new node
  // array, and edges are only drawn between measured handles. Losing that
  // race left a graph with no edges on about one load in three
  const nodes: Node<OpData>[] = layout.nodes.map((n) => ({
    id: n.id,
    type: 'op',
    position: { x: n.x, y: n.y },
    width: n.width,
    height: n.height,
    measured: { width: n.width, height: n.height },
    draggable: false,
    data: {
      spec: byId.get(n.id)!,
      shape: props.shapes[n.id],
      value: props.values[n.id],
      grad: props.grads[n.id],
      gradLabel: props.gradLabel ?? '∂',
      done: highlight.valueDone.has(n.id),
      gradDone: highlight.gradDone.has(n.id),
      active: highlight.active === n.id ? highlight.direction : undefined,
      selected: props.selected === n.id,
      fault: props.faultNode === n.id,
      compact,
      vertical,
      stop: !!props.stops?.has(n.id),
    },
  }));

  let maxNorm = 0;
  if (props.edgeWeights) for (const s of Object.values(props.grads)) maxNorm = Math.max(maxNorm, s.l2);
  const edges: Edge[] = layout.edges.map((e) => {
    const active = highlight.activeEdge && highlight.activeEdge.from === e.from && highlight.activeEdge.to === e.to &&
      highlight.activeEdge.slot === e.slot;
    const w = props.edgeWeights && maxNorm > 0 ? 1 + 5 * Math.sqrt((props.grads[e.from]?.l2 ?? 0) / maxNorm) : 1.5;
    const stopped = !!props.stops?.has(e.to);
    return {
      id: e.id, source: e.from, target: e.to,
      animated: !!active,
      className: active ? 'gedge-active' : stopped ? 'gedge-stopped' : '',
      style: { strokeWidth: active ? 3 : w },
    };
  });

  // on a phone the graph is drawn at a readable zoom and sized to itself:
  // wider than the screen, it scrolls sideways in its box, and the page
  // scrolls up and down past it. A very big graph is scaled down to stay
  // within about a screen and a half
  const box = useRef<HTMLDivElement>(null);
  const [boxWidth, setBoxWidth] = useState(0);
  useEffect(() => {
    if (!vertical || !box.current) return;
    const ro = new ResizeObserver(([e]) => setBoxWidth(e.contentRect.width));
    ro.observe(box.current);
    return () => ro.disconnect();
  }, [vertical]);
  const zoom = vertical && boxWidth
    ? Math.max(0.3, Math.min(1.1, Math.max(boxWidth / layout.width, 0.8), (window.innerHeight * 1.5) / layout.height))
    : 1;
  const phone = vertical && boxWidth > 0;

  return (
    <div ref={box} className={phone ? 'graph-box graph-scroll' : 'graph-box'}>
    <div style={phone ? { width: Math.max(boxWidth, layout.width * zoom), height: layout.height * zoom } : { width: '100%', height: '100%' }}>
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      fitView={!phone}
      fitViewOptions={{ padding: 0.08, maxZoom: 1.4 }}
      viewport={phone ? { x: Math.max(0, (boxWidth - layout.width * zoom) / 2), y: 0, zoom } : undefined}
      onNodeClick={(_, n) => props.onSelect(n.id === props.selected ? undefined : n.id)}
      onPaneClick={() => props.onSelect(undefined)}
      nodesConnectable={false}
      minZoom={0.1}
      panOnDrag={!touch && !phone}
      zoomOnScroll={!phone}
      zoomOnPinch={!phone}
      zoomOnDoubleClick={!phone}
      preventScrolling={!touch && !phone}
    >
      <Background gap={20} color="#eee" />
      {!phone && <Controls showInteractive={false} />}
    </ReactFlow>
    </div>
    </div>
  );
}

// a new graph gets a fresh React Flow, which fits it to the view once measured
const graphKeys = new WeakMap<GraphSpec, number>();
let nextGraphKey = 1;

// memo: the training view re-renders every frame, the diagram only when its
// own inputs change
// On a phone the graph runs top to bottom, so it fills the narrow width and
// the page grows taller instead; on a touch screen a finger scrolls the page,
// not the graph (pinch still zooms, and the buttons zoom and fit).
export const GraphCanvas = memo(function GraphCanvas(props: GraphCanvasProps) {
  const vertical = useMedia(PHONE);
  const touch = useMedia(TOUCH);
  if (!graphKeys.has(props.spec)) graphKeys.set(props.spec, nextGraphKey++);
  return (
    <ReactFlowProvider key={`${graphKeys.get(props.spec)}${vertical ? 'v' : 'h'}`}>
      <Inner {...props} vertical={vertical} touch={touch} />
    </ReactFlowProvider>
  );
});
