// The network as a data-flow diagram. Each node shows its op, shape, value
// and gradient; the step that just ran is outlined, and during the backward
// pass the edge carrying gradient is drawn moving against the arrow.

import { memo, useMemo } from 'react';
import { Background, Controls, Handle, Position, ReactFlow, ReactFlowProvider } from '@xyflow/react';
import type { Edge, Node, NodeProps } from '@xyflow/react';
import type { GraphSpec, NodeSpec } from '../graph/spec.ts';
import type { Stats } from '../engine/protocol.ts';
import { fmt } from '../views/draw.ts';
import { layoutGraph, OP_SYMBOL } from '../views/graph.ts';
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
}

function summary(s: Stats | undefined, shape: number[] | undefined): string {
  if (!s) return '';
  const n = shape ? shape.reduce((a, b) => a * b, 1) : 1;
  if (s.nonFinite) return 'nan';
  return n === 1 ? fmt(s.mean) : `μ ${fmt(s.mean)}  σ ${fmt(s.std)}`;
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
      <Handle type="target" position={Position.Left} isConnectable={false} />
      <div className="gnode-head">
        {showOp && <span className="gnode-op">{OP_SYMBOL[spec.op] ?? spec.op}</span>}
        <span className="gnode-name">{label}</span>
      </div>
      {!data.compact && shape && shape.length > 0 && <div className="gnode-shape">[{shape.join(', ')}]</div>}
      <div className="gnode-value">{data.done ? summary(data.value, shape) : ' '}</div>
      <div className="gnode-grad">{data.gradDone && data.grad ? `${data.gradLabel} ${summary(data.grad, shape)}` : ' '}</div>
      <Handle type="source" position={Position.Right} isConnectable={false} />
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
}

function Inner(props: GraphCanvasProps) {
  const { spec, highlight } = props;
  const compact = spec.nodes.length > 40;
  const layout = useMemo(
    () => layoutGraph(spec, () => (compact ? { width: 92, height: 46 } : { width: 138, height: 70 })),
    [spec, compact],
  );
  const byId = useMemo(() => new Map(spec.nodes.map((n) => [n.id, n])), [spec]);
  // sizes are fixed by the layout, so React Flow never has to measure a node
  // before showing it or fitting the view
  const nodes: Node<OpData>[] = layout.nodes.map((n) => ({
    id: n.id,
    type: 'op',
    position: { x: n.x, y: n.y },
    width: n.width,
    height: n.height,
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
    },
  }));

  let maxNorm = 0;
  if (props.edgeWeights) for (const s of Object.values(props.grads)) maxNorm = Math.max(maxNorm, s.l2);
  const edges: Edge[] = layout.edges.map((e) => {
    const active = highlight.activeEdge && highlight.activeEdge.from === e.from && highlight.activeEdge.to === e.to &&
      highlight.activeEdge.slot === e.slot;
    const w = props.edgeWeights && maxNorm > 0 ? 1 + 5 * Math.sqrt((props.grads[e.from]?.l2 ?? 0) / maxNorm) : 1.5;
    return {
      id: e.id, source: e.from, target: e.to,
      animated: !!active,
      className: active ? 'gedge-active' : '',
      style: { strokeWidth: active ? 3 : w },
    };
  });

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      fitView
      fitViewOptions={{ padding: 0.08, maxZoom: 1.4 }}
      onNodeClick={(_, n) => props.onSelect(n.id === props.selected ? undefined : n.id)}
      onPaneClick={() => props.onSelect(undefined)}
      nodesConnectable={false}
      minZoom={0.1}
    >
      <Background gap={20} color="#eee" />
      <Controls showInteractive={false} />
    </ReactFlow>
  );
}

// a new graph gets a fresh React Flow, which fits it to the view once measured
const graphKeys = new WeakMap<GraphSpec, number>();
let nextGraphKey = 1;

export function GraphCanvas(props: GraphCanvasProps) {
  if (!graphKeys.has(props.spec)) graphKeys.set(props.spec, nextGraphKey++);
  return (
    <ReactFlowProvider key={graphKeys.get(props.spec)}>
      <Inner {...props} />
    </ReactFlowProvider>
  );
}
