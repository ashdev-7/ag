#include "ag/graph.hpp"
#include "ag/ops.hpp"

#include <cmath>

using namespace std;

namespace ag {

static bool validOp(Op op) { return static_cast<int>(op) < kOpCount; }

// ── building ─────────────────────────────────────────────────────────────────
// inputs must already exist, which is what keeps the node list topological

Status addNode(Graph* g, Op op, span<const NodeId> inputs, const Attrs& attrs,
               string_view name, string_view module, NodeId* outId) {
    if (!g || !outId) return makeError(Code::InvalidArgument, "addNode: null argument");
    if (!validOp(op)) return makeError(Code::InvalidGraph, "addNode: unknown op");

    NodeId id = static_cast<NodeId>(g->nodes.size());
    const OpInfo& info = opInfo(op);
    if (inputs.size() < info.minInputs || inputs.size() > info.maxInputs)
        return makeError(Code::InvalidGraph,
                         string(info.name) + " takes " + to_string(info.minInputs) +
                         (info.minInputs == info.maxInputs ? "" : " to " + to_string(info.maxInputs)) +
                         " inputs, got " + to_string(inputs.size()), id);

    Node n;
    n.op = op;
    n.numInputs = static_cast<uint8_t>(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i] < 0 || inputs[i] >= id)
            return makeError(Code::InvalidGraph, "input refers to a node that does not exist yet",
                             id, static_cast<int32_t>(i));
        n.inputs[i] = inputs[i];
    }
    n.attrs = attrs;
    n.name = name;
    n.module = module;

    g->nodes.push_back(std::move(n));
    *outId = id;
    return {};
}

Status setLoss(Graph* g, NodeId node) {
    if (!g) return makeError(Code::InvalidArgument, "setLoss: null graph");
    if (node < 0 || node >= static_cast<NodeId>(g->nodes.size()))
        return makeError(Code::InvalidGraph, "setLoss: no such node", node);
    g->loss = node;
    return {};
}

Status addOutput(Graph* g, NodeId node) {
    if (!g) return makeError(Code::InvalidArgument, "addOutput: null graph");
    if (node < 0 || node >= static_cast<NodeId>(g->nodes.size()))
        return makeError(Code::InvalidGraph, "addOutput: no such node", node);
    g->outputs.push_back(node);
    return {};
}

// ── whole-graph checks ───────────────────────────────────────────────────────

// addNode already guarantees most of this; validate exists because a Graph is
// plain data and can be edited directly, and because limits apply here

Status validate(const Graph& g, const Limits& limits) {
    NodeId n = static_cast<NodeId>(g.nodes.size());
    if (n == 0) return makeError(Code::InvalidGraph, "the graph is empty");
    if (static_cast<uint32_t>(n) > limits.maxNodes)
        return makeError(Code::LimitExceeded, "the graph has " + to_string(n) + " nodes; the limit is " +
                                              to_string(limits.maxNodes));

    int features = 0, targets = 0;
    for (NodeId id = 0; id < n; ++id) {
        const Node& node = g.nodes[id];
        if (!validOp(node.op)) return makeError(Code::InvalidGraph, "unknown op", id);
        const OpInfo& info = opInfo(node.op);
        if (!info.infer || !info.forward)
            return makeError(Code::Unimplemented, string(info.name) + " has no kernel", id);
        if (node.numInputs < info.minInputs || node.numInputs > info.maxInputs)
            return makeError(Code::InvalidGraph, string(info.name) + " has the wrong number of inputs", id);
        for (int32_t i = 0; i < node.numInputs; ++i)
            if (node.inputs[i] < 0 || node.inputs[i] >= id)
                return makeError(Code::InvalidGraph, "input refers to a node that does not come before it", id, i);
        // a nan exponent or init scale would only surface later as a nan
        // somewhere else; say where it comes from instead
        if (!std::isfinite(node.attrs.scalar) || !std::isfinite(node.attrs.initScale))
            return makeError(Code::InvalidGraph, "attributes must be finite numbers", id);
        if (node.attrs.noGrad && opInfo(node.op).category == Category::Source)
            return makeError(Code::InvalidGraph, "no_grad applies to computed nodes, not to inputs or parameters", id);
        if (node.attrs.role != Role::None && node.op != Op::Input)
            return makeError(Code::InvalidGraph, "only an input can carry a data role", id);
        features += node.attrs.role == Role::Features;
        targets += node.attrs.role == Role::Targets;
    }
    if (features > 1 || targets > 1)
        return makeError(Code::InvalidGraph, "at most one input may hold features and one targets");
    if (g.loss != kNoNode && (g.loss < 0 || g.loss >= n))
        return makeError(Code::InvalidGraph, "the loss is not a node of this graph");
    for (NodeId o : g.outputs)
        if (o < 0 || o >= n) return makeError(Code::InvalidGraph, "an output is not a node of this graph");
    return {};
}

// one line per node: id, op, inputs, attributes that op reads, name

string dump(const Graph& g) {
    string s;
    for (size_t id = 0; id < g.nodes.size(); ++id) {
        const Node& node = g.nodes[id];
        s += "%" + to_string(id) + " = " + opInfo(node.op).name + "(";
        for (int32_t i = 0; i < node.numInputs; ++i) s += (i ? ", %" : "%") + to_string(node.inputs[i]);
        s += ")";
        if (opInfo(node.op).category == Category::Source || node.op == Op::Reshape) s += " shape=" + toString(node.attrs.shape);
        if (node.attrs.axis != kAllAxes) s += " axis=" + to_string(node.attrs.axis);
        if (node.attrs.keepDim) s += " keepdim";
        if (node.attrs.noGrad) s += " no_grad";
        if (node.attrs.scalar != 0.0f) s += " scalar=" + to_string(node.attrs.scalar);
        if (!node.name.empty()) s += "  # " + node.name;
        s += "\n";
    }
    s += "loss = %" + to_string(g.loss) + "\n";
    return s;
}

} // namespace ag
