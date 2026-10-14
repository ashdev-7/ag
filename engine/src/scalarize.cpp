#include "ag/scalarize.hpp"
#include "bcast.hpp"
#include "scalar_rules.hpp"

using namespace std;

namespace ag {

// "w" with shape [2, 3], element 4 -> "w[1,1]"
static string elemName(const string& base, const Shape& s, int64_t e) {
    if (s.rank == 0) return base;
    array<int64_t, kMaxRank> idx{};
    for (int32_t i = s.rank - 1; i >= 0; --i) {
        idx[i] = e % s.dim[i];
        e /= s.dim[i];
    }
    string out = base + "[";
    for (int32_t i = 0; i < s.rank; ++i) out += (i ? "," : "") + to_string(idx[i]);
    return out + "]";
}

// append one rank-0 node. every scalar node goes through here, so this is
// where the size limit is enforced
static Status emit(const ScalarizeArgs& a, Op op, initializer_list<NodeId> in, float scalar, const string& name,
                   NodeId* out) {
    if (a.out->nodes.size() >= a.maxNodes)
        return makeError(Code::LimitExceeded, "the scalar view is limited to " + to_string(a.maxNodes) +
                                              " nodes; this graph needs more. use a smaller batch or layer");
    Attrs attrs;
    attrs.scalar = scalar;
    return addNode(a.out, op, span<const NodeId>(in.begin(), in.size()), attrs, name, a.node.name, out);
}

static string outName(const ScalarizeArgs& a, int64_t e) { return elemName(a.node.name, a.outShape, e); }

// a + b + c + ... as a left-leaning chain; the last node gets `name`
static Status chainSum(const ScalarizeArgs& a, const vector<NodeId>& items, const string& name, NodeId* out) {
    NodeId acc = items[0];
    for (size_t k = 1; k < items.size(); ++k)
        AG_TRY(emit(a, Op::Add, {acc, items[k]}, 0, k + 1 == items.size() ? name : "", &acc));
    *out = acc;
    return {};
}

static Status scaleBy(const ScalarizeArgs& a, NodeId x, float k, const string& name, NodeId* out) {
    NodeId c;
    AG_TRY(emit(a, Op::Const, {}, k, "", &c));
    return emit(a, Op::Mul, {x, c}, 0, name, out);
}

// ── rules ────────────────────────────────────────────────────────────────────

Status scalarizeElementwise(const ScalarizeArgs& a) {
    int64_t n = numel(a.outShape);
    a.outElems->resize(n);
    if (a.node.numInputs == 1) {
        for (int64_t i = 0; i < n; ++i)
            AG_TRY(emit(a, a.node.op, {a.in[0][i]}, a.node.attrs.scalar, outName(a, i), &(*a.outElems)[i]));
        return {};
    }
    Bcast bx = bcastTo(a.outShape, a.inShapes[0]), by = bcastTo(a.outShape, a.inShapes[1]);
    for (int64_t i = 0; i < n; ++i)
        AG_TRY(emit(a, a.node.op, {a.in[0][at(bx, i)], a.in[1][at(by, i)]}, 0, outName(a, i), &(*a.outElems)[i]));
    return {};
}

Status scalarizeMatMul(const ScalarizeArgs& a) {
    int64_t m = a.inShapes[0].dim[0], k = a.inShapes[0].dim[1], n = a.inShapes[1].dim[1];
    a.outElems->resize(m * n);
    for (int64_t i = 0; i < m; ++i)
        for (int64_t j = 0; j < n; ++j) {
            vector<NodeId> prods(k);
            string name = outName(a, i * n + j);
            for (int64_t p = 0; p < k; ++p)
                AG_TRY(emit(a, Op::Mul, {a.in[0][i * k + p], a.in[1][p * n + j]}, 0, k == 1 ? name : "", &prods[p]));
            AG_TRY(chainSum(a, prods, name, &(*a.outElems)[i * n + j]));
        }
    return {};
}

Status scalarizeReduce(const ScalarizeArgs& a) {
    const Shape& s = a.inShapes[0];
    int64_t outer = 1, len = numel(s), inner = 1;
    if (a.node.attrs.axis != kAllAxes) {
        int32_t axis = a.node.attrs.axis < 0 ? a.node.attrs.axis + s.rank : a.node.attrs.axis;
        outer = 1;
        for (int32_t i = 0; i < axis; ++i) outer *= s.dim[i];
        len = s.dim[axis];
        inner = 1;
        for (int32_t i = axis + 1; i < s.rank; ++i) inner *= s.dim[i];
    }
    bool mean = a.node.op == Op::Mean;
    a.outElems->resize(outer * inner);
    for (int64_t o = 0; o < outer; ++o)
        for (int64_t i = 0; i < inner; ++i) {
            vector<NodeId> items(len);
            for (int64_t k = 0; k < len; ++k) items[k] = a.in[0][(o * len + k) * inner + i];
            string name = outName(a, o * inner + i);
            NodeId total;
            AG_TRY(chainSum(a, items, mean && len > 1 ? "" : name, &total));
            if (mean && len > 1) AG_TRY(scaleBy(a, total, 1.0f / float(len), name, &total));
            (*a.outElems)[o * inner + i] = total;
        }
    return {};
}

Status scalarizeReshape(const ScalarizeArgs& a) {
    *a.outElems = a.in[0];
    return {};
}

Status scalarizeTranspose(const ScalarizeArgs& a) {
    int64_t r = a.inShapes[0].dim[0], c = a.inShapes[0].dim[1];
    a.outElems->resize(r * c);
    for (int64_t i = 0; i < r; ++i)
        for (int64_t j = 0; j < c; ++j) (*a.outElems)[j * r + i] = a.in[0][i * c + j];
    return {};
}

// written without the max subtraction: exp, sum, divide. the values agree with
// the tensor op to rounding as long as the logits are moderate, which they are
// at the sizes the scalar view allows
Status scalarizeSoftmax(const ScalarizeArgs& a) {
    const Shape& s = a.inShapes[0];
    int32_t axis = a.node.attrs.axis == kAllAxes ? s.rank - 1 : (a.node.attrs.axis < 0 ? a.node.attrs.axis + s.rank : a.node.attrs.axis);
    int64_t outer = 1, len = s.dim[axis], inner = 1;
    for (int32_t i = 0; i < axis; ++i) outer *= s.dim[i];
    for (int32_t i = axis + 1; i < s.rank; ++i) inner *= s.dim[i];
    a.outElems->resize(numel(s));
    for (int64_t o = 0; o < outer; ++o)
        for (int64_t i = 0; i < inner; ++i) {
            vector<NodeId> e(len);
            for (int64_t k = 0; k < len; ++k) AG_TRY(emit(a, Op::Exp, {a.in[0][(o * len + k) * inner + i]}, 0, "", &e[k]));
            NodeId total;
            AG_TRY(chainSum(a, e, "", &total));
            for (int64_t k = 0; k < len; ++k) {
                int64_t idx = (o * len + k) * inner + i;
                AG_TRY(emit(a, Op::Div, {e[k], total}, 0, outName(a, idx), &(*a.outElems)[idx]));
            }
        }
    return {};
}

Status scalarizeMse(const ScalarizeArgs& a) {
    int64_t n = numel(a.inShapes[0]);
    vector<NodeId> sq(n);
    for (int64_t i = 0; i < n; ++i) {
        NodeId d;
        AG_TRY(emit(a, Op::Sub, {a.in[0][i], a.in[1][i]}, 0, "", &d));
        AG_TRY(emit(a, Op::Mul, {d, d}, 0, "", &sq[i]));
    }
    NodeId total;
    AG_TRY(chainSum(a, sq, "", &total));
    a.outElems->resize(1);
    return scaleBy(a, total, 1.0f / float(n), a.node.name, &(*a.outElems)[0]);
}

// loss = (1/R) Σ_rows Σ_c t_c (log Σ_k exp z_k − z_c)
Status scalarizeCrossEntropy(const ScalarizeArgs& a) {
    const Shape& s = a.inShapes[0];
    int64_t c = s.dim[s.rank - 1], rows = numel(s) / c;
    vector<NodeId> terms;
    for (int64_t r = 0; r < rows; ++r) {
        vector<NodeId> e(c);
        for (int64_t k = 0; k < c; ++k) AG_TRY(emit(a, Op::Exp, {a.in[0][r * c + k]}, 0, "", &e[k]));
        NodeId total, lse;
        AG_TRY(chainSum(a, e, "", &total));
        AG_TRY(emit(a, Op::Log, {total}, 0, "lse[" + to_string(r) + "]", &lse));
        for (int64_t k = 0; k < c; ++k) {
            NodeId d, t;
            AG_TRY(emit(a, Op::Sub, {lse, a.in[0][r * c + k]}, 0, "", &d));
            AG_TRY(emit(a, Op::Mul, {a.in[1][r * c + k], d}, 0, "", &t));
            terms.push_back(t);
        }
    }
    NodeId total;
    AG_TRY(chainSum(a, terms, "", &total));
    a.outElems->resize(1);
    return scaleBy(a, total, 1.0f / float(rows), a.node.name, &(*a.outElems)[0]);
}

// ── driver ───────────────────────────────────────────────────────────────────

Status scalarize(const Program& p, const Limits& limits, Graph* out, ScalarMap* map) {
    if (!out || !map) return makeError(Code::InvalidArgument, "scalarize: null output");
    Graph g;
    ScalarMap m;
    NodeId n = static_cast<NodeId>(p.graph.nodes.size());
    m.elems.resize(n);

    for (NodeId id = 0; id < n; ++id) {
        const Node& node = p.graph.nodes[id];
        const OpInfo& info = opInfo(node.op);
        const Shape& shape = p.nodes[id].shape;
        size_t start = g.nodes.size();
        ScalarizeArgs args{node, {}, shape, {}, &g, &m.elems[id], limits.maxScalarNodes};

        if (info.category == Category::Source) {
            // leaves: values are filled in by the caller. an input that does
            // not need a gradient becomes a constant
            int64_t count = numel(shape);
            m.elems[id].resize(count);
            for (int64_t e = 0; e < count; ++e) {
                Op op = node.op == Op::Input && !node.attrs.requiresGrad ? Op::Const : node.op;
                AG_TRY(emit(args, op, {}, node.op == Op::Const ? node.attrs.scalar : 0.0f,
                            elemName(node.name, shape, e), &m.elems[id][e]));
                Attrs& at = g.nodes.back().attrs;
                at.init = Init::Constant;
                at.requiresGrad = node.attrs.requiresGrad;
            }
        } else {
            if (!info.scalarize)
                return makeError(Code::Unimplemented, string(info.name) + " has no scalar view", id);
            array<Shape, kMaxInputs> inShapes;
            for (int32_t i = 0; i < node.numInputs; ++i) inShapes[i] = p.nodes[node.inputs[i]].shape;
            vector<vector<NodeId>> inElems(node.numInputs);
            for (int32_t i = 0; i < node.numInputs; ++i) inElems[i] = m.elems[node.inputs[i]];
            ScalarizeArgs a{node, span<const Shape>(inShapes.data(), node.numInputs), shape, inElems, &g,
                            &m.elems[id], limits.maxScalarNodes};
            Status s = info.scalarize(a);
            if (!s.ok()) {
                s.node = id;
                return s;
            }
        }

        m.source.resize(g.nodes.size(), id);
        m.element.resize(g.nodes.size(), -1);
        for (size_t e = 0; e < m.elems[id].size(); ++e)
            if (static_cast<size_t>(m.elems[id][e]) >= start) m.element[m.elems[id][e]] = static_cast<int32_t>(e);
    }

    if (p.graph.loss != kNoNode) g.loss = m.elems[p.graph.loss][0];
    for (NodeId o : p.graph.outputs)
        for (NodeId e : m.elems[o]) g.outputs.push_back(e);
    *out = std::move(g);
    *map = std::move(m);
    return {};
}

} // namespace ag
