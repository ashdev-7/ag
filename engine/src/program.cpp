#include "ag/program.hpp"
#include "ag/ops.hpp"

using namespace std;

namespace ag {

// errors from infer() do not know which node they are about; add it
static Status atNode(Status s, NodeId node) {
    if (!s.ok() && s.node == kNoNode) s.node = node;
    return s;
}

// ── compile ──────────────────────────────────────────────────────────────────
// every check and every size calculation happens here, once. builds into a
// local Program and assigns to *out only on success

Status compile(const Graph& g, const CompileOptions& opts, Program* out) {
    if (!out) return makeError(Code::InvalidArgument, "compile: null output");
    if (opts.batchSize < 1) return makeError(Code::InvalidArgument, "batch size must be at least 1");
    AG_TRY(validate(g, opts.limits));

    Program p;
    p.graph = g;
    p.batchSize = opts.batchSize;
    NodeId n = static_cast<NodeId>(g.nodes.size());
    p.nodes.resize(n);

    // 1. resolve the batch dimension and infer every shape, in node order
    for (NodeId id = 0; id < n; ++id) {
        Node& node = p.graph.nodes[id];
        const OpInfo& info = opInfo(node.op);
        Shape& declared = node.attrs.shape;
        for (int32_t d = 0; d < declared.rank; ++d)
            if (declared.dim[d] == kBatchDim) {
                if (node.op == Op::Param)
                    return makeError(Code::InvalidGraph, "a parameter cannot have a batch dimension", id);
                declared.dim[d] = opts.batchSize;
            }
        if (info.category == Category::Source || node.op == Op::Reshape)
            AG_TRY(atNode(validateShape(declared, opts.limits), id));

        array<Shape, kMaxInputs> in;
        for (int32_t i = 0; i < node.numInputs; ++i) in[i] = p.nodes[node.inputs[i]].shape;
        AG_TRY(atNode(info.infer(span<const Shape>(in.data(), node.numInputs), node.attrs, &p.nodes[id].shape), id));
        AG_TRY(atNode(validateShape(p.nodes[id].shape, opts.limits), id));
    }
    if (g.loss != kNoNode && numel(p.nodes[g.loss].shape) != 1)
        return makeError(Code::ShapeMismatch, "the loss must be a single number, got " + toString(p.nodes[g.loss].shape), g.loss);

    // 2. requiresGrad flows forward from parameters and marked inputs
    for (NodeId id = 0; id < n; ++id) {
        const Node& node = p.graph.nodes[id];
        bool rg = node.op == Op::Param || (node.op == Op::Input && node.attrs.requiresGrad);
        for (int32_t i = 0; i < node.numInputs; ++i) rg = rg || p.nodes[node.inputs[i]].requiresGrad;
        p.nodes[id].requiresGrad = rg;
    }

    // 3. which nodes the loss depends on, and how many gradients each receives
    vector<bool> reaches(n, false);
    if (g.loss != kNoNode) {
        reaches[g.loss] = true;
        for (NodeId id = g.loss; id >= 0; --id) {
            if (!reaches[id]) continue;
            const Node& node = p.graph.nodes[id];
            for (int32_t i = 0; i < node.numInputs; ++i) {
                reaches[node.inputs[i]] = true;
                if (p.nodes[node.inputs[i]].requiresGrad) p.nodes[node.inputs[i]].numConsumers++;
            }
        }
    }

    // 4. buffers. one arena: values for every node, grads for nodes on a path
    //    from a parameter to the loss, tangents for every node that has one
    uint64_t off = 0, poff = 0;
    for (NodeId id = 0; id < n; ++id) {
        NodePlan& np = p.nodes[id];
        uint64_t size = static_cast<uint64_t>(numel(np.shape));
        np.valueOff = static_cast<uint32_t>(off);
        off += size;
        if ((np.requiresGrad && reaches[id]) || id == g.loss) {
            np.gradOff = static_cast<uint32_t>(off);
            off += size;
        }
        if (opts.withTangents && np.requiresGrad) {
            np.tangentOff = static_cast<uint32_t>(off);
            off += size;
        }
        if (p.graph.nodes[id].op == Op::Param) {
            np.paramIndex = static_cast<int32_t>(p.params.size());
            p.params.push_back({id, static_cast<uint32_t>(poff), static_cast<uint32_t>(size)});
            poff += size;
        }
        if (off > opts.limits.maxArenaFloats)
            return makeError(Code::LimitExceeded, "the model needs more than " + to_string(opts.limits.maxArenaFloats) +
                                                  " floats of working memory at batch size " + to_string(opts.batchSize), id);
    }
    if (poff > opts.limits.maxParams)
        return makeError(Code::LimitExceeded, "the model has " + to_string(poff) + " parameters; the limit is " +
                                              to_string(opts.limits.maxParams));
    p.arenaFloats = static_cast<uint32_t>(off);
    p.paramFloats = static_cast<uint32_t>(poff);

    // 5. schedules. reverse node order is a valid backward order because every
    //    consumer of a node has a larger id
    auto& fwd = p.schedules[size_t(PassKind::Forward)];
    for (NodeId id = 0; id < n; ++id) fwd.push_back({StepKind::Forward, 0, id});

    if (g.loss != kNoNode) {
        auto& rev = p.schedules[size_t(PassKind::Reverse)];
        rev = fwd;
        rev.push_back({StepKind::SeedGrad, 0, g.loss});
        for (NodeId id = g.loss; id >= 0; --id) {
            if (!reaches[id] || !p.nodes[id].requiresGrad) continue;
            const Node& node = p.graph.nodes[id];
            for (int32_t i = 0; i < node.numInputs; ++i)
                if (p.nodes[node.inputs[i]].requiresGrad) rev.push_back({StepKind::Vjp, static_cast<uint8_t>(i), id});
        }
        auto& train = p.schedules[size_t(PassKind::Train)];
        train = rev;
        for (const ParamPlan& pp : p.params)
            if (reaches[pp.node]) train.push_back({StepKind::Update, 0, pp.node});
    }

    if (opts.withTangents) {
        auto& fm = p.schedules[size_t(PassKind::ForwardMode)];
        fm = fwd;
        fm.push_back({StepKind::SeedTangent, 0, kNoNode});
        for (NodeId id = 0; id < n; ++id)
            if (p.nodes[id].requiresGrad && opInfo(p.graph.nodes[id].op).category != Category::Source)
                fm.push_back({StepKind::Jvp, 0, id});
    }

    // 6. liveness over the Reverse schedule: the last step that reads a value.
    //    logical only (every buffer is allocated for the whole run); it feeds
    //    the memory timeline
    const auto& rev = p.schedules[size_t(PassKind::Reverse)];
    const auto& order = rev.empty() ? fwd : rev;
    for (int32_t k = 0; k < static_cast<int32_t>(order.size()); ++k) {
        const Step& st = order[k];
        const Node& node = p.graph.nodes[st.node];
        uint8_t saves = opInfo(node.op).saves;
        if (st.kind == StepKind::Forward) {
            p.nodes[st.node].lastUse = k;
            for (int32_t i = 0; i < node.numInputs; ++i) p.nodes[node.inputs[i]].lastUse = k;
        } else if (st.kind == StepKind::Vjp) {
            if (saves & kSavesInputs)
                for (int32_t i = 0; i < node.numInputs; ++i) p.nodes[node.inputs[i]].lastUse = k;
            if (saves & kSavesOutput) p.nodes[st.node].lastUse = k;
        }
    }

    *out = std::move(p);
    return {};
}

span<const Step> schedule(const Program& p, PassKind pass) {
    return p.schedules[static_cast<size_t>(pass)];
}

bool sameParamLayout(const Program& a, const Program& b) {
    if (a.params.size() != b.params.size()) return false;
    for (size_t i = 0; i < a.params.size(); ++i)
        if (a.params[i].node != b.params[i].node || a.params[i].offset != b.params[i].offset ||
            a.params[i].size != b.params[i].size)
            return false;
    return true;
}

} // namespace ag
