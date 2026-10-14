#include "ag/exec.hpp"
#include "ag/ops.hpp"
#include "ag/optim.hpp"

#include <algorithm>
#include <cmath>

using namespace std;

namespace ag {

// ── parameters ───────────────────────────────────────────────────────────────

void fillInit(span<float> out, const Shape& shape, Init init, float scale, Rng* rng) {
    float fanIn = shape.rank >= 1 ? float(shape.dim[0]) : 1.0f;
    float fanOut = shape.rank >= 2 ? float(shape.dim[1]) : fanIn;
    // the kind decides uniform or normal, not the sign of the bound: a
    // negative scale is a mirror image, not a request for zeros
    float width = 0;
    bool uniform = true;
    switch (init) {
        case Init::Zeros:         fill(out.begin(), out.end(), 0.0f); return;
        case Init::Constant:      fill(out.begin(), out.end(), scale); return;
        case Init::Uniform:       width = scale; break;
        case Init::Normal:        width = scale; uniform = false; break;
        case Init::XavierUniform: width = scale * sqrt(6.0f / (fanIn + fanOut)); break;
        case Init::XavierNormal:  width = scale * sqrt(2.0f / (fanIn + fanOut)); uniform = false; break;
        case Init::HeUniform:     width = scale * sqrt(6.0f / fanIn); break;
        case Init::HeNormal:      width = scale * sqrt(2.0f / fanIn); uniform = false; break;
    }
    width = fabs(width);
    for (float& x : out) x = uniform ? rngUniform(rng, -width, width) : width * rngNormal(rng);
}

Status initParams(const Program& p, uint64_t seed, ParamStore* out) {
    if (!out) return makeError(Code::InvalidArgument, "initParams: null output");
    ParamStore ps;
    ps.values.assign(p.paramFloats, 0.0f);
    ps.layout = p.params;
    for (const ParamPlan& pp : p.params) {
        const Node& node = p.graph.nodes[pp.node];
        Rng rng;
        rngSeed(&rng, seed, static_cast<uint64_t>(pp.node) + 1);
        fillInit(span<float>(ps.values).subspan(pp.offset, pp.size), p.nodes[pp.node].shape,
                 node.attrs.init, node.attrs.initScale, &rng);
    }
    *out = std::move(ps);
    return {};
}

Status exportParams(const ParamStore& ps, span<float> out) {
    if (out.size() != ps.values.size())
        return makeError(Code::InvalidArgument, "expected " + to_string(ps.values.size()) + " parameters, buffer holds " +
                                                to_string(out.size()));
    copy(ps.values.begin(), ps.values.end(), out.begin());
    return {};
}

Status importParams(ParamStore* ps, span<const float> in) {
    if (in.size() != ps->values.size())
        return makeError(Code::InvalidArgument, "expected " + to_string(ps->values.size()) + " parameters, got " +
                                                to_string(in.size()));
    for (size_t i = 0; i < in.size(); ++i)
        if (!isfinite(in[i])) return makeError(Code::NonFinite, "imported parameters contain nan or inf");
    copy(in.begin(), in.end(), ps->values.begin());
    return {};
}

// ── lifecycle ────────────────────────────────────────────────────────────────

Status execInit(const Program* prog, ParamStore* params, Exec* out) {
    if (!prog || !params || !out) return makeError(Code::InvalidArgument, "execInit: null argument");
    if (params->values.size() != prog->paramFloats || params->layout.size() != prog->params.size())
        return makeError(Code::InvalidArgument, "parameter store does not match the program");
    Exec e;
    e.prog = prog;
    e.params = params;
    // the only allocation the executor makes. calloc (with ABORTING_MALLOC=0
    // in the WASM build) returns null when the fixed heap is full
    if (!e.arena.allocate(prog->arenaFloats))
        return makeError(Code::OutOfMemory, "not enough memory for " + to_string(prog->arenaFloats) +
                                            " floats of working memory; close other runs or use a smaller model");
    *out = std::move(e);
    return {};
}

void execBindOptimizer(Exec* e, Optimizer* optim) { e->optim = optim; }

Status execBindInput(Exec* e, NodeId input, span<const float> data) {
    const Program& p = *e->prog;
    if (input < 0 || input >= static_cast<NodeId>(p.nodes.size()) || p.graph.nodes[input].op != Op::Input)
        return makeError(Code::InvalidArgument, "not an input node", input);
    const NodePlan& np = p.nodes[input];
    if (static_cast<int64_t>(data.size()) != numel(np.shape))
        return makeError(Code::ShapeMismatch, "input " + toString(np.shape) + " needs " + to_string(numel(np.shape)) +
                                              " values, got " + to_string(data.size()), input);
    copy(data.begin(), data.end(), e->arena.begin() + np.valueOff);
    return {};
}

Status execBegin(Exec* e, PassKind pass, TangentSeed seed) {
    const Program& p = *e->prog;
    if (pass == PassKind::Train && !e->optim) return makeError(Code::InvalidState, "a training pass needs an optimizer");
    if (pass == PassKind::ForwardMode) {
        if (p.schedules[size_t(pass)].empty())
            return makeError(Code::InvalidState, "this program was compiled without tangents");
        if (seed.node < 0 || seed.node >= static_cast<NodeId>(p.nodes.size()))
            return makeError(Code::InvalidArgument, "forward mode needs a seed node", seed.node);
        Op op = p.graph.nodes[seed.node].op;
        if ((op != Op::Param && op != Op::Input) || p.nodes[seed.node].tangentOff == kNoBuffer)
            return makeError(Code::InvalidArgument, "forward mode is seeded at a parameter or an input that requires grad", seed.node);
        if (seed.element < 0 || seed.element >= numel(p.nodes[seed.node].shape))
            return makeError(Code::InvalidArgument, "seed element out of range", seed.node);
    }
    // everything the pass computes starts at zero, so the state after k steps
    // does not depend on what ran before. input values stay: they are bound
    for (size_t i = 0; i < p.nodes.size(); ++i) {
        const NodePlan& np = p.nodes[i];
        int64_t size = numel(np.shape);
        if (p.graph.nodes[i].op != Op::Input) fill_n(e->arena.begin() + np.valueOff, size, 0.0f);
        if (np.gradOff != kNoBuffer) fill_n(e->arena.begin() + np.gradOff, size, 0.0f);
        if (np.tangentOff != kNoBuffer) fill_n(e->arena.begin() + np.tangentOff, size, 0.0f);
    }
    e->pass = pass;
    e->seed = seed;
    e->cursor = 0;
    e->begun = true;
    e->fault = {};
    return {};
}

// ── the loop ─────────────────────────────────────────────────────────────────

static TensorView view(Exec* e, NodeId node, uint32_t off) {
    return {off == kNoBuffer ? nullptr : e->arena.data() + off, e->prog->nodes[node].shape};
}

static ConstTensorView cview(const Exec& e, NodeId node, uint32_t off) {
    return {off == kNoBuffer ? nullptr : e.arena.data() + off, e.prog->nodes[node].shape};
}

static Status checkFinite(const Exec& e, NodeId node, uint32_t off, const char* what) {
    int64_t bad = firstNonFinite(cview(e, node, off));
    if (bad < 0) return {};
    const Node& n = e.prog->graph.nodes[node];
    return makeError(Code::NonFinite, string(what) + " of " + (n.name.empty() ? opInfo(n.op).name : n.name) +
                                      " became nan or inf at element " + to_string(bad), node);
}

static Status runStep(Exec* e, const Step& st) {
    const Program& p = *e->prog;
    const NodePlan& np = p.nodes[st.node < 0 ? 0 : st.node];
    const Node& node = p.graph.nodes[st.node < 0 ? 0 : st.node];
    const OpInfo& info = opInfo(node.op);

    array<ConstTensorView, kMaxInputs> in;
    for (int32_t i = 0; i < node.numInputs; ++i) in[i] = cview(*e, node.inputs[i], p.nodes[node.inputs[i]].valueOff);
    span<const ConstTensorView> ins(in.data(), node.numInputs);

    switch (st.kind) {
        case StepKind::Forward: {
            TensorView out = view(e, st.node, np.valueOff);
            if (node.op == Op::Param) {
                const ParamPlan& pp = p.params[np.paramIndex];
                copy_n(e->params->values.begin() + pp.offset, pp.size, out.data);
            } else {
                info.forward({ins, out, node.attrs});
            }
            return checkFinite(*e, st.node, np.valueOff, "value");
        }
        case StepKind::SeedGrad:
            e->arena[np.gradOff] = 1.0f;
            return {};
        case StepKind::Vjp: {
            array<TensorView, kMaxInputs> gin;
            for (int32_t i = 0; i < node.numInputs; ++i) gin[i] = view(e, node.inputs[i], p.nodes[node.inputs[i]].gradOff);
            info.vjp({ins, cview(*e, st.node, np.valueOff), cview(*e, st.node, np.gradOff),
                      span<const TensorView>(gin.data(), node.numInputs), 1u << st.slot, node.attrs});
            NodeId target = node.inputs[st.slot];
            return checkFinite(*e, target, p.nodes[target].gradOff, "gradient");
        }
        case StepKind::Update: {
            const ParamPlan& pp = p.params[np.paramIndex];
            Optimizer* o = e->optim;
            float lr = lrAt(o->cfg.lr, o->t);
            float scale = clipScale(execGradNorm(*e), o->cfg.clipNorm);
            span<float> value = span<float>(e->params->values).subspan(pp.offset, pp.size);
            optimUpdate(o, pp, value, span<const float>(e->arena.data() + np.gradOff, pp.size), lr, scale);
            for (uint32_t i = 0; i < pp.size; ++i)
                if (!isfinite(value[i]))
                    return makeError(Code::NonFinite, "update made " + node.name + " nan or inf at element " + to_string(i), st.node);
            return {};
        }
        case StepKind::SeedTangent: {
            const NodePlan& sp = p.nodes[e->seed.node];
            e->arena[sp.tangentOff + e->seed.element] = 1.0f;
            return {};
        }
        case StepKind::Jvp: {
            array<ConstTensorView, kMaxInputs> tin;
            for (int32_t i = 0; i < node.numInputs; ++i) tin[i] = cview(*e, node.inputs[i], p.nodes[node.inputs[i]].tangentOff);
            info.jvp({ins, cview(*e, st.node, np.valueOff), span<const ConstTensorView>(tin.data(), node.numInputs),
                      view(e, st.node, np.tangentOff), node.attrs});
            return checkFinite(*e, st.node, np.tangentOff, "tangent");
        }
    }
    return makeError(Code::Internal, "unknown step kind");
}

Status execAdvance(Exec* e, int32_t n, int32_t* executed) {
    if (executed) *executed = 0;
    if (!e->begun) return makeError(Code::InvalidState, "call execBegin before advancing");
    if (!e->fault.ok()) return e->fault;
    span<const Step> steps = schedule(*e->prog, e->pass);
    int32_t len = static_cast<int32_t>(steps.size());
    for (int32_t k = 0; k < n && e->cursor < len; ++k) {
        Status s = runStep(e, steps[e->cursor]);
        if (!s.ok()) {
            e->fault = s;
            return s;
        }
        ++e->cursor;
        if (executed) ++*executed;
        // the last Update of a training step completes it
        if (e->pass == PassKind::Train && e->cursor == len) optimEndStep(e->optim);
    }
    return {};
}

Status execRun(Exec* e) { return execAdvance(e, execLength(*e), nullptr); }

Status execSeek(Exec* e, int32_t index) {
    if (!e->begun) return makeError(Code::InvalidState, "call execBegin before seeking");
    if (e->pass == PassKind::Train) return makeError(Code::InvalidState, "a training pass is seeked by the session");
    if (index < 0 || index > execLength(*e)) return makeError(Code::InvalidArgument, "seek index out of range");
    AG_TRY(execBegin(e, e->pass, e->seed));
    return execAdvance(e, index, nullptr);
}

int32_t execLength(const Exec& e) {
    return e.prog ? static_cast<int32_t>(schedule(*e.prog, e.pass).size()) : 0;
}

// ── reading ──────────────────────────────────────────────────────────────────

static uint32_t offsetOf(const NodePlan& np, Which which) {
    switch (which) {
        case Which::Value:   return np.valueOff;
        case Which::Grad:    return np.gradOff;
        case Which::Tangent: return np.tangentOff;
    }
    return kNoBuffer;
}

ConstTensorView execView(const Exec& e, NodeId node, Which which) {
    if (!e.prog || node < 0 || node >= static_cast<NodeId>(e.prog->nodes.size())) return {};
    return cview(e, node, offsetOf(e.prog->nodes[node], which));
}

Status execRead(const Exec& e, NodeId node, Which which, span<float> out) {
    ConstTensorView v = execView(e, node, which);
    if (!v.data) return makeError(Code::NotFound, "this node has no such buffer", node);
    int64_t n = numel(v.shape);
    if (static_cast<int64_t>(out.size()) < n) return makeError(Code::InvalidArgument, "read buffer too small", node);
    copy_n(v.data, n, out.begin());
    return {};
}

Status execStats(const Exec& e, NodeId node, Which which, Stats* out) {
    ConstTensorView v = execView(e, node, which);
    if (!v.data) return makeError(Code::NotFound, "this node has no such buffer", node);
    computeStats(v, out);
    return {};
}

float execLoss(const Exec& e) {
    NodeId loss = e.prog ? e.prog->graph.loss : kNoNode;
    return loss == kNoNode ? NAN : e.arena[e.prog->nodes[loss].valueOff];
}

float execGradNorm(const Exec& e) {
    double acc = 0;
    for (const ParamPlan& pp : e.prog->params) {
        uint32_t off = e.prog->nodes[pp.node].gradOff;
        if (off == kNoBuffer) continue;
        for (uint32_t i = 0; i < pp.size; ++i) acc += double(e.arena[off + i]) * e.arena[off + i];
    }
    return toFloat(sqrt(acc));
}

} // namespace ag
