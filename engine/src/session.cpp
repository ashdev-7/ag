#include "ag/session.hpp"

#include <algorithm>
#include <cmath>

using namespace std;

namespace ag {

// ── helpers ──────────────────────────────────────────────────────────────────

static NodeId findRole(const Graph& g, Role role) {
    for (size_t i = 0; i < g.nodes.size(); ++i)
        if (g.nodes[i].op == Op::Input && g.nodes[i].attrs.role == role) return static_cast<NodeId>(i);
    return kNoNode;
}

// inputs without a data role (the a and b of "a * b + a") get their values
// from their init rule, seeded like parameters
static Status fillFreeInputs(Session* s, Exec* e, const Program& p) {
    for (size_t i = 0; i < p.graph.nodes.size(); ++i) {
        const Node& node = p.graph.nodes[i];
        if (node.op != Op::Input || node.attrs.role != Role::None) continue;
        vector<float> buf(numel(p.nodes[i].shape));
        Rng rng;
        rngSeed(&rng, s->cfg.seed, i + 1);
        fillInit(buf, p.nodes[i].shape, node.attrs.init, node.attrs.initScale, &rng);
        AG_TRY(execBindInput(e, static_cast<NodeId>(i), buf));
    }
    return {};
}

// rows of the dataset into the features and targets inputs of one executor
static Status bindRows(Session* s, Exec* e, span<const float> x, span<const float> y) {
    NodeId fx = findRole(e->prog->graph, Role::Features), fy = findRole(e->prog->graph, Role::Targets);
    if (fx != kNoNode) AG_TRY(execBindInput(e, fx, x));
    if (fy != kNoNode) AG_TRY(execBindInput(e, fy, y));
    (void)s;
    return {};
}

static Status drawBatch(Session* s) {
    if (s->data.n == 0) return {};
    AG_TRY(nextBatch(&s->batcher, s->data, s->cfg.batchSize, s->batchX, s->batchY));
    return bindRows(s, &s->train, s->batchX, s->batchY);
}

static void takeSnapshot(Session* s) {
    Snapshot& snap = s->stepStart;
    snap.params = s->params.values;
    snap.m = s->optim.m;
    snap.v = s->optim.v;
    snap.t = s->optim.t;
    snap.batcher = s->batcher;
    snap.step = s->step;
    snap.valid = true;
}

static void restoreSnapshot(Session* s, bool batcherToo) {
    const Snapshot& snap = s->stepStart;
    if (!snap.valid) return;
    s->params.values = snap.params;
    s->optim.m = snap.m;
    s->optim.v = snap.v;
    s->optim.t = snap.t;
    s->step = snap.step;
    if (batcherToo) s->batcher = snap.batcher;
}

// a data input must be [B, d] where d is the dataset's width
static Status checkDataInput(const Program& p, NodeId node, int32_t width, const char* what) {
    if (node == kNoNode || width == 0) return {};
    const Shape& sh = p.nodes[node].shape;
    if (sh.rank != 2 || sh.dim[0] != p.batchSize || sh.dim[1] != width)
        return makeError(Code::ShapeMismatch, string(what) + " must have shape [B, " + to_string(width) +
                                              "] for this dataset, got " + toString(sh), node);
    return {};
}

// ── lifecycle ────────────────────────────────────────────────────────────────
// everything is built into a local and moved to *out only on success, so a
// failed create leaves nothing behind

Status sessionCreate(const Graph& g, const SessionConfig& cfg, unique_ptr<Session>* out) {
    if (!out) return makeError(Code::InvalidArgument, "sessionCreate: null output");
    if (cfg.batchSize < 1 || cfg.evalBatchSize < 1) return makeError(Code::InvalidArgument, "batch sizes must be at least 1");
    AG_TRY(optimValidate(cfg.optim));

    auto s = make_unique<Session>();
    s->cfg = cfg;
    AG_TRY(makeDataset(cfg.data, &s->data));
    if (s->data.n > 0 && cfg.batchSize > s->data.nTrain)
        return makeError(Code::InvalidArgument, "batch size " + to_string(cfg.batchSize) + " is larger than the " +
                                                to_string(s->data.nTrain) + " training rows");

    CompileOptions co;
    co.batchSize = cfg.batchSize;
    co.withTangents = cfg.withTangents;
    co.limits = cfg.limits;
    AG_TRY(compile(g, co, &s->trainProg));
    co.batchSize = cfg.evalBatchSize;
    co.withTangents = false;
    AG_TRY(compile(g, co, &s->evalProg));
    if (!sameParamLayout(s->trainProg, s->evalProg))
        return makeError(Code::Internal, "train and eval programs disagree on parameters");

    for (const Program* p : {&s->trainProg, &s->evalProg}) {
        AG_TRY(checkDataInput(*p, findRole(g, Role::Features), s->data.xDim, "the features input"));
        AG_TRY(checkDataInput(*p, findRole(g, Role::Targets), s->data.yDim, "the targets input"));
    }
    s->batchX.resize(size_t(cfg.batchSize) * s->data.xDim);
    s->batchY.resize(size_t(cfg.batchSize) * s->data.yDim);

    AG_TRY(sessionReset(s.get(), cfg.seed));
    *out = std::move(s);
    return {};
}

Status sessionReset(Session* s, uint64_t seed) {
    s->cfg.seed = seed;
    AG_TRY(initParams(s->trainProg, seed, &s->params));
    AG_TRY(optimInit(s->cfg.optim, s->params, &s->optim));
    AG_TRY(execInit(&s->trainProg, &s->params, &s->train));
    AG_TRY(execInit(&s->evalProg, &s->params, &s->eval));
    execBindOptimizer(&s->train, &s->optim);
    AG_TRY(fillFreeInputs(s, &s->train, s->trainProg));
    AG_TRY(fillFreeInputs(s, &s->eval, s->evalProg));
    batcherInit(&s->batcher, s->data, seed);
    s->step = 0;
    s->fault = {};
    s->stepStart = {};
    // a first batch, so the stepper has something to show before training
    return drawBatch(s);
}

// changing the optimizer or the parameters halfway through a stepped training
// pass leaves no correct state to roll back to, so that pass is cancelled:
// advancing or seeking it now fails with InvalidState until a new begin
static void cancelTrainPass(Session* s) {
    if (s->train.pass == PassKind::Train) s->train.begun = false;
    s->stepStart.valid = false;
}

Status sessionSetOptimizer(Session* s, const OptimConfig& cfg) {
    Optimizer o;
    AG_TRY(optimInit(cfg, s->params, &o));
    s->optim = std::move(o);
    s->cfg.optim = cfg;
    cancelTrainPass(s);
    return {};
}

Status sessionImportParams(Session* s, std::span<const float> in) {
    AG_TRY(importParams(&s->params, in));
    cancelTrainPass(s);
    return {};
}

// ── stepping ─────────────────────────────────────────────────────────────────
// only a Train pass draws a new batch: forward and reverse mode must see the
// same data for their gradients to be comparable

Status sessionBegin(Session* s, PassKind pass, TangentSeed seed) {
    takeSnapshot(s);
    if (pass == PassKind::Train) AG_TRY(drawBatch(s));
    return execBegin(&s->train, pass, seed);
}

Status sessionAdvance(Session* s, int32_t n, int32_t* executed) {
    int32_t len = execLength(s->train), before = s->train.cursor;
    Status st = execAdvance(&s->train, n, executed);
    if (s->train.pass == PassKind::Train) {
        if (st.code == Code::NonFinite) {
            // a nan never reaches the parameters: undo this step's updates.
            // only for a nan: any other error (a pass that was never begun)
            // changed nothing and must not roll anything back
            restoreSnapshot(s, false);
            s->fault = st;
        } else if (before < len && s->train.cursor == len) {
            ++s->step;
            s->fault = {};
        }
    }
    return st;
}

Status sessionSeek(Session* s, int32_t index) {
    if (s->train.pass != PassKind::Train) return execSeek(&s->train, index);
    if (!s->train.begun || !s->stepStart.valid)
        return makeError(Code::InvalidState, "begin a training step before seeking in it");
    if (index < 0 || index > execLength(s->train)) return makeError(Code::InvalidArgument, "seek index out of range");
    // the batch stays bound in the arena; only parameters and optimizer state
    // need to go back to where the step began
    restoreSnapshot(s, false);
    AG_TRY(execBegin(&s->train, PassKind::Train, s->train.seed));
    return sessionAdvance(s, index, nullptr);
}

// ── training ─────────────────────────────────────────────────────────────────
// each step is a transaction: snapshot, draw, run. on a fault everything,
// including the batch position, goes back to the snapshot

Status sessionTrain(Session* s, int32_t nSteps, TrainStats* out) {
    TrainStats st;
    for (int32_t i = 0; i < nSteps; ++i) {
        float lr = lrAt(s->optim.cfg.lr, s->optim.t);
        takeSnapshot(s);
        AG_TRY(drawBatch(s));
        AG_TRY(execBegin(&s->train, PassKind::Train));
        Status r = execRun(&s->train);
        if (!r.ok()) {
            restoreSnapshot(s, true);
            s->fault = r;
            st.step = s->step;
            if (out) *out = st;
            return r;
        }
        ++s->step;
        ++st.stepsRun;
        st.loss = execLoss(s->train);
        st.gradNorm = execGradNorm(s->train);
        st.lr = lr;
    }
    s->fault = {};
    st.step = s->step;
    if (out) *out = st;
    return {};
}

// ── reading ──────────────────────────────────────────────────────────────────

Status sessionRead(const Session& s, NodeId node, Which which, span<float> out) {
    return execRead(s.train, node, which, out);
}

Status sessionStats(const Session& s, NodeId node, Which which, Stats* out) {
    return execStats(s.train, node, which, out);
}

// run eval forward over `count` rows starting at `first`, B at a time. the
// last chunk is filled by cycling rows and weighted by its real size
static Status evalRows(Session* s, int32_t first, int32_t count, float* out) {
    int32_t B = s->cfg.evalBatchSize;
    const Dataset& d = s->data;
    vector<float> x(size_t(B) * d.xDim), y(size_t(B) * d.yDim);
    double acc = 0;
    for (int32_t c = 0; c * B < count; ++c) {
        for (int32_t r = 0; r < B; ++r) {
            int32_t row = first + (c * B + r) % count;
            copy_n(d.x.begin() + size_t(row) * d.xDim, d.xDim, x.begin() + size_t(r) * d.xDim);
            copy_n(d.y.begin() + size_t(row) * d.yDim, d.yDim, y.begin() + size_t(r) * d.yDim);
        }
        AG_TRY(bindRows(s, &s->eval, x, y));
        AG_TRY(execBegin(&s->eval, PassKind::Forward));
        AG_TRY(execRun(&s->eval));
        acc += double(execLoss(s->eval)) * min(B, count - c * B);
    }
    *out = toFloat(acc / count);
    return {};
}

Status sessionEvalLoss(Session* s, Split split, float* out) {
    if (!out) return makeError(Code::InvalidArgument, "sessionEvalLoss: null output");
    if (s->trainProg.graph.loss == kNoNode) return makeError(Code::InvalidState, "the graph has no loss");
    if (s->data.n == 0) {
        AG_TRY(execBegin(&s->eval, PassKind::Forward));
        AG_TRY(execRun(&s->eval));
        *out = execLoss(s->eval);
        return {};
    }
    int32_t first = split == Split::Train ? 0 : s->data.nTrain;
    int32_t count = split == Split::Train ? s->data.nTrain : s->data.n - s->data.nTrain;
    if (count == 0) return makeError(Code::InvalidState, "the dataset has no validation rows");
    return evalRows(s, first, count, out);
}

static Status checkGrid(const GridSpec& g) {
    if (g.res < 2 || g.res > 256) return makeError(Code::InvalidArgument, "grid resolution must be between 2 and 256");
    if (!(g.x1 > g.x0) || !(g.y1 > g.y0) || !isfinite(g.x0) || !isfinite(g.x1) || !isfinite(g.y0) || !isfinite(g.y1))
        return makeError(Code::InvalidArgument, "grid bounds must be finite and increasing");
    return {};
}

Status sessionEvalGrid(Session* s, const GridSpec& grid, span<float> out) {
    AG_TRY(checkGrid(grid));
    const Program& p = s->evalProg;
    NodeId fx = findRole(p.graph, Role::Features);
    if (fx == kNoNode) return makeError(Code::InvalidState, "the graph has no features input to sweep");
    int32_t B = p.batchSize, dim = p.nodes[fx].shape.dim[1];
    if (p.nodes[fx].shape.rank != 2 || dim < 1 || dim > 2)
        return makeError(Code::InvalidState, "the decision boundary needs 1 or 2 input features", fx);

    NodeId node = grid.node != kNoNode ? grid.node : (p.graph.outputs.empty() ? kNoNode : p.graph.outputs[0]);
    if (node < 0 || node >= static_cast<NodeId>(p.nodes.size()))
        return makeError(Code::InvalidArgument, "no node to sample; mark an output", node);
    const Shape& ns = p.nodes[node].shape;
    if (ns.rank < 1 || ns.dim[0] != B) return makeError(Code::InvalidArgument, "the sampled node has no batch axis", node);
    int64_t rowElems = numel(ns) / B;
    if (grid.element < 0 || grid.element >= rowElems) return makeError(Code::InvalidArgument, "element out of range", node);

    int32_t count = dim == 2 ? grid.res * grid.res : grid.res;
    if (static_cast<int32_t>(out.size()) < count) return makeError(Code::InvalidArgument, "grid buffer too small");

    vector<float> x(size_t(B) * dim);
    float dx = (grid.x1 - grid.x0) / grid.res, dy = (grid.y1 - grid.y0) / grid.res;
    for (int32_t c = 0; c * B < count; ++c) {
        for (int32_t r = 0; r < B; ++r) {
            int32_t pt = min(c * B + r, count - 1);
            if (dim == 2) {
                x[r * 2] = grid.x0 + (float(pt % grid.res) + 0.5f) * dx;
                x[r * 2 + 1] = grid.y1 - (float(pt / grid.res) + 0.5f) * dy;
            } else {
                x[r] = grid.x0 + (float(pt) + 0.5f) * dx;
            }
        }
        AG_TRY(execBindInput(&s->eval, fx, x));
        AG_TRY(execBegin(&s->eval, PassKind::Forward));
        AG_TRY(execRun(&s->eval));
        const float* v = s->eval.arena.data() + p.nodes[node].valueOff;
        for (int32_t r = 0; r < B && c * B + r < count; ++r) out[c * B + r] = v[r * rowElems + grid.element];
    }
    return {};
}

// flat index into ParamStore::values of one parameter element, or -1
static int64_t paramElement(const Program& p, NodeId param, int32_t elem) {
    if (param < 0 || param >= static_cast<NodeId>(p.nodes.size()) || p.graph.nodes[param].op != Op::Param) return -1;
    const ParamPlan& pp = p.params[p.nodes[param].paramIndex];
    if (elem < 0 || uint32_t(elem) >= pp.size) return -1;
    return int64_t(pp.offset) + elem;
}

Status sessionLossSurface(Session* s, NodeId paramX, int32_t elemX, NodeId paramY, int32_t elemY, const GridSpec& grid,
                          span<float> out) {
    AG_TRY(checkGrid(grid));
    const Program& p = s->evalProg;
    if (p.graph.loss == kNoNode) return makeError(Code::InvalidState, "the graph has no loss");
    int64_t ix = paramElement(p, paramX, elemX), iy = paramElement(p, paramY, elemY);
    if (ix < 0) return makeError(Code::InvalidArgument, "x is not an element of a parameter", paramX);
    if (iy < 0) return makeError(Code::InvalidArgument, "y is not an element of a parameter", paramY);
    if (ix == iy) return makeError(Code::InvalidArgument, "pick two different parameter elements", paramY);
    if (static_cast<int32_t>(out.size()) < grid.res * grid.res) return makeError(Code::InvalidArgument, "grid buffer too small");

    // a fixed batch: the first eval-batch of training rows
    if (s->data.n > 0) {
        int32_t B = p.batchSize;
        const Dataset& d = s->data;
        vector<float> x(size_t(B) * d.xDim), y(size_t(B) * d.yDim);
        for (int32_t r = 0; r < B; ++r) {
            int32_t row = r % d.nTrain;
            copy_n(d.x.begin() + size_t(row) * d.xDim, d.xDim, x.begin() + size_t(r) * d.xDim);
            copy_n(d.y.begin() + size_t(row) * d.yDim, d.yDim, y.begin() + size_t(r) * d.yDim);
        }
        AG_TRY(bindRows(s, &s->eval, x, y));
    }

    float& wx = s->params.values[ix];
    float& wy = s->params.values[iy];
    float keepX = wx, keepY = wy;
    float dx = (grid.x1 - grid.x0) / grid.res, dy = (grid.y1 - grid.y0) / grid.res;
    for (int32_t i = 0; i < grid.res * grid.res; ++i) {
        wx = grid.x0 + (float(i % grid.res) + 0.5f) * dx;
        wy = grid.y1 - (float(i / grid.res) + 0.5f) * dy;
        Status st = execBegin(&s->eval, PassKind::Forward);
        if (st.ok()) st = execRun(&s->eval);
        // a point where the loss overflows is a gap in the picture, not an error
        out[i] = st.ok() ? execLoss(s->eval) : NAN;
    }
    wx = keepX;
    wy = keepY;
    return {};
}

Status sessionReadOptim(const Session& s, NodeId param, int32_t which, span<float> out) {
    int64_t first = paramElement(s.trainProg, param, 0);
    if (first < 0) return makeError(Code::InvalidArgument, "not a parameter", param);
    if (which < 0 || which > 1) return makeError(Code::InvalidArgument, "optimizer state is 0 (first moment) or 1 (second)");
    const ParamPlan& pp = s.trainProg.params[s.trainProg.nodes[param].paramIndex];
    if (out.size() < pp.size) return makeError(Code::InvalidArgument, "read buffer too small", param);
    const vector<float>& buf = which == 0 ? s.optim.m : s.optim.v;
    copy_n(buf.begin() + first, pp.size, out.begin());
    return {};
}

// ── scalar zoom ──────────────────────────────────────────────────────────────

Status sessionScalarize(const Session& s, unique_ptr<Session>* out, ScalarMap* map) {
    if (!out || !map) return makeError(Code::InvalidArgument, "sessionScalarize: null output");
    Graph g;
    ScalarMap m;
    AG_TRY(scalarize(s.trainProg, s.cfg.limits, &g, &m));

    // leaves take the values the tensor session holds right now
    for (size_t i = 0; i < g.nodes.size(); ++i) {
        Node& node = g.nodes[i];
        NodeId src = m.source[i];
        int32_t e = m.element[i];
        Op srcOp = s.trainProg.graph.nodes[src].op;
        if (e < 0 || (srcOp != Op::Input && srcOp != Op::Param)) continue;
        const NodePlan& np = s.trainProg.nodes[src];
        float value = srcOp == Op::Param ? s.params.values[s.trainProg.params[np.paramIndex].offset + e]
                                         : s.train.arena[np.valueOff + e];
        node.attrs.scalar = value;
        node.attrs.initScale = value;
    }

    SessionConfig cfg = s.cfg;
    cfg.batchSize = 1;
    cfg.evalBatchSize = 1;
    cfg.withTangents = true;
    cfg.limits.maxNodes = max(cfg.limits.maxNodes, cfg.limits.maxScalarNodes);
    cfg.data = DatasetConfig{};
    cfg.data.kind = DatasetKind::None;
    unique_ptr<Session> scalar;
    AG_TRY(sessionCreate(g, cfg, &scalar));
    *out = std::move(scalar);
    *map = std::move(m);
    return {};
}

} // namespace ag
