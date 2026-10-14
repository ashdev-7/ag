#include "ag/capi.h"

#include "ag/ops.hpp"
#include "ag/session.hpp"

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

using namespace std;
using namespace ag;

// ── last error ───────────────────────────────────────────────────────────────
// the wasm build is single threaded, so one slot is enough

static Status g_last;

static ag_status report(Status s) {
    g_last = std::move(s);
    return static_cast<ag_status>(g_last.code);
}

static ag_status bad(const char* fn, const string& why) {
    return report(makeError(Code::InvalidArgument, string(fn) + ": " + why));
}

// ── handle tables ────────────────────────────────────────────────────────────
// handle = generation << 16 | (slot + 1). a freed slot bumps its generation,
// so an old handle no longer matches and returns NotFound

template <class T>
struct Slot {
    unique_ptr<T> object;
    uint16_t      generation = 1;
};

static vector<Slot<Graph>>   g_graphs;
static vector<Slot<Session>> g_sessions;

// the slot number has 16 bits in a handle; past that, handles would alias
static constexpr size_t kMaxSlots = 0xffff;

// returns 0 (never a valid handle) when the table is full
template <class T>
static ag_handle acquire(vector<Slot<T>>& table, unique_ptr<T> object) {
    size_t i = 0;
    while (i < table.size() && table[i].object) ++i;
    if (i == kMaxSlots) return 0;
    if (i == table.size()) table.push_back({});
    table[i].object = std::move(object);
    return (uint32_t(table[i].generation) << 16) | uint32_t(i + 1);
}

template <class T>
static T* lookup(vector<Slot<T>>& table, ag_handle h) {
    uint32_t slot = h & 0xffffu, gen = h >> 16;
    if (slot == 0 || slot > table.size()) return nullptr;
    Slot<T>& s = table[slot - 1];
    return s.object && s.generation == gen ? s.object.get() : nullptr;
}

template <class T>
static bool release(vector<Slot<T>>& table, ag_handle h) {
    if (!lookup(table, h)) return false;
    Slot<T>& s = table[(h & 0xffffu) - 1];
    s.object.reset();
    s.generation = s.generation == 0xffff ? 1 : s.generation + 1;
    return true;
}

#define AG_GRAPH(g, h)                                                                     \
    Graph* g = lookup(g_graphs, h);                                                        \
    if (!g) return report(makeError(Code::NotFound, string(__func__) + ": no such graph"))

#define AG_SESSION(s, h)                                                                   \
    Session* s = lookup(g_sessions, h);                                                    \
    if (!s) return report(makeError(Code::NotFound, string(__func__) + ": no such session"))

#define AG_OUT(p) \
    if (!(p)) return bad(__func__, "null output")

// ── conversions ──────────────────────────────────────────────────────────────

static bool toShape(const ag_shape& in, Shape* out) {
    if (in.rank < 0 || in.rank > kMaxRank) return false;
    out->rank = in.rank;
    for (int i = 0; i < kMaxRank; ++i) out->dim[i] = i < in.rank ? in.dim[i] : 0;
    return true;
}

static ag_shape fromShape(const Shape& s) {
    ag_shape out{};
    out.rank = s.rank;
    for (int i = 0; i < s.rank; ++i) out.dim[i] = s.dim[i];
    return out;
}

static OptimConfig toOptim(const ag_optim_config& c) {
    OptimConfig o;
    o.kind = static_cast<OptimKind>(c.kind < 0 ? 255 : c.kind);
    o.lr.kind = static_cast<LrKind>(c.lr_kind < 0 ? 255 : c.lr_kind);
    o.lr.baseLr = c.base_lr;
    o.lr.minLr = c.min_lr;
    o.lr.gamma = c.gamma;
    o.lr.period = c.period;
    o.lr.warmupSteps = c.warmup_steps;
    o.lr.totalSteps = c.total_steps;
    o.momentum = c.momentum;
    o.beta1 = c.beta1;
    o.beta2 = c.beta2;
    o.eps = c.eps;
    o.weightDecay = c.weight_decay;
    o.clipNorm = c.clip_norm;
    return o;
}

static void toStats(const Stats& s, ag_stats* out) {
    out->min = s.min;
    out->max = s.max;
    out->mean = s.mean;
    out->std = s.std;
    out->l2 = s.l2;
    out->frac_zero = s.fracZero;
    out->non_finite = s.nonFinite;
    out->hist_lo = s.histLo;
    out->hist_hi = s.histHi;
    for (int i = 0; i < AG_HIST_BINS; ++i) out->hist[i] = s.hist[i];
}

static bool toGrid(const ag_grid& g, GridSpec* out) {
    out->x0 = g.x0;
    out->x1 = g.x1;
    out->y0 = g.y0;
    out->y1 = g.y1;
    out->res = g.res;
    out->node = g.node;
    out->element = g.element;
    return true;
}

static bool validPass(int32_t pass) { return pass >= 0 && pass < static_cast<int32_t>(PassKind::Count_); }
static bool validWhich(int32_t w) { return w >= 0 && w <= 2; }

extern "C" {

// ── errors and metadata ──────────────────────────────────────────────────────

uint32_t    ag_abi_version(void) { return AG_ABI_VERSION; }
const char* ag_last_error(void) { return g_last.message.c_str(); }
int32_t     ag_last_error_node(void) { return g_last.node; }
int32_t     ag_last_error_slot(void) { return g_last.slot; }

int32_t ag_op_count(void) { return kOpCount; }

const char* ag_op_name(int32_t op) {
    return op >= 0 && op < kOpCount ? opInfo(static_cast<Op>(op)).name : nullptr;
}

const char* ag_op_symbol(int32_t op) {
    return op >= 0 && op < kOpCount ? opInfo(static_cast<Op>(op)).symbol : nullptr;
}

ag_status ag_op_describe(int32_t op, ag_op_desc* out) {
    if (!out || op < 0 || op >= kOpCount)
        return report(makeError(Code::InvalidArgument, "ag_op_describe: bad op or null out"));
    const OpInfo& info = opInfo(static_cast<Op>(op));
    out->category = static_cast<int32_t>(info.category);
    out->min_inputs = info.minInputs;
    out->max_inputs = info.maxInputs;
    out->saves = info.saves;
    out->has_vjp = info.vjp != nullptr;
    out->has_jvp = info.jvp != nullptr;
    out->has_scalarize = info.scalarize != nullptr;
    return report({});
}

void* ag_alloc(uint32_t bytes) { return malloc(bytes ? bytes : 1); }
void  ag_free(void* p) { free(p); }

// ── graph ────────────────────────────────────────────────────────────────────

ag_status ag_graph_new(ag_handle* out) {
    AG_OUT(out);
    *out = acquire(g_graphs, make_unique<Graph>());
    if (!*out) return report(makeError(Code::LimitExceeded, "too many graphs are open; free some"));
    return report({});
}

ag_status ag_graph_free(ag_handle h) {
    if (!release(g_graphs, h)) return report(makeError(Code::NotFound, "ag_graph_free: no such graph"));
    return report({});
}

ag_status ag_graph_add_node(ag_handle h, int32_t op, const int32_t* inputs, int32_t n_inputs, const ag_attrs* attrs,
                            const char* name, const char* module, int32_t* out_id) {
    AG_GRAPH(g, h);
    AG_OUT(out_id);
    if (op < 0 || op >= kOpCount) return bad(__func__, "unknown op " + to_string(op));
    // compile would refuse a bigger graph anyway; stopping here keeps a
    // runaway caller from filling the fixed heap first
    if (g->nodes.size() >= Limits{}.maxNodes)
        return report(makeError(Code::LimitExceeded, "a graph can have at most " + to_string(Limits{}.maxNodes) + " nodes"));
    if (n_inputs < 0 || n_inputs > kMaxInputs || (n_inputs > 0 && !inputs)) return bad(__func__, "bad input list");
    Attrs a;
    if (attrs) {
        if (!toShape(attrs->shape, &a.shape)) return bad(__func__, "bad shape rank");
        if (attrs->role < 0 || attrs->role > static_cast<int32_t>(Role::Targets)) return bad(__func__, "bad role");
        if (attrs->init < 0 || attrs->init > static_cast<int32_t>(Init::HeNormal)) return bad(__func__, "bad init");
        a.axis = attrs->axis;
        a.keepDim = attrs->keep_dim != 0;
        a.scalar = attrs->scalar;
        a.role = static_cast<Role>(attrs->role);
        a.init = static_cast<Init>(attrs->init);
        a.initScale = attrs->init_scale;
        a.requiresGrad = attrs->requires_grad != 0;
    }
    NodeId id;
    Status s = addNode(g, static_cast<Op>(op), span<const NodeId>(inputs, size_t(n_inputs)), a, name ? name : "",
                       module ? module : "", &id);
    if (s.ok()) *out_id = id;
    return report(std::move(s));
}

ag_status ag_graph_set_loss(ag_handle h, int32_t node) {
    AG_GRAPH(g, h);
    return report(setLoss(g, node));
}

ag_status ag_graph_add_output(ag_handle h, int32_t node) {
    AG_GRAPH(g, h);
    return report(addOutput(g, node));
}

// ── session ──────────────────────────────────────────────────────────────────

static size_t liveSessions() {
    size_t n = 0;
    for (const auto& s : g_sessions) n += s.object != nullptr;
    return n;
}

ag_status ag_session_new(ag_handle gh, const ag_session_config* c, ag_handle* out) {
    AG_GRAPH(g, gh);
    AG_OUT(out);
    if (!c) return bad(__func__, "null config");
    SessionConfig cfg;
    if (liveSessions() >= cfg.limits.maxSessions)
        return report(makeError(Code::LimitExceeded, "at most " + to_string(cfg.limits.maxSessions) + " sessions at once"));
    if (c->dataset_kind < 0 || c->dataset_kind > static_cast<int32_t>(DatasetKind::None)) return bad(__func__, "bad dataset");
    cfg.batchSize = c->batch_size;
    cfg.evalBatchSize = c->eval_batch_size;
    cfg.seed = (uint64_t(c->seed_hi) << 32) | c->seed_lo;
    cfg.withTangents = c->with_tangents != 0;
    cfg.optim = toOptim(c->optim);
    cfg.data.kind = static_cast<DatasetKind>(c->dataset_kind);
    cfg.data.n = c->dataset_n;
    cfg.data.noise = c->dataset_noise;
    cfg.data.valFraction = c->dataset_val_fraction;
    cfg.data.seed = cfg.seed;
    unique_ptr<Session> s;
    Status st = sessionCreate(*g, cfg, &s);
    if (!st.ok()) return report(std::move(st));
    *out = acquire(g_sessions, std::move(s));
    if (!*out) return report(makeError(Code::LimitExceeded, "too many sessions"));
    return report({});
}

ag_status ag_session_free(ag_handle h) {
    if (!release(g_sessions, h)) return report(makeError(Code::NotFound, "ag_session_free: no such session"));
    return report({});
}

ag_status ag_session_reset(ag_handle h, uint32_t seed_lo, uint32_t seed_hi) {
    AG_SESSION(s, h);
    return report(sessionReset(s, (uint64_t(seed_hi) << 32) | seed_lo));
}

ag_status ag_session_set_optimizer(ag_handle h, const ag_optim_config* c) {
    AG_SESSION(s, h);
    if (!c) return bad(__func__, "null config");
    return report(sessionSetOptimizer(s, toOptim(*c)));
}

ag_status ag_session_node_count(ag_handle h, int32_t* out) {
    AG_SESSION(s, h);
    AG_OUT(out);
    *out = static_cast<int32_t>(s->trainProg.nodes.size());
    return report({});
}

ag_status ag_session_node_shape(ag_handle h, int32_t node, ag_shape* out, int32_t* requires_grad) {
    AG_SESSION(s, h);
    AG_OUT(out);
    if (node < 0 || node >= static_cast<int32_t>(s->trainProg.nodes.size())) return bad(__func__, "no such node");
    *out = fromShape(s->trainProg.nodes[node].shape);
    if (requires_grad) *requires_grad = s->trainProg.nodes[node].requiresGrad;
    return report({});
}

ag_status ag_session_node_op(ag_handle h, int32_t node, int32_t* op, int32_t* inputs, int32_t* n_inputs) {
    AG_SESSION(s, h);
    AG_OUT(op && inputs && n_inputs);
    if (node < 0 || node >= static_cast<int32_t>(s->trainProg.nodes.size())) return bad(__func__, "no such node");
    const Node& n = s->trainProg.graph.nodes[node];
    *op = static_cast<int32_t>(n.op);
    *n_inputs = n.numInputs;
    for (int i = 0; i < kMaxInputs; ++i) inputs[i] = n.inputs[i];
    return report({});
}

const char* ag_session_node_name(ag_handle h, int32_t node) {
    Session* s = lookup(g_sessions, h);
    if (!s || node < 0 || node >= static_cast<int32_t>(s->trainProg.nodes.size())) return nullptr;
    return s->trainProg.graph.nodes[node].name.c_str();
}

ag_status ag_session_schedule_len(ag_handle h, int32_t pass, int32_t* out) {
    AG_SESSION(s, h);
    AG_OUT(out);
    if (!validPass(pass)) return bad(__func__, "bad pass");
    *out = static_cast<int32_t>(schedule(s->trainProg, static_cast<PassKind>(pass)).size());
    return report({});
}

ag_status ag_session_schedule(ag_handle h, int32_t pass, ag_step* out, int32_t cap) {
    AG_SESSION(s, h);
    AG_OUT(out);
    if (!validPass(pass)) return bad(__func__, "bad pass");
    auto steps = schedule(s->trainProg, static_cast<PassKind>(pass));
    if (cap < static_cast<int32_t>(steps.size())) return bad(__func__, "buffer too small");
    for (size_t i = 0; i < steps.size(); ++i)
        out[i] = {static_cast<int32_t>(steps[i].kind), steps[i].slot, steps[i].node};
    return report({});
}

ag_status ag_session_begin(ag_handle h, int32_t pass, int32_t seed_node, int32_t seed_element) {
    AG_SESSION(s, h);
    if (!validPass(pass)) return bad(__func__, "bad pass");
    return report(sessionBegin(s, static_cast<PassKind>(pass), TangentSeed{seed_node, seed_element}));
}

ag_status ag_session_advance(ag_handle h, int32_t n, int32_t* executed) {
    AG_SESSION(s, h);
    if (n < 0) return bad(__func__, "n cannot be negative");
    return report(sessionAdvance(s, n, executed));
}

ag_status ag_session_seek(ag_handle h, int32_t index) {
    AG_SESSION(s, h);
    return report(sessionSeek(s, index));
}

ag_status ag_session_cursor(ag_handle h, int32_t* out) {
    AG_SESSION(s, h);
    AG_OUT(out);
    *out = s->train.cursor;
    return report({});
}

ag_status ag_session_train(ag_handle h, int32_t n_steps, ag_train_stats* out) {
    AG_SESSION(s, h);
    if (n_steps < 0) return bad(__func__, "n_steps cannot be negative");
    TrainStats st;
    Status r = sessionTrain(s, n_steps, &st);
    if (out) {
        out->step_lo = int32_t(st.step & 0xffffffff);
        out->step_hi = int32_t(st.step >> 32);
        out->steps_run = st.stepsRun;
        out->loss = st.loss;
        out->grad_norm = st.gradNorm;
        out->lr = st.lr;
    }
    return report(std::move(r));
}

ag_status ag_session_read(ag_handle h, int32_t node, int32_t which, float* out, int32_t cap, int32_t* written) {
    AG_SESSION(s, h);
    AG_OUT(out);
    if (!validWhich(which)) return bad(__func__, "bad buffer kind");
    if (node < 0 || node >= static_cast<int32_t>(s->trainProg.nodes.size())) return bad(__func__, "no such node");
    int64_t n = numel(s->trainProg.nodes[node].shape);
    if (cap < n) return bad(__func__, "buffer too small");
    Status st = sessionRead(*s, node, static_cast<Which>(which), span<float>(out, size_t(cap)));
    if (st.ok() && written) *written = static_cast<int32_t>(n);
    return report(std::move(st));
}

ag_status ag_session_stats(ag_handle h, int32_t node, int32_t which, ag_stats* out) {
    AG_SESSION(s, h);
    AG_OUT(out);
    if (!validWhich(which)) return bad(__func__, "bad buffer kind");
    if (node < 0 || node >= static_cast<int32_t>(s->trainProg.nodes.size())) return bad(__func__, "no such node");
    Stats st;
    Status r = sessionStats(*s, node, static_cast<Which>(which), &st);
    if (r.ok()) toStats(st, out);
    return report(std::move(r));
}

ag_status ag_session_eval_loss(ag_handle h, int32_t split, float* out) {
    AG_SESSION(s, h);
    AG_OUT(out);
    if (split < 0 || split > 1) return bad(__func__, "bad split");
    return report(sessionEvalLoss(s, static_cast<Split>(split), out));
}

ag_status ag_session_eval_grid(ag_handle h, const ag_grid* grid, float* out, int32_t cap) {
    AG_SESSION(s, h);
    AG_OUT(out);
    if (!grid || cap < 0) return bad(__func__, "null grid");
    GridSpec g;
    toGrid(*grid, &g);
    return report(sessionEvalGrid(s, g, span<float>(out, size_t(cap))));
}

ag_status ag_session_loss_surface(ag_handle h, int32_t param_x, int32_t elem_x, int32_t param_y, int32_t elem_y,
                                  const ag_grid* grid, float* out, int32_t cap) {
    AG_SESSION(s, h);
    AG_OUT(out);
    if (!grid || cap < 0) return bad(__func__, "null grid");
    GridSpec g;
    toGrid(*grid, &g);
    return report(sessionLossSurface(s, param_x, elem_x, param_y, elem_y, g, span<float>(out, size_t(cap))));
}

ag_status ag_session_dataset(ag_handle h, int32_t* dims, float* x, int32_t x_cap, float* y, int32_t y_cap) {
    AG_SESSION(s, h);
    AG_OUT(dims);
    const Dataset& d = s->data;
    dims[0] = d.n;
    dims[1] = d.nTrain;
    dims[2] = d.xDim;
    dims[3] = d.yDim;
    if (x) {
        if (x_cap < static_cast<int32_t>(d.x.size())) return bad(__func__, "x buffer too small");
        copy(d.x.begin(), d.x.end(), x);
    }
    if (y) {
        if (y_cap < static_cast<int32_t>(d.y.size())) return bad(__func__, "y buffer too small");
        copy(d.y.begin(), d.y.end(), y);
    }
    return report({});
}

ag_status ag_session_read_optim(ag_handle h, int32_t param, int32_t which, float* out, int32_t cap) {
    AG_SESSION(s, h);
    AG_OUT(out);
    if (cap < 0) return bad(__func__, "negative capacity");
    return report(sessionReadOptim(*s, param, which, span<float>(out, size_t(cap))));
}

ag_status ag_session_param_count(ag_handle h, int32_t* out) {
    AG_SESSION(s, h);
    AG_OUT(out);
    *out = static_cast<int32_t>(s->params.values.size());
    return report({});
}

ag_status ag_session_export_params(ag_handle h, float* out, int32_t cap) {
    AG_SESSION(s, h);
    AG_OUT(out);
    if (cap < static_cast<int32_t>(s->params.values.size())) return bad(__func__, "buffer too small");
    return report(exportParams(s->params, span<float>(out, s->params.values.size())));
}

ag_status ag_session_import_params(ag_handle h, const float* in, int32_t n) {
    AG_SESSION(s, h);
    if (!in && n > 0) return bad(__func__, "null input");
    return report(sessionImportParams(s, span<const float>(in, size_t(max(n, 0)))));
}

ag_status ag_session_scalarize(ag_handle h, ag_handle* out, int32_t* map_source, int32_t* map_element, int32_t cap) {
    AG_SESSION(s, h);
    AG_OUT(out);
    if (liveSessions() >= s->cfg.limits.maxSessions)
        return report(makeError(Code::LimitExceeded, "at most " + to_string(s->cfg.limits.maxSessions) + " sessions at once"));
    unique_ptr<Session> scalar;
    ScalarMap map;
    Status st = sessionScalarize(*s, &scalar, &map);
    if (!st.ok()) return report(std::move(st));
    int32_t count = static_cast<int32_t>(map.source.size());
    if ((map_source || map_element) && cap < count) return bad(__func__, "map buffers need " + to_string(count) + " entries");
    for (int32_t i = 0; i < count; ++i) {
        if (map_source) map_source[i] = map.source[i];
        if (map_element) map_element[i] = map.element[i];
    }
    *out = acquire(g_sessions, std::move(scalar));
    if (!*out) return report(makeError(Code::LimitExceeded, "too many sessions"));
    return report({});
}

} // extern "C"
