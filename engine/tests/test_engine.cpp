// the engine's main tests. exits non-zero if any check fails.
//
//   ag_test_engine               run everything
//   ag_test_engine --dump FILE   also write every op case and an optimizer
//                                trajectory as JSON for tests/torch_check.py

#include "ag/capi.h"
#include "ag/session.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;
using namespace ag;

static int g_failed = 0, g_passed = 0;
static ostringstream g_dump;   // JSON cases for the PyTorch comparison
static bool g_firstCase = true;
static vector<float> g_moons;  // batch loss every 250 steps, for the wasm comparison

static void check(bool condition, const string& label) {
    if (condition) ++g_passed;
    else ++g_failed;
    if (!condition) cout << "  FAIL  " << label << "\n";
    else cout << "  PASS  " << label << "\n";
}

static bool near(double a, double b, double tol) { return fabs(a - b) <= tol * max(1.0, max(fabs(a), fabs(b))); }

// ── a tiny graph builder ─────────────────────────────────────────────────────

struct Builder {
    Graph g;

    NodeId add(Op op, initializer_list<NodeId> in, Attrs a = {}, const string& name = "") {
        NodeId id;
        Status s = addNode(&g, op, span<const NodeId>(in.begin(), in.size()), a, name.empty() ? opInfo(op).name : name, "", &id);
        if (!s.ok()) {
            cout << "  build error: " << s.message << "\n";
            exit(2);
        }
        return id;
    }
    static Attrs shaped(initializer_list<int32_t> dims, Init init = Init::Normal, float scale = 1.0f) {
        Attrs a;
        a.shape = makeShape(span<const int32_t>(dims.begin(), dims.size()));
        a.init = init;
        a.initScale = scale;
        return a;
    }
    NodeId param(initializer_list<int32_t> dims, const string& name, Init init = Init::Normal, float scale = 1.0f) {
        return add(Op::Param, {}, shaped(dims, init, scale), name);
    }
    // a fixed input that does not need a gradient, filled from its init rule
    NodeId fixed(initializer_list<int32_t> dims, const string& name, Init init = Init::Normal, float scale = 1.0f) {
        return add(Op::Input, {}, shaped(dims, init, scale), name);
    }
    NodeId op(Op o, initializer_list<NodeId> in, const string& name = "") { return add(o, in, {}, name); }
    NodeId axisOp(Op o, NodeId x, int32_t axis, bool keepDim = false) {
        Attrs a;
        a.axis = axis;
        a.keepDim = keepDim;
        return add(o, {x}, a);
    }
    NodeId scalarOp(Op o, NodeId x, float s) {
        Attrs a;
        a.scalar = s;
        return add(o, {x}, a);
    }
};

static SessionConfig noData(uint64_t seed = 1) {
    SessionConfig c;
    c.seed = seed;
    c.batchSize = 1;
    c.evalBatchSize = 1;
    c.data.kind = DatasetKind::None;
    return c;
}

static vector<float> readAll(const Session& s, NodeId n, Which w) {
    vector<float> v(numel(s.trainProg.nodes[n].shape));
    if (!sessionRead(s, n, w, v).ok()) v.clear();
    return v;
}

// ── 1. the first example ─────────────────────────────────────────────────────
// d = a*b + a at a = 2, b = 3: d = 8, dd/da = b + 1 = 4, dd/db = a = 2

static void testExample() {
    cout << "\n[d = a*b + a]\n";
    Builder b;
    Attrs aa = Builder::shaped({}, Init::Constant, 2.0f), ab = Builder::shaped({}, Init::Constant, 3.0f);
    aa.requiresGrad = ab.requiresGrad = true;
    NodeId a = b.add(Op::Input, {}, aa, "a"), bb = b.add(Op::Input, {}, ab, "b");
    NodeId c = b.op(Op::Mul, {a, bb}, "c"), d = b.op(Op::Add, {c, a}, "d");
    setLoss(&b.g, d);

    unique_ptr<Session> s;
    Status st = sessionCreate(b.g, noData(), &s);
    check(st.ok(), "session created " + st.message);
    if (!st.ok()) return;

    auto rev = schedule(s->trainProg, PassKind::Reverse);
    bool shape = rev.size() == 9 && rev[4].kind == StepKind::SeedGrad && rev[5].kind == StepKind::Vjp && rev[5].node == d &&
                 rev[5].slot == 0 && rev[6].node == d && rev[6].slot == 1 && rev[7].node == c && rev[8].slot == 1;
    check(shape, "reverse schedule is the 9 steps in DESIGN §4.2");
    check(s->trainProg.nodes[a].numConsumers == 2, "a has two consumers");

    sessionBegin(s.get(), PassKind::Reverse);
    int32_t ran = 0;
    sessionAdvance(s.get(), 100, &ran);
    check(ran == 9 && readAll(*s, d, Which::Value)[0] == 8.0f, "d = 8");
    check(readAll(*s, a, Which::Grad)[0] == 4.0f, "dd/da = 4");
    check(readAll(*s, bb, Which::Grad)[0] == 2.0f, "dd/db = 2");

    // stop between the two contributions to a.grad: after step 6 only d -> a
    sessionSeek(s.get(), 7);
    check(readAll(*s, a, Which::Grad)[0] == 1.0f, "after step 7 a.grad holds only the direct contribution (1)");

    // forward mode: one pass per input, each gives one derivative
    sessionBegin(s.get(), PassKind::ForwardMode, {a, 0});
    sessionAdvance(s.get(), 100, nullptr);
    float ta = readAll(*s, d, Which::Tangent)[0];
    sessionBegin(s.get(), PassKind::ForwardMode, {bb, 0});
    sessionAdvance(s.get(), 100, nullptr);
    float tb = readAll(*s, d, Which::Tangent)[0];
    check(ta == 4.0f && tb == 2.0f, "forward mode gives the same 4 and 2");
}

// ── 2. every op: finite differences and vjp against jvp ──────────────────────

static void dumpCase(const string& name, const Session& s) {
    const Program& p = s.trainProg;
    g_dump << (g_firstCase ? "" : ",\n") << "{\"name\":\"" << name << "\",\"nodes\":[";
    g_firstCase = false;
    for (size_t i = 0; i < p.graph.nodes.size(); ++i) {
        const Node& n = p.graph.nodes[i];
        g_dump << (i ? "," : "") << "{\"op\":\"" << opInfo(n.op).name << "\",\"inputs\":[";
        for (int k = 0; k < n.numInputs; ++k) g_dump << (k ? "," : "") << n.inputs[k];
        g_dump << "],\"shape\":[";
        for (int k = 0; k < p.nodes[i].shape.rank; ++k) g_dump << (k ? "," : "") << p.nodes[i].shape.dim[k];
        g_dump << "],\"axis\":" << n.attrs.axis << ",\"keepDim\":" << (n.attrs.keepDim ? "true" : "false")
               << ",\"scalar\":" << n.attrs.scalar << ",\"value\":[";
        vector<float> v = readAll(s, NodeId(i), Which::Value);
        for (size_t k = 0; k < v.size(); ++k) g_dump << (k ? "," : "") << v[k];
        g_dump << "],\"grad\":[";
        if (p.nodes[i].gradOff != kNoBuffer) {
            vector<float> gr = readAll(s, NodeId(i), Which::Grad);
            for (size_t k = 0; k < gr.size(); ++k) g_dump << (k ? "," : "") << gr[k];
        }
        g_dump << "]}";
    }
    g_dump << "],\"loss\":" << p.graph.loss << "}";
}

// the loss is Σ out ⊙ r for a fixed random r, so every output element matters
// with a different weight (a plain sum would hide softmax's gradient)
static void gradCheck(const string& name, Builder& b, NodeId out) {
    Graph& g = b.g;
    NodeId loss = out;
    Program probe;   // the output's shape is only known after compile
    Status st = compile(g, {}, &probe);
    if (!st.ok()) {
        check(false, name + ": compiles (" + st.message + ")");
        return;
    }
    if (numel(probe.nodes[out].shape) != 1) {
        Attrs ra;
        ra.shape = probe.nodes[out].shape;
        ra.init = Init::Normal;
        NodeId r = b.add(Op::Input, {}, ra, "r");
        loss = b.op(Op::Sum, {b.op(Op::Mul, {out, r})}, "loss");
    }
    setLoss(&g, loss);

    unique_ptr<Session> s;
    st = sessionCreate(g, noData(7), &s);
    if (!st.ok()) {
        check(false, name + ": session (" + st.message + ")");
        return;
    }
    sessionBegin(s.get(), PassKind::Reverse);
    st = sessionAdvance(s.get(), 1 << 20, nullptr);
    if (!st.ok()) {
        check(false, name + ": reverse pass (" + st.message + ")");
        return;
    }
    dumpCase(name, *s);

    // analytic grads of every parameter element
    const Program& p = s->trainProg;
    vector<float> grads(p.paramFloats);
    for (const ParamPlan& pp : p.params) {
        vector<float> gp = readAll(*s, pp.node, Which::Grad);
        copy(gp.begin(), gp.end(), grads.begin() + pp.offset);
    }

    auto lossAt = [&](uint32_t i, float delta) {
        float keep = s->params.values[i];
        s->params.values[i] = keep + delta;
        float v;
        sessionEvalLoss(s.get(), Split::Train, &v);
        s->params.values[i] = keep;
        return double(v);
    };

    // finite differences: h relative to float precision, compared loosely
    double worst = 0;
    bool fdOk = true;
    for (uint32_t i = 0; i < p.paramFloats; ++i) {
        float h = 1e-2f;
        double num = (lossAt(i, h) - lossAt(i, -h)) / (2.0 * h);
        double err = fabs(num - grads[i]) / max(1.0, max(fabs(num), fabs(double(grads[i]))));
        worst = max(worst, err);
        fdOk = fdOk && err < 2e-2;
    }
    char buf[64];
    snprintf(buf, sizeof buf, " (worst rel err %.1e)", worst);
    check(fdOk, name + ": gradients match finite differences" + buf);

    // forward mode, one parameter element at a time, must reproduce each grad
    bool jvpOk = true;
    for (const ParamPlan& pp : p.params)
        for (uint32_t e = 0; e < pp.size; ++e) {
            sessionBegin(s.get(), PassKind::ForwardMode, {pp.node, int32_t(e)});
            sessionAdvance(s.get(), 1 << 20, nullptr);
            float t = readAll(*s, loss, Which::Tangent)[0];
            jvpOk = jvpOk && near(t, grads[pp.offset + e], 1e-4);
        }
    check(jvpOk, name + ": forward mode agrees with reverse mode");

    // the single-number graph must build for every op and give the same
    // loss and the same gradients
    unique_ptr<Session> sc;
    ScalarMap map;
    sessionBegin(s.get(), PassKind::Reverse);
    sessionAdvance(s.get(), 1 << 20, nullptr);
    st = sessionScalarize(*s, &sc, &map);
    if (!st.ok()) {
        check(false, name + ": scalar view builds (" + st.message + ")");
        return;
    }
    sessionBegin(sc.get(), PassKind::Reverse);
    st = sessionAdvance(sc.get(), 1 << 20, nullptr);
    bool same = st.ok() && near(execLoss(s->train), execLoss(sc->train), 1e-5);
    for (const ParamPlan& pp : p.params)
        for (uint32_t e = 0; e < pp.size; ++e)
            same = same && near(grads[pp.offset + e], readAll(*sc, map.elems[pp.node][e], Which::Grad)[0], 1e-4);
    check(same, name + ": scalar view gives the same loss and gradients (" +
                    to_string(sc->trainProg.graph.nodes.size()) + " nodes)");
}

static void testOps() {
    cout << "\n[ops: finite differences, vjp vs jvp]\n";
    auto run = [](const string& name, function<NodeId(Builder&)> make) {
        Builder b;
        NodeId out = make(b);
        gradCheck(name, b, out);
    };
    // positive values for log, div and fractional pow come from exp
    auto positive = [](Builder& b, initializer_list<int32_t> dims, const string& n) {
        return b.op(Op::Exp, {b.param(dims, n, Init::Normal, 0.5f)});
    };

    run("add broadcast [3,4]+[4]", [](Builder& b) { return b.op(Op::Add, {b.param({3, 4}, "x"), b.param({4}, "y")}); });
    run("sub with a scalar", [](Builder& b) { return b.op(Op::Sub, {b.param({3, 4}, "x"), b.param({}, "y")}); });
    run("mul broadcast [3,1]*[1,4]", [](Builder& b) { return b.op(Op::Mul, {b.param({3, 1}, "x"), b.param({1, 4}, "y")}); });
    run("div [3,4]/[3,1]", [&](Builder& b) { return b.op(Op::Div, {b.param({3, 4}, "x"), positive(b, {3, 1}, "y")}); });
    run("neg", [](Builder& b) { return b.op(Op::Neg, {b.param({5}, "x")}); });
    run("pow 3", [](Builder& b) { return b.scalarOp(Op::Pow, b.param({5}, "x"), 3.0f); });
    run("pow 0.5", [&](Builder& b) { return b.scalarOp(Op::Pow, positive(b, {5}, "x"), 0.5f); });
    run("exp", [](Builder& b) { return b.op(Op::Exp, {b.param({5}, "x")}); });
    run("log", [&](Builder& b) { return b.op(Op::Log, {positive(b, {5}, "x")}); });
    run("matmul [3,4]@[4,2]", [](Builder& b) { return b.op(Op::MatMul, {b.param({3, 4}, "x"), b.param({4, 2}, "w")}); });
    run("sum all", [](Builder& b) { return b.op(Op::Sum, {b.param({3, 4}, "x")}); });
    run("sum axis 0", [](Builder& b) { return b.axisOp(Op::Sum, b.param({3, 4}, "x"), 0); });
    run("sum axis -1 keepdim", [](Builder& b) { return b.axisOp(Op::Sum, b.param({2, 3, 4}, "x"), -1, true); });
    run("mean axis 1", [](Builder& b) { return b.axisOp(Op::Mean, b.param({3, 4}, "x"), 1); });
    run("mean all", [](Builder& b) { return b.op(Op::Mean, {b.param({3, 4}, "x")}); });
    run("reshape [3,4]->[2,6]", [](Builder& b) {
        Attrs a = Builder::shaped({2, 6});
        return b.add(Op::Reshape, {b.param({3, 4}, "x")}, a);
    });
    run("transpose", [](Builder& b) { return b.op(Op::Transpose, {b.param({3, 4}, "x")}); });
    run("relu", [](Builder& b) { return b.op(Op::Relu, {b.param({8}, "x")}); });
    run("leaky_relu 0.1", [](Builder& b) { return b.scalarOp(Op::LeakyRelu, b.param({8}, "x"), 0.1f); });
    run("tanh", [](Builder& b) { return b.op(Op::Tanh, {b.param({3, 4}, "x")}); });
    run("sigmoid", [](Builder& b) { return b.op(Op::Sigmoid, {b.param({3, 4}, "x")}); });
    run("softmax last axis", [](Builder& b) { return b.op(Op::Softmax, {b.param({3, 4}, "x")}); });
    run("softmax axis 0", [](Builder& b) { return b.axisOp(Op::Softmax, b.param({3, 4}, "x"), 0); });
    run("mse_loss", [](Builder& b) { return b.op(Op::MseLoss, {b.param({3, 2}, "p"), b.param({3, 2}, "t")}); });
    run("softmax_cross_entropy", [](Builder& b) {
        NodeId t = b.op(Op::Softmax, {b.param({3, 4}, "tlogits")});
        return b.op(Op::SoftmaxCrossEntropy, {b.param({3, 4}, "z"), t});
    });
    // one tensor feeding both inputs of an op: two gradient contributions into one buffer
    run("x * x (same input twice)", [](Builder& b) { NodeId x = b.param({3, 2}, "x"); return b.op(Op::Mul, {x, x}); });
    run("matmul x @ xᵀ", [](Builder& b) {
        NodeId x = b.param({3, 2}, "x");
        return b.op(Op::MatMul, {x, b.op(Op::Transpose, {x})});
    });
    run("rank-4 broadcast [2,1,3,1] - [4,1,1]", [](Builder& b) { return b.op(Op::Sub, {b.param({2, 1, 3, 1}, "x"), b.param({4, 1, 1}, "y")}); });
    run("mean axis 1 keepdim of rank 3", [](Builder& b) { return b.axisOp(Op::Mean, b.param({2, 3, 2}, "x"), 1, true); });
    run("reshape to a scalar", [](Builder& b) {
        Attrs a;   // rank 0
        return b.add(Op::Reshape, {b.param({1, 1}, "x")}, a);
    });
    run("two-layer mlp", [](Builder& b) {
        NodeId x = b.fixed({4, 3}, "x");
        NodeId h = b.op(Op::Tanh, {b.op(Op::Add, {b.op(Op::MatMul, {x, b.param({3, 5}, "W1", Init::XavierNormal)}), b.param({5}, "b1")})});
        return b.op(Op::MatMul, {h, b.param({5, 2}, "W2", Init::XavierNormal)});
    });
}

// ── 3. compile errors name the node and the input ────────────────────────────

static void testCompileErrors() {
    cout << "\n[compile errors]\n";
    {
        Builder b;
        NodeId x = b.param({3, 4}, "x"), w = b.param({5, 2}, "w");
        NodeId y = b.op(Op::MatMul, {x, w});
        Program p;
        p.batchSize = 99;
        Status s = compile(b.g, {}, &p);
        check(s.code == Code::ShapeMismatch && s.node == y && s.slot == 1, "matmul [3,4]@[5,2] names the node and input 1");
        check(p.batchSize == 99, "a failed compile leaves the output untouched");
    }
    {
        Builder b;
        NodeId x = b.param({3}, "x");
        setLoss(&b.g, x);
        Program p;
        check(compile(b.g, {}, &p).code == Code::ShapeMismatch, "a loss that is not a single number is rejected");
    }
    {
        Builder b;
        Attrs a = Builder::shaped({kBatchDim, 2});
        b.add(Op::Param, {}, a, "w");
        Program p;
        check(compile(b.g, {}, &p).code == Code::InvalidGraph, "a parameter with a batch dimension is rejected");
    }
    {
        Builder b;
        b.param({300, 300}, "big");
        Program p;
        check(compile(b.g, {}, &p).code == Code::LimitExceeded, "an oversized tensor hits a limit");
    }
    {
        OptimConfig o;
        o.lr.baseLr = -1;
        check(optimValidate(o).code == Code::InvalidArgument, "a negative learning rate is rejected");
    }
}

// ── 3b. hostile inputs: overflow, nan, wrong order, too many ────────────────
// each check here was a real bug, found by an audit and reproduced first

static void testRobustness() {
    cout << "\n[robustness]\n";
    {
        // 65536^4 = 2^64 wrapped to 0 elements and passed the size limit
        Builder b;
        b.param({65536, 65536, 65536, 65536}, "huge");
        Program p;
        check(compile(b.g, {}, &p).code == Code::LimitExceeded, "a shape whose size overflows 64 bits is refused");
        Builder r;
        Attrs a = Builder::shaped({65536, 65536, 65536, 65536});
        r.add(Op::Reshape, {r.param({4}, "x")}, a);
        check(compile(r.g, {}, &p).code == Code::LimitExceeded, "a reshape to such a shape is refused");
    }
    {
        // hi - lo overflowed float, the histogram bin became int(nan)
        float v[3] = {-3e38f, 0.0f, 3e38f};
        Stats st;
        ConstTensorView t{v, Builder::shaped({3}).shape};
        computeStats(t, &st);
        uint32_t total = 0;
        for (uint32_t h : st.hist) total += h;
        check(total == 3 && st.hist[0] == 1 && st.hist[kHistBins - 1] == 1 && isinf(st.l2),
              "statistics of values near the float limit: every value binned, norm inf");
        float big[2] = {3e38f, 3e38f};
        computeStats({big, Builder::shaped({2}).shape}, &st);
        check(st.mean == 3e38f && st.hist[kHistBins / 2] == 2, "a constant tensor near the limit");
    }
    {
        Builder b;
        Attrs a = Builder::shaped({2}, Init::Constant, NAN);
        NodeId p = b.add(Op::Param, {}, a, "p");
        Program prog;
        Status s = compile(b.g, {}, &prog);
        check(s.code == Code::InvalidGraph && s.node == p, "a nan init scale is refused at the node");
        Builder c;
        Attrs inf;
        inf.scalar = INFINITY;
        NodeId q = c.add(Op::Pow, {c.param({2}, "x")}, inf);
        s = compile(c.g, {}, &prog);
        check(s.code == Code::InvalidGraph && s.node == q, "an infinite exponent is refused at the node");
    }
    {
        // a negative uniform scale used to give all zeros
        Builder b;
        b.param({64}, "w", Init::Uniform, -0.5f);
        Program p;
        compile(b.g, {}, &p);
        ParamStore ps;
        initParams(p, 1, &ps);
        bool nonzero = false, bounded = true;
        for (float x : ps.values) nonzero = nonzero || x != 0, bounded = bounded && fabs(x) <= 0.5f;
        check(nonzero && bounded, "a negative uniform scale initialises like its mirror image");
    }
    {
        // changing the optimizer halfway through a stepped training pass
        // left a pass whose rollback point was gone
        Builder b;
        NodeId p = b.param({1}, "p", Init::Constant, 3.0f);
        setLoss(&b.g, b.op(Op::Sum, {b.scalarOp(Op::Pow, p, 4.0f)}));
        SessionConfig c = noData();
        c.optim.lr.baseLr = 0.001f;
        unique_ptr<Session> s;
        sessionCreate(b.g, c, &s);
        sessionBegin(s.get(), PassKind::Train);
        sessionAdvance(s.get(), 3, nullptr);
        sessionSetOptimizer(s.get(), c.optim);
        vector<float> before = s->params.values;
        Status adv = sessionAdvance(s.get(), 100, nullptr);
        Status sk = sessionSeek(s.get(), 1);
        check(adv.code == Code::InvalidState && sk.code == Code::InvalidState && s->params.values == before,
              "a training pass interrupted by an optimizer change cannot be continued or seeked");
        // and an error that is not a nan must not roll anything back
        sessionTrain(s.get(), 5, nullptr);
        vector<float> trained = s->params.values;
        sessionAdvance(s.get(), 1, nullptr);
        check(s->params.values == trained && s->step == 5, "a refused advance leaves trained parameters alone");
        sessionBegin(s.get(), PassKind::Train);
        check(sessionAdvance(s.get(), 100, nullptr).ok() && s->step == 6, "a new pass after that works");
    }
    {
        ag_handle g;
        ag_graph_new(&g);
        ag_attrs a{};
        a.axis = kAllAxes;
        int32_t id, ok = 0;
        for (int i = 0; i < 1000; ++i) ok += ag_graph_add_node(g, int32_t(Op::Const), nullptr, 0, &a, "c", "", &id) == 0;
        check(ok == int32_t(Limits{}.maxNodes) && ag_graph_add_node(g, int32_t(Op::Const), nullptr, 0, &a, "c", "", &id) ==
                                                      int32_t(Code::LimitExceeded),
              "the C API stops a graph at the node limit");
        ag_graph_free(g);
    }
}

// ── 4. a two-layer MLP fits moons ────────────────────────────────────────────

static Graph moonsMlp(int hidden) {
    Builder b;
    Attrs xa = Builder::shaped({kBatchDim, 2}, Init::Zeros), ya = xa;
    xa.role = Role::Features;
    ya.role = Role::Targets;
    NodeId x = b.add(Op::Input, {}, xa, "x"), y = b.add(Op::Input, {}, ya, "y");
    NodeId w1 = b.param({2, hidden}, "W1", Init::XavierNormal), b1 = b.param({hidden}, "b1", Init::Zeros);
    NodeId w2 = b.param({hidden, 2}, "W2", Init::XavierNormal), b2 = b.param({2}, "b2", Init::Zeros);
    NodeId h = b.op(Op::Tanh, {b.op(Op::Add, {b.op(Op::MatMul, {x, w1}), b1})}, "h");
    NodeId logits = b.op(Op::Add, {b.op(Op::MatMul, {h, w2}), b2}, "logits");
    NodeId probs = b.op(Op::Softmax, {logits}, "probs");
    setLoss(&b.g, b.op(Op::SoftmaxCrossEntropy, {logits, y}, "loss"));
    addOutput(&b.g, probs);
    return b.g;
}

static SessionConfig moonsConfig(OptimKind kind, float lr) {
    SessionConfig c;
    c.seed = 42;
    c.batchSize = 32;
    c.evalBatchSize = 128;
    c.data.kind = DatasetKind::Moons;
    c.data.n = 400;
    c.data.noise = 0.1f;
    c.data.seed = 42;
    c.optim.kind = kind;
    c.optim.lr.baseLr = lr;
    return c;
}

static void testMoons() {
    cout << "\n[moons]\n";
    unique_ptr<Session> s;
    Status st = sessionCreate(moonsMlp(16), moonsConfig(OptimKind::Sgd, 0.5f), &s);
    check(st.ok(), "moons session " + st.message);
    if (!st.ok()) return;
    float before, after, val;
    sessionEvalLoss(s.get(), Split::Train, &before);
    TrainStats ts;
    for (int k = 0; k < 6; ++k) {
        st = sessionTrain(s.get(), 250, &ts);
        g_moons.push_back(ts.loss);
        sessionEvalLoss(s.get(), Split::Train, &after);
        printf("  step %5lld  batch loss %.4f  train loss %.4f  grad norm %.3f\n", (long long)ts.step, ts.loss, after, ts.gradNorm);
    }
    sessionEvalLoss(s.get(), Split::Val, &val);
    printf("  val loss %.4f (started at %.4f)\n", val, before);
    check(st.ok() && after < 0.1f && val < 0.15f, "SGD fits moons: train loss < 0.1, val loss < 0.15");

    // accuracy on the validation split through the decision-boundary path
    int correct = 0, total = s->data.n - s->data.nTrain;
    for (int r = s->data.nTrain; r < s->data.n; ++r) {
        GridSpec g;
        float px = s->data.x[r * 2], py = s->data.x[r * 2 + 1];
        g.x0 = px - 1e-3f; g.x1 = px + 1e-3f; g.y0 = py - 1e-3f; g.y1 = py + 1e-3f;
        g.res = 2;
        g.element = 1;
        float out[4];
        sessionEvalGrid(s.get(), g, out);
        correct += (out[0] > 0.5f) == (s->data.y[r * 2 + 1] > 0.5f);
    }
    printf("  val accuracy %d / %d\n", correct, total);
    check(correct >= total * 95 / 100, "val accuracy at least 95%");
}

// ── 5. session invariants ────────────────────────────────────────────────────

static void testSession() {
    cout << "\n[session]\n";
    SessionConfig cfg = moonsConfig(OptimKind::Adam, 0.01f);
    cfg.batchSize = 8;
    unique_ptr<Session> s;
    check(sessionCreate(moonsMlp(8), cfg, &s).ok(), "session created");

    // seek to i equals advancing i steps, for every i in a reverse pass
    sessionBegin(s.get(), PassKind::Reverse);
    int32_t len = execLength(s->train);
    bool same = true;
    for (int32_t i = 0; i <= len; i += 3) {
        sessionBegin(s.get(), PassKind::Reverse);
        sessionAdvance(s.get(), i, nullptr);
        vector<float> a(s->train.arena.begin(), s->train.arena.end());
        sessionAdvance(s.get(), len, nullptr);
        sessionSeek(s.get(), i);
        same = same && a == vector<float>(s->train.arena.begin(), s->train.arena.end());
    }
    check(same, "seek(i) gives exactly the state of advancing i steps");

    // seeking back inside a training step restores the parameters
    vector<float> start = s->params.values;
    sessionBegin(s.get(), PassKind::Train);
    sessionAdvance(s.get(), 1000, nullptr);
    check(s->step == 1 && s->params.values != start, "a stepped training pass updates parameters");
    sessionSeek(s.get(), 3);
    check(s->step == 0 && s->params.values == start, "seeking back before the updates restores them");

    // reset reproduces a run
    sessionReset(s.get(), 5);
    sessionTrain(s.get(), 50, nullptr);
    vector<float> run1 = s->params.values;
    sessionReset(s.get(), 5);
    sessionTrain(s.get(), 50, nullptr);
    check(run1 == s->params.values, "reset with the same seed reproduces a run exactly");

    // ten thousand steps allocate nothing in the executor
    const float* arena = s->train.arena.data();
    size_t cap = s->train.arena.size();
    auto t0 = chrono::steady_clock::now();
    sessionTrain(s.get(), 10000, nullptr);
    double ms = chrono::duration<double, milli>(chrono::steady_clock::now() - t0).count();
    check(arena == s->train.arena.data() && cap == s->train.arena.size(), "no arena growth over 10,000 steps");
    printf("  10,000 steps in %.0f ms (%.1f us/step, batch 8, hidden 8)\n", ms, ms / 10);
}

static void testFaults() {
    cout << "\n[faults]\n";
    // log of a negative parameter: nan in the forward pass
    {
        Builder b;
        NodeId p = b.param({3}, "p", Init::Constant, -1.0f);
        NodeId l = b.op(Op::Log, {p}, "log");
        setLoss(&b.g, b.op(Op::Sum, {l}));
        unique_ptr<Session> s;
        sessionCreate(b.g, noData(), &s);
        TrainStats ts;
        Status st = sessionTrain(s.get(), 5, &ts);
        check(st.code == Code::NonFinite && st.node == l, "nan is reported at the node where it appeared (" + st.message + ")");
        check(ts.stepsRun == 0 && s->step == 0, "no step is counted");
    }
    // divergence: the update that overflows is rolled back
    {
        Builder b;
        NodeId p = b.param({1}, "p", Init::Constant, 3.0f);
        setLoss(&b.g, b.op(Op::Sum, {b.scalarOp(Op::Pow, p, 4.0f)}));
        SessionConfig c = noData();
        c.optim.lr.baseLr = 100.0f;
        unique_ptr<Session> s;
        sessionCreate(b.g, c, &s);
        TrainStats ts;
        Status st = sessionTrain(s.get(), 100, &ts);
        float last = s->params.values[0];
        check(st.code == Code::NonFinite && isfinite(last), "a diverging run stops with finite parameters");
        check(ts.stepsRun >= 1 && s->step == ts.step, "the steps before the fault are kept");
        vector<float> keep = s->params.values;
        sessionTrain(s.get(), 1, nullptr);
        check(keep == s->params.values, "the faulting step is reproducible and still rolled back");
    }
    // bad optimizer settings leave the old optimizer in place
    {
        Builder b;
        setLoss(&b.g, b.op(Op::Sum, {b.param({2}, "p")}));
        unique_ptr<Session> s;
        sessionCreate(b.g, noData(), &s);
        OptimConfig bad;
        bad.beta1 = 1.5f;
        OptimKind before = s->optim.cfg.kind;
        check(sessionSetOptimizer(s.get(), bad).code == Code::InvalidArgument && s->optim.cfg.kind == before,
              "invalid optimizer settings are rejected and the old optimizer kept");
    }
}

// ── 6. the optimizer race: loss surface over two parameters ──────────────────
// rosenbrock (1 − x)² + 100 (y − x²)², x and y each a scalar parameter

static void testRace() {
    cout << "\n[optimizer race]\n";
    Builder b;
    NodeId x = b.param({}, "x", Init::Constant, -1.5f), y = b.param({}, "y", Init::Constant, 2.0f);
    Attrs one;
    one.scalar = 1.0f;
    Attrs hundred;
    hundred.scalar = 100.0f;
    NodeId a = b.scalarOp(Op::Pow, b.op(Op::Sub, {b.add(Op::Const, {}, one), x}), 2.0f);
    NodeId v = b.scalarOp(Op::Pow, b.op(Op::Sub, {y, b.scalarOp(Op::Pow, x, 2.0f)}), 2.0f);
    setLoss(&b.g, b.op(Op::Add, {a, b.op(Op::Mul, {b.add(Op::Const, {}, hundred), v})}));

    SessionConfig c = noData();
    c.optim.kind = OptimKind::Adam;
    c.optim.lr.baseLr = 0.05f;
    unique_ptr<Session> s;
    check(sessionCreate(b.g, c, &s).ok(), "rosenbrock session");
    GridSpec g;
    g.x0 = 0; g.x1 = 2; g.y0 = 0; g.y1 = 2; g.res = 2;
    float out[4];
    Status st = sessionLossSurface(s.get(), x, 0, y, 0, g, out);
    check(st.ok() && near(out[0], 156.5, 1e-6), "surface at (0.5, 1.5) is 156.5");
    check(s->params.values[0] == -1.5f && s->params.values[1] == 2.0f, "parameters restored after the sweep");
    check(sessionLossSurface(s.get(), x, 0, x, 0, g, out).code == Code::InvalidArgument, "the same element twice is rejected");

    sessionTrain(s.get(), 3000, nullptr);
    printf("  adam after 3000 steps: (%.3f, %.3f)\n", s->params.values[0], s->params.values[1]);
    check(fabs(s->params.values[0] - 1) < 0.05f && fabs(s->params.values[1] - 1) < 0.1f, "adam reaches the valley floor near (1, 1)");
    float m[1];
    check(sessionReadOptim(*s, x, 0, m).ok() && m[0] != 0.0f, "optimizer state is readable");
}

// ── 7. the scalar view agrees with the tensor view ───────────────────────────

static void testScalarize() {
    cout << "\n[scalar view]\n";
    SessionConfig cfg = moonsConfig(OptimKind::Sgd, 0.1f);
    cfg.batchSize = 2;
    unique_ptr<Session> s;
    check(sessionCreate(moonsMlp(3), cfg, &s).ok(), "small mlp session");
    sessionTrain(s.get(), 20, nullptr);
    sessionBegin(s.get(), PassKind::Reverse);
    sessionAdvance(s.get(), 1 << 20, nullptr);

    unique_ptr<Session> sc;
    ScalarMap map;
    Status st = sessionScalarize(*s, &sc, &map);
    check(st.ok(), "scalarize " + st.message);
    if (!st.ok()) return;
    printf("  %zu scalar nodes from %zu tensor nodes\n", sc->trainProg.graph.nodes.size(), s->trainProg.graph.nodes.size());
    sessionBegin(sc.get(), PassKind::Reverse);
    sessionAdvance(sc.get(), 1 << 20, nullptr);

    float tl = execLoss(s->train), sl = execLoss(sc->train);
    check(near(tl, sl, 1e-5), "same loss in both views");
    bool same = true;
    for (const ParamPlan& pp : s->trainProg.params) {
        vector<float> g = readAll(*s, pp.node, Which::Grad);
        for (uint32_t e = 0; e < pp.size; ++e)
            same = same && near(g[e], readAll(*sc, map.elems[pp.node][e], Which::Grad)[0], 1e-4);
    }
    check(same, "every parameter gradient matches element by element");

    cfg.batchSize = 64;
    sessionCreate(moonsMlp(16), cfg, &s);
    check(sessionScalarize(*s, &sc, &map).code == Code::LimitExceeded, "a large graph is refused with LimitExceeded");
}

// ── 7. the C API ─────────────────────────────────────────────────────────────

static void testCapi() {
    cout << "\n[c api]\n";
    ag_handle g = 0, s = 0;
    check(ag_graph_new(&g) == 0 && g != 0, "graph handle");
    ag_attrs a{};
    a.axis = kAllAxes;
    a.init = static_cast<int32_t>(Init::Constant);
    a.requires_grad = 1;
    int32_t ia, ib, ic, id;
    a.init_scale = 2.0f;
    ag_graph_add_node(g, int32_t(Op::Input), nullptr, 0, &a, "a", "", &ia);
    a.init_scale = 3.0f;
    ag_graph_add_node(g, int32_t(Op::Input), nullptr, 0, &a, "b", "", &ib);
    int32_t ab[2] = {ia, ib};
    ag_graph_add_node(g, int32_t(Op::Mul), ab, 2, nullptr, "c", "", &ic);
    int32_t ca[2] = {ic, ia};
    ag_graph_add_node(g, int32_t(Op::Add), ca, 2, nullptr, "d", "", &id);
    ag_graph_set_loss(g, id);

    ag_session_config cfg{};
    cfg.batch_size = 1;
    cfg.eval_batch_size = 1;
    cfg.with_tangents = 1;
    cfg.dataset_kind = int32_t(DatasetKind::None);
    cfg.optim.base_lr = 0.1f;
    cfg.optim.gamma = 0.5f;
    cfg.optim.period = 10;
    cfg.optim.total_steps = 100;
    cfg.optim.beta1 = 0.9f;
    cfg.optim.beta2 = 0.999f;
    cfg.optim.eps = 1e-8f;
    ag_status st = ag_session_new(g, &cfg, &s);
    check(st == 0, string("session handle ") + ag_last_error());

    int32_t len = 0;
    ag_session_schedule_len(s, int32_t(PassKind::Reverse), &len);
    check(len == 9, "schedule length through the C API");
    ag_session_begin(s, int32_t(PassKind::Reverse), -1, 0);
    int32_t ran = 0;
    ag_session_advance(s, len, &ran);
    float ga = 0, gb = 0;
    int32_t w;
    ag_session_read(s, ia, 1, &ga, 1, &w);
    ag_session_read(s, ib, 1, &gb, 1, &w);
    check(ran == 9 && ga == 4.0f && gb == 2.0f, "4 and 2 through the C API");

    check(ag_session_begin(s, int32_t(PassKind::ForwardMode), ic, 0) == int32_t(Code::InvalidArgument),
          "forward mode refuses a seed that is not a leaf");

    int32_t op, ins[4], n;
    ag_session_node_op(s, ic, &op, ins, &n);
    check(op == int32_t(Op::Mul) && n == 2 && ins[1] == ib && string(ag_session_node_name(s, ic)) == "c", "node structure readable");

    check(ag_session_free(s) == 0, "free session");
    check(ag_session_advance(s, 1, &ran) == int32_t(Code::NotFound), "a freed handle returns NotFound");
    ag_handle s2;
    ag_session_new(g, &cfg, &s2);
    check(s2 != s && ag_session_advance(s, 1, &ran) == int32_t(Code::NotFound), "a reused slot does not revive the old handle");
    ag_session_free(s2);
    ag_graph_free(g);

    cfg.optim.base_lr = -1;
    ag_graph_new(&g);
    ag_graph_add_node(g, int32_t(Op::Input), nullptr, 0, &a, "a", "", &ia);
    check(ag_session_new(g, &cfg, &s) == int32_t(Code::InvalidArgument) && strlen(ag_last_error()) > 0,
          "bad settings are reported with a message");
    ag_graph_free(g);
}

// ── 8. an optimizer trajectory for torch.optim ───────────────────────────────

static void dumpOptimizers() {
    const char* names[] = {"sgd", "momentum", "nesterov", "rmsprop", "adam", "adamw"};
    g_dump << "],\"optim\":[";
    for (int k = 0; k <= int(OptimKind::AdamW); ++k) {
        Builder b;
        NodeId p = b.param({4}, "p", Init::Normal);
        NodeId t = b.fixed({4}, "t", Init::Normal, 2.0f);
        setLoss(&b.g, b.op(Op::MseLoss, {b.scalarOp(Op::Pow, p, 3.0f), t}));
        SessionConfig c = noData(3);
        c.optim.kind = OptimKind(k);
        c.optim.lr.baseLr = 0.05f;
        c.optim.weightDecay = 0.01f;
        c.optim.clipNorm = 5.0f;
        unique_ptr<Session> s;
        sessionCreate(b.g, c, &s);
        g_dump << (k ? "," : "") << "{\"kind\":\"" << names[k] << "\",\"lr\":0.05,\"weightDecay\":0.01,\"clipNorm\":5,\"t\":[";
        vector<float> tv = readAll(*s, t, Which::Value);
        for (size_t i = 0; i < tv.size(); ++i) g_dump << (i ? "," : "") << tv[i];
        g_dump << "],\"steps\":[";
        for (int step = 0; step <= 10; ++step) {
            if (step) sessionTrain(s.get(), 1, nullptr);
            g_dump << (step ? "," : "") << "[";
            for (size_t i = 0; i < s->params.values.size(); ++i) g_dump << (i ? "," : "") << s->params.values[i];
            g_dump << "]";
        }
        g_dump << "]}";
    }
}

int main(int argc, char** argv) {
    cout << "=== ag engine tests ===\n";
    cout.precision(9);
    g_dump.precision(9);
    g_dump << "{\"cases\":[";

    testExample();
    testOps();
    testCompileErrors();
    testRobustness();
    testMoons();
    testSession();
    testFaults();
    testRace();
    testScalarize();
    testCapi();
    dumpOptimizers();
    g_dump << "],\"moons\":[";
    for (size_t i = 0; i < g_moons.size(); ++i) g_dump << (i ? "," : "") << g_moons[i];
    g_dump << "]}\n";

    if (argc == 3 && string(argv[1]) == "--dump") {
        ofstream f(argv[2]);
        f << g_dump.str();
        cout << "\nwrote " << argv[2] << "\n";
    }
    cout << "\n=== " << g_passed << " passed, " << g_failed << " failed ===\n";
    return g_failed ? 1 : 0;
}
