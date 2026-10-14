// step time against model size. the same file builds natively (make bench)
// and to WebAssembly under node (make wasm-bench), so the two can be compared.
// an MLP of `depth` tanh layers of width `width` on moons, softmax
// cross-entropy, Adam

#include "ag/session.hpp"

#include <chrono>
#include <cstdio>

using namespace ag;
using namespace std;

static Graph mlp(int depth, int width) {
    Graph g;
    auto add = [&](Op op, initializer_list<NodeId> in, Attrs a = {}) {
        NodeId id;
        addNode(&g, op, span<const NodeId>(in.begin(), in.size()), a, "", "", &id);
        return id;
    };
    auto shaped = [](initializer_list<int32_t> d, Init init) {
        Attrs a;
        a.shape = makeShape(span<const int32_t>(d.begin(), d.size()));
        a.init = init;
        return a;
    };
    Attrs xa = shaped({kBatchDim, 2}, Init::Zeros), ya = xa;
    xa.role = Role::Features;
    ya.role = Role::Targets;
    NodeId h = add(Op::Input, {}, xa), y = add(Op::Input, {}, ya);
    int in = 2;
    for (int l = 0; l < depth; ++l) {
        NodeId w = add(Op::Param, {}, shaped({in, width}, Init::XavierNormal));
        NodeId b = add(Op::Param, {}, shaped({width}, Init::Zeros));
        h = add(Op::Tanh, {add(Op::Add, {add(Op::MatMul, {h, w}), b})});
        in = width;
    }
    NodeId w = add(Op::Param, {}, shaped({in, 2}, Init::XavierNormal));
    NodeId logits = add(Op::MatMul, {h, w});
    setLoss(&g, add(Op::SoftmaxCrossEntropy, {logits, y}));
    return g;
}

int main() {
    printf("%-6s %-6s %-6s %10s %12s %14s\n", "depth", "width", "batch", "params", "arena floats", "us per step");
    int configs[][3] = {{2, 8, 32}, {2, 16, 32}, {2, 32, 32}, {2, 64, 32}, {2, 128, 32}, {4, 64, 32}, {2, 64, 128}, {4, 128, 128}};
    for (auto& c : configs) {
        SessionConfig cfg;
        cfg.batchSize = c[2];
        cfg.evalBatchSize = c[2];
        cfg.data.n = 1000;
        cfg.optim.kind = OptimKind::Adam;
        cfg.optim.lr.baseLr = 0.01f;
        unique_ptr<Session> s;
        Status st = sessionCreate(mlp(c[0], c[1]), cfg, &s);
        if (!st.ok()) {
            printf("%d x %d: %s\n", c[0], c[1], st.message.c_str());
            continue;
        }
        sessionTrain(s.get(), 20, nullptr);   // warm up
        int steps = 0;
        auto t0 = chrono::steady_clock::now();
        double us = 0;
        while (us < 300000) {
            sessionTrain(s.get(), 10, nullptr);
            steps += 10;
            us = chrono::duration<double, micro>(chrono::steady_clock::now() - t0).count();
        }
        printf("%-6d %-6d %-6d %10u %12u %14.1f\n", c[0], c[1], c[2], s->trainProg.paramFloats, s->trainProg.arenaFloats, us / steps);
    }
    return 0;
}
