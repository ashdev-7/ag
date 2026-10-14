// checks the parts of the engine that exist before any kernel is written:
// the op table, graph building, shapes, the rng and the C surface.
// exits non-zero if any check fails

#include "ag/capi.h"
#include "ag/ops.hpp"
#include "ag/rng.hpp"
#include "ag/session.hpp"

#include <cstring>
#include <iostream>
#include <set>
#include <string>

using namespace std;
using namespace ag;

static int g_failed = 0;

static void check(bool condition, const string& label) {
    cout << (condition ? "  PASS  " : "  FAIL  ") << label << "\n";
    if (!condition) ++g_failed;
}

// ── op table ─────────────────────────────────────────────────────────────────

static void testOpTable() {
    cout << "\n[op table]\n";

    auto ops = allOps();
    check(static_cast<int>(ops.size()) == kOpCount, "one row per enum value");

    bool ordered = true, arity = true;
    set<string> names;
    int missing = 0;
    for (int i = 0; i < kOpCount; ++i) {
        ordered = ordered && static_cast<int>(ops[i].op) == i;
        arity = arity && ops[i].minInputs <= ops[i].maxInputs && ops[i].maxInputs <= kMaxInputs;
        names.insert(ops[i].name);
        if (!ops[i].forward) ++missing;
    }
    check(ordered, "rows are in enum order");
    check(arity, "arity bounds are sane");
    check(static_cast<int>(names.size()) == kOpCount, "names are unique");
    check(findOp("matmul") && findOp("matmul")->op == Op::MatMul, "findOp finds matmul");
    check(findOp("nope") == nullptr, "findOp returns null for an unknown name");

    cout << "  note  " << missing << " of " << kOpCount << " ops have no forward kernel yet\n";
}

// ── graph building ───────────────────────────────────────────────────────────

static void testGraph() {
    cout << "\n[graph]\n";

    Graph g;
    NodeId x, w, y, bad;
    array<NodeId, 0> none{};

    check(addNode(&g, Op::Input, none, {}, "x", "", &x).ok(), "add input");
    check(addNode(&g, Op::Param, none, {}, "w", "", &w).ok(), "add param");

    array<NodeId, 2> xw{x, w};
    check(addNode(&g, Op::MatMul, xw, {}, "y", "", &y).ok(), "add matmul");
    check(y == 2 && g.nodes[y].numInputs == 2, "ids are dense and inputs recorded");

    array<NodeId, 1> one{x};
    Status s = addNode(&g, Op::MatMul, one, {}, "bad", "", &bad);
    check(s.code == Code::InvalidGraph, "wrong arity is rejected");

    array<NodeId, 2> future{x, 99};
    s = addNode(&g, Op::Add, future, {}, "bad", "", &bad);
    check(s.code == Code::InvalidGraph && s.slot == 1, "forward reference is rejected and names the slot");
    check(g.nodes.size() == 3, "rejected nodes are not added");

    check(setLoss(&g, y).ok() && g.loss == y, "set loss");
    check(setLoss(&g, 42).code == Code::InvalidGraph, "loss must exist");
}

// ── shapes ───────────────────────────────────────────────────────────────────

static void testShapes() {
    cout << "\n[shapes]\n";

    array<int32_t, 2> a{3, 4};
    array<int32_t, 1> b{4};
    array<int32_t, 2> c{3, 5};
    Shape out;

    check(numel(makeShape(a)) == 12, "numel of [3, 4]");
    check(numel(Shape{}) == 1, "rank 0 has one element");
    check(broadcastShapes(makeShape(a), makeShape(b), &out).ok() && sameShape(out, makeShape(a)),
          "[3, 4] with [4] broadcasts to [3, 4]");
    check(broadcastShapes(makeShape(a), makeShape(c), &out).code == Code::ShapeMismatch,
          "[3, 4] with [3, 5] is a mismatch");
    check(toString(makeShape(a)) == "[3, 4]", "toString");

    Limits limits;
    array<int32_t, 2> big{1000, 1000};
    check(validateShape(makeShape(big), limits).code == Code::LimitExceeded, "oversized tensor is rejected");
}

// ── rng ──────────────────────────────────────────────────────────────────────

static void testRng() {
    cout << "\n[rng]\n";

    Rng a, b, c;
    rngSeed(&a, 42);
    rngSeed(&b, 42);
    rngSeed(&c, 43);

    bool same = true, differs = false, inRange = true;
    for (int i = 0; i < 1000; ++i) {
        uint32_t va = rngU32(&a);
        same = same && va == rngU32(&b);
        differs = differs || va != rngU32(&c);
        float u = rngUniform(&a);
        rngUniform(&b);
        inRange = inRange && u >= 0.0f && u < 1.0f;
    }
    check(same, "same seed gives the same stream");
    check(differs, "different seed gives a different stream");
    check(inRange, "uniform stays in [0, 1)");

    // reference value from the PCG32 demo: seed 42, stream 54
    Rng ref;
    rngSeed(&ref, 42, 54);
    check(rngU32(&ref) == 0xa15c02b7u, "matches the PCG32 reference output");
}

// ── C surface ────────────────────────────────────────────────────────────────

static void testCapi() {
    cout << "\n[capi]\n";

    check(ag_abi_version() == AG_ABI_VERSION, "abi version");
    check(ag_op_count() == kOpCount, "op count");
    check(ag_op_name(-1) == nullptr && ag_op_name(kOpCount) == nullptr, "out-of-range op name is null");
    check(strcmp(ag_op_name(static_cast<int32_t>(Op::Relu)), "relu") == 0, "op name by number");

    ag_op_desc desc;
    check(ag_op_describe(static_cast<int32_t>(Op::Add), &desc) == 0 && desc.min_inputs == 2, "describe add");
    check(ag_op_describe(999, &desc) != 0, "describe rejects a bad op");

    check(ag_graph_free(12345) == static_cast<ag_status>(Code::NotFound) && strlen(ag_last_error()) > 0,
          "a bogus handle reports NotFound with a message");

    // layout the web client depends on
    check(sizeof(ag_shape) == 20 && sizeof(ag_step) == 12 && sizeof(ag_attrs) == 48,
          "struct sizes match the documented ABI");
    // offsets the web bridge (web/src/engine/bridge.ts) writes and reads
    check(sizeof(ag_stats) == 164 && sizeof(ag_optim_config) == 56 && sizeof(ag_session_config) == 92 &&
              sizeof(ag_train_stats) == 24 && sizeof(ag_grid) == 28,
          "struct sizes used by the web bridge");
}

int main() {
    cout << "=== ag engine skeleton tests ===\n";

    testOpTable();
    testGraph();
    testShapes();
    testRng();
    testCapi();

    cout << "\n=== " << (g_failed ? "FAILED: " + to_string(g_failed) : string("all passed")) << " ===\n";
    return g_failed ? 1 : 0;
}
