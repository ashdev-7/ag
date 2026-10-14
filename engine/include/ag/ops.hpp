#pragma once

// ops — one table row per op. everything that needs to know about ops (shape
// checking, forward, both AD modes, the scalar zoom level, the palette) reads
// this table. adding an op is adding a row and four small functions

#include "ag/graph.hpp"

#include <span>
#include <string_view>

namespace ag {

enum class Category : uint8_t { Source, Arithmetic, Linear, Reduce, ShapeOp, Activation, Loss };

// which forward values the vjp reads; drives the memory timeline in phase 2
enum SaveBits : uint8_t { kSavesNothing = 0, kSavesInputs = 1, kSavesOutput = 2 };

// ── kernel arguments ─────────────────────────────────────────────────────────
// kernels cannot fail: shapes were proven at compile() and buffers are sized.
// non-finite results are detected by the executor after the kernel returns

struct ForwardArgs {
    std::span<const ConstTensorView> in;
    TensorView                       out;
    const Attrs&                     attrs;
};

// accumulate (+=) d(loss)/d(in[i]) into gin[i] for every i with bit i set in
// `need`. never overwrite: a value used twice receives two contributions
struct VjpArgs {
    std::span<const ConstTensorView> in;
    ConstTensorView                  out;
    ConstTensorView                  gout;
    std::span<const TensorView>      gin;
    uint32_t                         need;
    const Attrs&                     attrs;
};

// write the tangent of the output. tin[i].data == nullptr means that input
// has a zero tangent
struct JvpArgs {
    std::span<const ConstTensorView> in;
    ConstTensorView                  out;
    std::span<const ConstTensorView> tin;
    TensorView                       tout;
    const Attrs&                     attrs;
};

struct ScalarizeArgs;   // defined in scalarize.hpp

using InferFn     = Status (*)(std::span<const Shape> in, const Attrs& attrs, Shape* out);
using ForwardFn   = void (*)(const ForwardArgs& a);
using VjpFn       = void (*)(const VjpArgs& a);
using JvpFn       = void (*)(const JvpArgs& a);
using ScalarizeFn = Status (*)(const ScalarizeArgs& a);

struct OpInfo {
    Op          op;
    const char* name;        // stable identifier, also the GraphSpec string
    const char* symbol;      // short label drawn on the node
    Category    category;
    uint8_t     minInputs;
    uint8_t     maxInputs;
    uint8_t     saves;       // SaveBits
    InferFn     infer;
    ForwardFn   forward;
    VjpFn       vjp;         // null for sources
    JvpFn       jvp;         // null for sources
    ScalarizeFn scalarize;   // null: op has no scalar expansion
};

const OpInfo&           opInfo(Op op);
std::span<const OpInfo> allOps();
const OpInfo*           findOp(std::string_view name);   // null if unknown

} // namespace ag
