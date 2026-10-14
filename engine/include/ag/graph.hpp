#pragma once

// graph — the static description the user edits. plain data, no values.
// a node may only take inputs with a smaller id, so the node list is already
// in topological order and a cycle cannot be expressed

#include "ag/core.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace ag {

// ── ops ──────────────────────────────────────────────────────────────────────
// order is part of the ABI: the web client refers to ops by this number.
// append only

enum class Op : uint8_t {
    // sources
    Input, Param, Const,
    // arithmetic, elementwise with broadcasting
    Add, Sub, Mul, Div, Neg, Pow, Exp, Log,
    // linear algebra
    MatMul,
    // reductions
    Sum, Mean,
    // shape
    Reshape, Transpose,
    // activations
    Relu, LeakyRelu, Tanh, Sigmoid, Softmax,
    // losses, always scalar output
    MseLoss, SoftmaxCrossEntropy,

    Count_
};

inline constexpr int kOpCount = static_cast<int>(Op::Count_);

// which dataset column feeds an Input node
enum class Role : uint8_t { None, Features, Targets };

enum class Init : uint8_t {
    Zeros, Constant, Uniform, Normal,
    XavierUniform, XavierNormal, HeUniform, HeNormal,
};

inline constexpr int32_t kAllAxes = -128;

// one fixed record for every op; each op reads only the fields it documents
// in docs (ops table in DESIGN_PHASE1.md)
struct Attrs {
    Shape   shape;                  // Input/Param/Const: declared shape. Reshape: target
    int32_t axis         = kAllAxes; // Sum, Mean, Softmax
    bool    keepDim      = false;    // Sum, Mean
    float   scalar       = 0.0f;     // Pow: exponent. LeakyRelu: slope. Const: fill value
    Role    role         = Role::None;
    Init    init         = Init::Zeros;
    float   initScale    = 1.0f;     // Uniform: half-width. Normal: std. Constant: value
    bool    requiresGrad = false;    // Input only; Param is always true
};

struct Node {
    Op                             op = Op::Const;
    uint8_t                        numInputs = 0;
    std::array<NodeId, kMaxInputs> inputs{kNoNode, kNoNode, kNoNode, kNoNode};
    Attrs                          attrs;
    std::string                    name;     // shown on the canvas
    std::string                    module;   // "mlp/layer1": path for module zoom
};

struct Graph {
    std::vector<Node>   nodes;
    NodeId              loss = kNoNode;      // root of the backward pass
    std::vector<NodeId> outputs;             // nodes evaluated for predictions
};

// ── building ─────────────────────────────────────────────────────────────────
// addNode checks arity and that every input id already exists. shapes are
// checked later, in compile(), where the batch size is known

Status addNode(Graph* g, Op op, std::span<const NodeId> inputs, const Attrs& attrs,
               std::string_view name, std::string_view module, NodeId* outId);
Status setLoss(Graph* g, NodeId node);
Status addOutput(Graph* g, NodeId node);

// whole-graph structural check: ids, arity, roles, one scalar loss, limits
Status validate(const Graph& g, const Limits& limits);

// stable text form for golden tests and bug reports
std::string dump(const Graph& g);

} // namespace ag
