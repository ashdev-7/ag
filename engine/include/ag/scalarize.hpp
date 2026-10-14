#pragma once

// scalarize — the scalar zoom level. rewrites a compiled tensor graph into an
// equivalent graph where every element is its own rank-0 node, built only
// from scalar Add/Mul/... ops. the same engine then runs it, so the scalar
// view can never disagree with the tensor view about a derivative

#include "ag/ops.hpp"
#include "ag/program.hpp"

#include <vector>

namespace ag {

// per-op expansion rule. `in[i][e]` is the scalar node holding element e of
// input i; the rule appends nodes to `out` and writes the scalar node of each
// output element to `outElems`. reshape and transpose only permute ids and
// add no nodes
struct ScalarizeArgs {
    const Node&                          node;
    std::span<const Shape>               inShapes;
    Shape                                outShape;
    std::span<const std::vector<NodeId>> in;
    Graph*                               out;
    std::vector<NodeId>*                 outElems;
    uint32_t                             maxNodes;   // a rule stops with LimitExceeded past this
};

struct ScalarMap {
    std::vector<std::vector<NodeId>> elems;   // tensor node -> scalar node of each element
    std::vector<NodeId>  source;              // scalar node -> tensor node it was made for
    std::vector<int32_t> element;             // scalar node -> flat element index, -1 if intermediate
};

// leaves become rank-0 Input / Param / Const nodes whose values the caller
// fills in (sessionScalarize copies them from the running session).
// fails with LimitExceeded if the result would pass limits.maxScalarNodes, and
// with Unimplemented naming the node if an op has no expansion rule
Status scalarize(const Program& p, const Limits& limits, Graph* out, ScalarMap* map);

} // namespace ag
