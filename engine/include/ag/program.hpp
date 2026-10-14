#pragma once

// program — a graph compiled for one batch size. compile() does every check
// and every size calculation once, so running a program allocates nothing and
// has no failure path except a non-finite number.
//
// the schedule is the product's timeline: a flat list of micro-steps that the
// stepper UI walks one at a time. it is fixed at compile time

#include "ag/graph.hpp"

#include <vector>

namespace ag {

enum class PassKind : uint8_t {
    Forward,       // values only
    Reverse,       // forward, seed, backward
    Train,         // forward, seed, backward, update
    ForwardMode,   // forward, seed tangent, jvp
    Count_
};

enum class StepKind : uint8_t {
    Forward,       // compute value of `node`
    SeedGrad,      // grad(loss) = 1
    Vjp,           // push grad of `node` into its input number `slot`
    Update,        // optimizer update of parameter node `node`
    SeedTangent,   // set the one-hot input direction
    Jvp,           // compute tangent of `node`
};

struct Step {
    StepKind kind = StepKind::Forward;
    uint8_t  slot = 0;
    NodeId   node = kNoNode;
};

inline constexpr uint32_t kNoBuffer = 0xffffffffu;

struct NodePlan {
    Shape    shape;                    // batch dimension resolved
    uint32_t valueOff   = kNoBuffer;   // float offset into the executor arena
    uint32_t gradOff    = kNoBuffer;   // kNoBuffer if !requiresGrad
    uint32_t tangentOff = kNoBuffer;   // kNoBuffer unless compiled with tangents
    int32_t  paramIndex = -1;          // index into Program::params, or -1
    bool     requiresGrad = false;     // true if a path from a Param reaches it without
                                       // passing a Detach or a noGrad node
    uint16_t numConsumers = 0;         // how many grads it will accumulate
    int32_t  lastUse = -1;             // last Reverse-schedule step reading its value
};

struct ParamPlan {
    NodeId   node = kNoNode;
    uint32_t offset = 0;               // float offset into ParamStore::values
    uint32_t size = 0;
};

struct Program {
    Graph                  graph;      // owned copy; the caller may edit theirs
    int32_t                batchSize = 1;
    std::vector<NodePlan>  nodes;
    std::vector<ParamPlan> params;
    uint32_t               arenaFloats = 0;
    uint32_t               paramFloats = 0;
    std::array<std::vector<Step>, static_cast<size_t>(PassKind::Count_)> schedules;
};

struct CompileOptions {
    int32_t batchSize    = 1;
    bool    withTangents = false;      // allocate tangent buffers for ForwardMode
    Limits  limits;
};

// validate, infer shapes, propagate requiresGrad, lay out buffers, build the
// four schedules. on error `out` is left untouched
Status compile(const Graph& g, const CompileOptions& opts, Program* out);

std::span<const Step> schedule(const Program& p, PassKind pass);

// two programs can share one ParamStore iff this holds
bool sameParamLayout(const Program& a, const Program& b);

} // namespace ag
