#pragma once

// exec — runs a program's schedule. a resumable cursor over the micro-steps:
// advance one step, many, or seek to any index by replaying from zero.
// one arena, allocated once in execInit, holds every value, grad and tangent

#include "ag/program.hpp"
#include "ag/rng.hpp"

#include <vector>

namespace ag {

// parameter values live outside the executor so that a training program and
// an evaluation program (different batch size) share them
struct ParamStore {
    std::vector<float>     values;
    std::vector<ParamPlan> layout;
};

// fill a buffer by an Init rule. fan-in and fan-out come from the shape: a
// matrix is [in, out], as in x @ W. also used for inputs that have no data role
void fillInit(std::span<float> out, const Shape& shape, Init init, float scale, Rng* rng);

// each parameter draws from its own stream (stream = node id), so adding a
// node does not change how the others are initialised
Status initParams(const Program& p, uint64_t seed, ParamStore* out);
Status exportParams(const ParamStore& ps, std::span<float> out);
Status importParams(ParamStore* ps, std::span<const float> in);

// forward mode differentiates along one direction: element `element` of the
// leaf `node`
struct TangentSeed {
    NodeId  node = kNoNode;
    int32_t element = 0;
};

struct Optimizer;   // optim.hpp

struct Exec {
    const Program*     prog = nullptr;
    ParamStore*        params = nullptr;
    Optimizer*         optim = nullptr;    // required for PassKind::Train only
    std::vector<float> arena;
    PassKind           pass = PassKind::Forward;
    TangentSeed        seed;
    int32_t            cursor = 0;         // next step to execute
    bool               begun = false;
    Status             fault;              // sticky until the next execBegin
};

enum class Which : uint8_t { Value, Grad, Tangent };

Status execInit(const Program* prog, ParamStore* params, Exec* out);
void   execBindOptimizer(Exec* e, Optimizer* optim);

// copy data into an Input node's value buffer; size must match exactly
Status execBindInput(Exec* e, NodeId input, std::span<const float> data);

// start a pass: zero grads and tangents, cursor to 0, clear any fault
Status execBegin(Exec* e, PassKind pass, TangentSeed seed = {});

// run up to n steps. stops early at the end of the schedule or on the first
// non-finite value, which is returned and also kept in e->fault
Status execAdvance(Exec* e, int32_t n, int32_t* executed);
Status execRun(Exec* e);

// reposition to `index` by replaying from step 0. not valid for Train passes,
// which mutate parameters; Session handles those with a snapshot
Status execSeek(Exec* e, int32_t index);

int32_t execLength(const Exec& e);

// views are valid until the Exec is destroyed; contents change as it advances
ConstTensorView execView(const Exec& e, NodeId node, Which which);
Status          execRead(const Exec& e, NodeId node, Which which, std::span<float> out);
Status          execStats(const Exec& e, NodeId node, Which which, Stats* out);
float           execLoss(const Exec& e);
float           execGradNorm(const Exec& e);   // global l2 norm over all params

} // namespace ag
