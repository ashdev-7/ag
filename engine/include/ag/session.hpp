#pragma once

// session — everything one training run owns: graph, two compiled programs
// (train batch, eval batch) sharing one set of parameters, optimizer, data.
// this is the object the web client holds a handle to.
//
// every training step is a transaction: state is snapshotted when the step
// begins and restored if it faults, so a nan never reaches the parameters

#include "ag/data.hpp"
#include "ag/exec.hpp"
#include "ag/optim.hpp"
#include "ag/scalarize.hpp"

#include <memory>

namespace ag {

struct SessionConfig {
    int32_t       batchSize = 16;
    int32_t       evalBatchSize = 256;
    uint64_t      seed = 0;
    bool          withTangents = true;
    Limits        limits;
    OptimConfig   optim;
    DatasetConfig data;
};

// state needed to undo or replay one training step
struct Snapshot {
    std::vector<float> params, m, v;
    int64_t            t = 0;
    Batcher            batcher;
    int64_t            step = 0;
    bool               valid = false;
};

struct Session {
    SessionConfig cfg;
    Program       trainProg;
    Program       evalProg;
    ParamStore    params;
    Exec          train;
    Exec          eval;
    Optimizer     optim;
    Dataset       data;
    Batcher       batcher;
    Snapshot      stepStart;
    int64_t       step = 0;          // completed training steps
    Status        fault;             // last fault; cleared by reset or a good step
    std::vector<float> batchX, batchY;
};

struct TrainStats {
    int64_t step = 0;
    int32_t stepsRun = 0;
    float   loss = 0;                // last step
    float   gradNorm = 0;            // before clipping
    float   lr = 0;
};

struct GridSpec {
    float   x0 = -1, x1 = 1, y0 = -1, y1 = 1;
    int32_t res = 100;               // res * res points, row-major, y descending
    NodeId  node = kNoNode;          // which node to sample; kNoNode = first output
    int32_t element = 0;             // which element of that node per sample
};

// ── lifecycle ────────────────────────────────────────────────────────────────

Status sessionCreate(const Graph& g, const SessionConfig& cfg, std::unique_ptr<Session>* out);

// re-initialise parameters, optimizer state and batch order from a seed
Status sessionReset(Session* s, uint64_t seed);

// replace the optimizer; resets its state, keeps parameters. both this and
// importing parameters cancel a stepped training pass that is in progress
Status sessionSetOptimizer(Session* s, const OptimConfig& cfg);
Status sessionImportParams(Session* s, std::span<const float> in);

// ── stepping (the stepper UI) ────────────────────────────────────────────────

// draw the next batch (Train and Reverse passes), snapshot, cursor to 0
Status sessionBegin(Session* s, PassKind pass, TangentSeed seed = {});
Status sessionAdvance(Session* s, int32_t n, int32_t* executed);

// any index in the current pass, forwards or backwards. restores the snapshot
// and replays, so it is exact
Status sessionSeek(Session* s, int32_t index);

// ── training (the run button) ────────────────────────────────────────────────

// run whole steps. on a fault the faulting step is rolled back, stepsRun says
// how many completed, and the Status names the node where nan/inf appeared
Status sessionTrain(Session* s, int32_t nSteps, TrainStats* out);

// ── reading ──────────────────────────────────────────────────────────────────

Status sessionRead(const Session& s, NodeId node, Which which, std::span<float> out);
Status sessionStats(const Session& s, NodeId node, Which which, Stats* out);
Status sessionEvalLoss(Session* s, Split split, float* out);

// decision boundary: forward a res*res grid of 2-d inputs through evalProg
Status sessionEvalGrid(Session* s, const GridSpec& grid, std::span<float> out);

// loss surface for the optimizer race: sweep two parameter elements (of the
// same or different parameters) over a grid, holding the rest fixed.
// parameters are restored afterwards
Status sessionLossSurface(Session* s, NodeId paramX, int32_t elemX, NodeId paramY, int32_t elemY,
                          const GridSpec& grid, std::span<float> out);

// optimizer state of one parameter: which 0 is the first moment / velocity,
// 1 the second moment. laid out like the parameter
Status sessionReadOptim(const Session& s, NodeId param, int32_t which, std::span<float> out);

// scalar zoom: a new session whose graph is the scalar expansion of this one,
// with leaf values copied from the current batch and parameters
Status sessionScalarize(const Session& s, std::unique_ptr<Session>* out, ScalarMap* map);

} // namespace ag
