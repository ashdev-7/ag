#pragma once

// core — the vocabulary every other file uses: limits, status codes, shapes,
// tensor views and summary statistics. No allocation, no graph knowledge.

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>

namespace ag {

// ── limits ───────────────────────────────────────────────────────────────────
// hard caps checked at compile(); exceeding one is an error, never a slowdown

inline constexpr int kMaxRank   = 4;
inline constexpr int kMaxInputs = 4;
inline constexpr int kHistBins  = 32;

struct Limits {
    uint32_t maxNodes        = 256;
    uint32_t maxTensorElems  = 65536;
    uint32_t maxParams       = 100000;
    uint32_t maxArenaFloats  = 8u << 20;   // 32 MB of float32 per executor
    uint32_t maxScalarNodes  = 500;        // scalar zoom level
    uint32_t maxSessions     = 8;
};

// ── status ───────────────────────────────────────────────────────────────────
// the engine never throws. every fallible function returns a Status; node and
// slot say where in the graph the problem is so the UI can point at it

using NodeId = int32_t;
inline constexpr NodeId kNoNode = -1;

enum class Code : int32_t {
    Ok = 0,
    InvalidArgument,   // caller passed something malformed
    InvalidGraph,      // structure: bad id, arity, missing loss
    ShapeMismatch,     // shapes do not agree at `node`, input `slot`
    LimitExceeded,     // a Limits cap was hit
    OutOfMemory,
    NonFinite,         // nan or inf first appeared at `node`
    InvalidState,      // call made in the wrong order
    NotFound,          // stale or unknown handle
    Unimplemented,
    Internal,          // invariant broken: a bug in the engine
};

struct Status {
    Code        code = Code::Ok;
    NodeId      node = kNoNode;
    int32_t     slot = -1;
    std::string message;

    bool ok() const { return code == Code::Ok; }
};

Status      makeError(Code code, std::string message, NodeId node = kNoNode, int32_t slot = -1);
const char* codeName(Code code);

// early-return helper used throughout the engine
#define AG_TRY(expr)                          \
    do {                                      \
        ::ag::Status ag_st_ = (expr);         \
        if (!ag_st_.ok()) return ag_st_;      \
    } while (0)

// ── shape ────────────────────────────────────────────────────────────────────
// row-major, always contiguous. there are no strides and no views: reshape and
// transpose produce a new buffer. rank 0 is a scalar with one element

inline constexpr int32_t kBatchDim = -1;   // placeholder resolved at compile()

struct Shape {
    int32_t                       rank = 0;
    std::array<int32_t, kMaxRank> dim{};
};

Shape       makeShape(std::span<const int32_t> dims);
int64_t     numel(const Shape& s);
bool        sameShape(const Shape& a, const Shape& b);
std::string toString(const Shape& s);
Status      validateShape(const Shape& s, const Limits& limits);

// numpy broadcasting rules; error names the first disagreeing axis
Status broadcastShapes(const Shape& a, const Shape& b, Shape* out);

// ── tensor views ─────────────────────────────────────────────────────────────
// non-owning. the buffers live in an executor arena or a parameter store

struct TensorView {
    float* data = nullptr;
    Shape  shape;
};

struct ConstTensorView {
    const float* data = nullptr;
    Shape        shape;
};

// double to float, saturating to ±inf. a plain conversion of a value outside
// float's range is undefined behaviour; sums taken in double can get there
// on a diverging run, and the executor must see inf, not garbage
inline float toFloat(double x) {
    constexpr double kMax = std::numeric_limits<float>::max();
    if (x > kMax) return std::numeric_limits<float>::infinity();
    if (x < -kMax) return -std::numeric_limits<float>::infinity();
    return static_cast<float>(x);   // in range, or nan
}

// ── statistics ───────────────────────────────────────────────────────────────
// what the UI needs per node per frame, so it never has to copy whole tensors
// while training is running

struct Stats {
    float    min = 0, max = 0, mean = 0, std = 0;
    float    l2 = 0;            // euclidean norm
    float    fracZero = 0;      // dead-unit detection
    uint32_t nonFinite = 0;     // count of nan/inf elements
    float    histLo = 0, histHi = 0;
    std::array<uint32_t, kHistBins> hist{};
};

void computeStats(ConstTensorView t, Stats* out);

// index of the first nan/inf element, or -1
int64_t firstNonFinite(ConstTensorView t);

} // namespace ag
