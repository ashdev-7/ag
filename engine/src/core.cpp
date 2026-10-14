#include "ag/core.hpp"

#include <algorithm>
#include <cmath>

using namespace std;

namespace ag {

// ── status ───────────────────────────────────────────────────────────────────

Status makeError(Code code, string message, NodeId node, int32_t slot) {
    Status s;
    s.code = code;
    s.node = node;
    s.slot = slot;
    s.message = std::move(message);
    return s;
}

const char* codeName(Code code) {
    switch (code) {
        case Code::Ok:              return "Ok";
        case Code::InvalidArgument: return "InvalidArgument";
        case Code::InvalidGraph:    return "InvalidGraph";
        case Code::ShapeMismatch:   return "ShapeMismatch";
        case Code::LimitExceeded:   return "LimitExceeded";
        case Code::OutOfMemory:     return "OutOfMemory";
        case Code::NonFinite:       return "NonFinite";
        case Code::InvalidState:    return "InvalidState";
        case Code::NotFound:        return "NotFound";
        case Code::Unimplemented:   return "Unimplemented";
        case Code::Internal:        return "Internal";
    }
    return "Unknown";
}

// ── shape ────────────────────────────────────────────────────────────────────

Shape makeShape(span<const int32_t> dims) {
    Shape s;
    s.rank = static_cast<int32_t>(min<size_t>(dims.size(), kMaxRank));
    for (int32_t i = 0; i < s.rank; ++i) s.dim[i] = dims[i];
    return s;
}

// saturates at INT64_MAX instead of overflowing, so a caller that has not
// validated the shape still gets "too big", never a small wrapped number
int64_t numel(const Shape& s) {
    int64_t n = 1;
    for (int32_t i = 0; i < s.rank; ++i)
        if (__builtin_mul_overflow(n, int64_t(s.dim[i]), &n)) return INT64_MAX;
    return n;
}

bool sameShape(const Shape& a, const Shape& b) {
    if (a.rank != b.rank) return false;
    for (int32_t i = 0; i < a.rank; ++i)
        if (a.dim[i] != b.dim[i]) return false;
    return true;
}

string toString(const Shape& s) {
    string out = "[";
    for (int32_t i = 0; i < s.rank; ++i) {
        if (i) out += ", ";
        out += s.dim[i] == kBatchDim ? "B" : to_string(s.dim[i]);
    }
    return out + "]";
}

// the size is checked as it is multiplied: four int32 dimensions can reach
// 2^124, and a wrapped product would let a huge tensor pass as a tiny one.
// stopping as soon as the running product passes the limit keeps every
// intermediate below 2^32 * 2^31
Status validateShape(const Shape& s, const Limits& limits) {
    if (s.rank < 0 || s.rank > kMaxRank)
        return makeError(Code::InvalidArgument, "rank must be between 0 and " + to_string(kMaxRank));
    for (int32_t i = 0; i < s.rank; ++i)
        if (s.dim[i] <= 0)
            return makeError(Code::InvalidArgument, "dimension " + to_string(i) + " must be positive in " + toString(s));
    int64_t n = 1;
    for (int32_t i = 0; i < s.rank; ++i) {
        n *= s.dim[i];
        if (n > static_cast<int64_t>(limits.maxTensorElems))
            return makeError(Code::LimitExceeded, "tensor " + toString(s) + " exceeds " +
                                                  to_string(limits.maxTensorElems) + " elements");
    }
    return {};
}

// align from the right; a dimension of 1 stretches to match the other

Status broadcastShapes(const Shape& a, const Shape& b, Shape* out) {
    Shape r;
    r.rank = max(a.rank, b.rank);
    for (int32_t i = 0; i < r.rank; ++i) {
        int32_t ai = i - (r.rank - a.rank);
        int32_t bi = i - (r.rank - b.rank);
        int32_t da = ai >= 0 ? a.dim[ai] : 1;
        int32_t db = bi >= 0 ? b.dim[bi] : 1;
        if (da != db && da != 1 && db != 1)
            return makeError(Code::ShapeMismatch, "cannot broadcast " + toString(a) + " with " +
                                                  toString(b) + " at axis " + to_string(i));
        r.dim[i] = max(da, db);
    }
    *out = r;
    return {};
}

// ── statistics ───────────────────────────────────────────────────────────────
// one pass for the moments (in double, so long tensors do not drift), a second
// for the histogram once the range is known. non-finite elements are counted
// and otherwise ignored

void computeStats(ConstTensorView t, Stats* out) {
    Stats s;
    int64_t n = numel(t.shape), finite = 0, zeros = 0;
    double sum = 0, sumSq = 0;
    float lo = INFINITY, hi = -INFINITY;
    for (int64_t i = 0; i < n; ++i) {
        float x = t.data[i];
        if (!isfinite(x)) { ++s.nonFinite; continue; }
        ++finite;
        if (x == 0.0f) ++zeros;
        lo = min(lo, x);
        hi = max(hi, x);
        sum += x;
        sumSq += double(x) * x;
    }
    if (finite > 0) {
        double mean = sum / finite;
        s.min = lo;
        s.max = hi;
        s.mean = toFloat(mean);
        s.std = toFloat(sqrt(max(0.0, sumSq / finite - mean * mean)));
        s.l2 = toFloat(sqrt(sumSq));
        s.fracZero = float(double(zeros) / double(n));
        s.histLo = lo;
        s.histHi = hi;
        // in double: for values near ±3e38, hi − lo overflows float to inf and
        // (x − lo) / width becomes nan, and converting nan to int is undefined
        // (in WebAssembly it can trap). in double both stay finite
        double width = double(hi) - double(lo);
        for (int64_t i = 0; i < n; ++i) {
            float x = t.data[i];
            if (!isfinite(x)) continue;
            double pos = width > 0 ? (double(x) - lo) / width * kHistBins : kHistBins / 2;
            int bin = pos >= kHistBins ? kHistBins - 1 : pos > 0 ? int(pos) : 0;
            s.hist[bin]++;
        }
    }
    *out = s;
}

int64_t firstNonFinite(ConstTensorView t) {
    int64_t n = numel(t.shape);
    for (int64_t i = 0; i < n; ++i)
        if (!isfinite(t.data[i])) return i;
    return -1;
}

} // namespace ag
