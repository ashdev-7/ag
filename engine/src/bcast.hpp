#pragma once

// broadcasting index helper shared by the kernels and the scalar expansion.
// strides of an input laid over the output's axes: 0 on a broadcast axis, so
// out element `i` reads input element at(b, i). in a vjp, `+=` through the
// same index sums the gradient back over the broadcast axes for free

#include "ag/core.hpp"

#include <array>

namespace ag {

struct Bcast {
    int32_t rank = 0;
    std::array<int32_t, kMaxRank> dim{};
    std::array<int64_t, kMaxRank> stride{};
};

inline Bcast bcastTo(const Shape& out, const Shape& in) {
    Bcast b;
    b.rank = out.rank;
    int64_t s = 1;
    for (int32_t i = out.rank - 1; i >= 0; --i) {
        int32_t j = i - (out.rank - in.rank);
        b.dim[i] = out.dim[i];
        b.stride[i] = (j >= 0 && in.dim[j] != 1) ? s : 0;
        if (j >= 0) s *= in.dim[j];
    }
    return b;
}

inline int64_t at(const Bcast& b, int64_t flat) {
    int64_t idx = 0;
    for (int32_t i = b.rank - 1; i >= 0; --i) {
        idx += (flat % b.dim[i]) * b.stride[i];
        flat /= b.dim[i];
    }
    return idx;
}

} // namespace ag
