#include "ag/ops.hpp"
#include "bcast.hpp"
#include "scalar_rules.hpp"

#include <algorithm>
#include <array>
#include <cmath>

using namespace std;

namespace ag {

// kernels are plain loops over contiguous float32. shapes were proven by
// infer() at compile time, so nothing here checks sizes. sums are taken in
// double where a tensor collapses to one number, so a loss over 64k elements
// does not drift

// ── infer ────────────────────────────────────────────────────────────────────

static Status inferSource(span<const Shape>, const Attrs& attrs, Shape* out) {
    *out = attrs.shape;
    return {};
}

static Status inferBinary(span<const Shape> in, const Attrs&, Shape* out) {
    Status s = broadcastShapes(in[0], in[1], out);
    if (!s.ok()) s.slot = 1;
    return s;
}

static Status inferSame(span<const Shape> in, const Attrs&, Shape* out) {
    *out = in[0];
    return {};
}

static Status inferMatMul(span<const Shape> in, const Attrs&, Shape* out) {
    if (in[0].rank != 2) return makeError(Code::ShapeMismatch, "matmul needs a matrix, got " + toString(in[0]), kNoNode, 0);
    if (in[1].rank != 2) return makeError(Code::ShapeMismatch, "matmul needs a matrix, got " + toString(in[1]), kNoNode, 1);
    if (in[0].dim[1] != in[1].dim[0])
        return makeError(Code::ShapeMismatch, "matmul " + toString(in[0]) + " @ " + toString(in[1]) +
                                              ": inner dimensions differ", kNoNode, 1);
    out->rank = 2;
    out->dim[0] = in[0].dim[0];
    out->dim[1] = in[1].dim[1];
    return {};
}

// a single axis, counted from the end if negative. kAllAxes reduces everything
static bool resolveAxis(int32_t axis, int32_t rank, int32_t* out) {
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank) return false;
    *out = axis;
    return true;
}

// split a shape around `axis` into outer x len x inner
static void around(const Shape& s, int32_t axis, int64_t* outer, int64_t* len, int64_t* inner) {
    *outer = *inner = 1;
    for (int32_t i = 0; i < axis; ++i) *outer *= s.dim[i];
    *len = s.dim[axis];
    for (int32_t i = axis + 1; i < s.rank; ++i) *inner *= s.dim[i];
}

static Status inferReduce(span<const Shape> in, const Attrs& attrs, Shape* out) {
    Shape r;
    if (attrs.axis == kAllAxes) {
        if (attrs.keepDim) {
            r.rank = in[0].rank;
            for (int32_t i = 0; i < r.rank; ++i) r.dim[i] = 1;
        }
        *out = r;
        return {};
    }
    int32_t axis;
    if (!resolveAxis(attrs.axis, in[0].rank, &axis))
        return makeError(Code::ShapeMismatch, "axis " + to_string(attrs.axis) + " is out of range for " + toString(in[0]), kNoNode, 0);
    for (int32_t i = 0; i < in[0].rank; ++i) {
        if (i == axis && !attrs.keepDim) continue;
        r.dim[r.rank++] = i == axis ? 1 : in[0].dim[i];
    }
    *out = r;
    return {};
}

static Status inferReshape(span<const Shape> in, const Attrs& attrs, Shape* out) {
    if (numel(in[0]) != numel(attrs.shape))
        return makeError(Code::ShapeMismatch, "cannot reshape " + toString(in[0]) + " to " + toString(attrs.shape), kNoNode, 0);
    *out = attrs.shape;
    return {};
}

static Status inferTranspose(span<const Shape> in, const Attrs&, Shape* out) {
    if (in[0].rank != 2) return makeError(Code::ShapeMismatch, "transpose needs a matrix, got " + toString(in[0]), kNoNode, 0);
    out->rank = 2;
    out->dim[0] = in[0].dim[1];
    out->dim[1] = in[0].dim[0];
    return {};
}

static Status inferSoftmax(span<const Shape> in, const Attrs& attrs, Shape* out) {
    int32_t axis;
    if (in[0].rank == 0 || !resolveAxis(attrs.axis == kAllAxes ? -1 : attrs.axis, in[0].rank, &axis))
        return makeError(Code::ShapeMismatch, "softmax axis is out of range for " + toString(in[0]), kNoNode, 0);
    *out = in[0];
    return {};
}

static Status inferLoss(span<const Shape> in, const Attrs&, Shape* out) {
    if (!sameShape(in[0], in[1]))
        return makeError(Code::ShapeMismatch, "prediction " + toString(in[0]) + " and target " + toString(in[1]) +
                                              " must have the same shape", kNoNode, 1);
    *out = Shape{};
    return {};
}

static Status inferCrossEntropy(span<const Shape> in, const Attrs& attrs, Shape* out) {
    if (in[0].rank == 0) return makeError(Code::ShapeMismatch, "cross-entropy needs logits with a class axis", kNoNode, 0);
    return inferLoss(in, attrs, out);
}

// ── sources ──────────────────────────────────────────────────────────────────
// Input values are bound by the caller and Param values copied from the
// ParamStore by the executor; only Const computes anything

static void sourceF(const ForwardArgs&) {}

static void constF(const ForwardArgs& a) {
    fill(a.out.data, a.out.data + numel(a.out.shape), a.attrs.scalar);
}

// ── elementwise binary ───────────────────────────────────────────────────────
// each op gives its value and the pair (d/dx, d/dy); the loops are shared

template <class F>
static void binaryForward(const ForwardArgs& a, F f) {
    Bcast bx = bcastTo(a.out.shape, a.in[0].shape), by = bcastTo(a.out.shape, a.in[1].shape);
    int64_t n = numel(a.out.shape);
    for (int64_t i = 0; i < n; ++i) a.out.data[i] = f(a.in[0].data[at(bx, i)], a.in[1].data[at(by, i)]);
}

template <class D>
static void binaryVjp(const VjpArgs& a, D d) {
    Bcast bx = bcastTo(a.out.shape, a.in[0].shape), by = bcastTo(a.out.shape, a.in[1].shape);
    int64_t n = numel(a.out.shape);
    for (int64_t i = 0; i < n; ++i) {
        int64_t ix = at(bx, i), iy = at(by, i);
        array<float, 2> dd = d(a.in[0].data[ix], a.in[1].data[iy]);
        if (a.need & 1) a.gin[0].data[ix] += a.gout.data[i] * dd[0];
        if (a.need & 2) a.gin[1].data[iy] += a.gout.data[i] * dd[1];
    }
}

template <class D>
static void binaryJvp(const JvpArgs& a, D d) {
    Bcast bx = bcastTo(a.out.shape, a.in[0].shape), by = bcastTo(a.out.shape, a.in[1].shape);
    int64_t n = numel(a.out.shape);
    for (int64_t i = 0; i < n; ++i) {
        int64_t ix = at(bx, i), iy = at(by, i);
        array<float, 2> dd = d(a.in[0].data[ix], a.in[1].data[iy]);
        float t = 0;
        if (a.tin[0].data) t += dd[0] * a.tin[0].data[ix];
        if (a.tin[1].data) t += dd[1] * a.tin[1].data[iy];
        a.tout.data[i] = t;
    }
}

#define AG_BINARY(name, value, dx, dy)                                                              \
    static void name##F(const ForwardArgs& a) {                                                     \
        binaryForward(a, [](float x, float y) { return value; });                                   \
    }                                                                                               \
    static array<float, 2> name##D([[maybe_unused]] float x, [[maybe_unused]] float y) {            \
        return {dx, dy};                                                                            \
    }                                                                                               \
    static void name##V(const VjpArgs& a) { binaryVjp(a, name##D); }                                \
    static void name##J(const JvpArgs& a) { binaryJvp(a, name##D); }

AG_BINARY(add, x + y, 1.0f, 1.0f)
AG_BINARY(sub, x - y, 1.0f, -1.0f)
AG_BINARY(mul, x * y, y, x)
AG_BINARY(div, x / y, 1.0f / y, -x / (y * y))

#undef AG_BINARY

// ── elementwise unary ────────────────────────────────────────────────────────
// derivative written in terms of the input x, the output y and the op's
// scalar attribute s (exponent, slope)

#define AG_UNARY(name, value, deriv)                                                                \
    static void name##F(const ForwardArgs& a) {                                                     \
        [[maybe_unused]] float s = a.attrs.scalar;                                                  \
        int64_t n = numel(a.out.shape);                                                             \
        for (int64_t i = 0; i < n; ++i) {                                                           \
            float x = a.in[0].data[i];                                                              \
            a.out.data[i] = (value);                                                                \
        }                                                                                           \
    }                                                                                               \
    static void name##V(const VjpArgs& a) {                                                         \
        [[maybe_unused]] float s = a.attrs.scalar;                                                  \
        int64_t n = numel(a.out.shape);                                                             \
        for (int64_t i = 0; i < n; ++i) {                                                           \
            [[maybe_unused]] float x = a.in[0].data[i], y = a.out.data[i];                          \
            a.gin[0].data[i] += a.gout.data[i] * (deriv);                                           \
        }                                                                                           \
    }                                                                                               \
    static void name##J(const JvpArgs& a) {                                                         \
        [[maybe_unused]] float s = a.attrs.scalar;                                                  \
        int64_t n = numel(a.out.shape);                                                             \
        for (int64_t i = 0; i < n; ++i) {                                                           \
            [[maybe_unused]] float x = a.in[0].data[i], y = a.out.data[i];                          \
            a.tout.data[i] = a.tin[0].data ? a.tin[0].data[i] * (deriv) : 0.0f;                     \
        }                                                                                           \
    }

AG_UNARY(neg,       -x,                          -1.0f)
AG_UNARY(pow,       std::pow(x, s),              s * std::pow(x, s - 1.0f))
AG_UNARY(exp,       std::exp(x),                 y)
AG_UNARY(log,       std::log(x),                 1.0f / x)
AG_UNARY(relu,      x > 0 ? x : 0.0f,            x > 0 ? 1.0f : 0.0f)
AG_UNARY(leakyRelu, x > 0 ? x : s * x,           x > 0 ? 1.0f : s)
AG_UNARY(tanh,      std::tanh(x),                1.0f - y * y)
AG_UNARY(sigmoid,   1.0f / (1.0f + std::exp(-x)), y * (1.0f - y))

#undef AG_UNARY

// ── matmul ───────────────────────────────────────────────────────────────────
// [m, k] @ [k, n]. grads: ga = g @ bᵀ, gb = aᵀ @ g

static void matmul(const float* a, const float* b, float* out, int64_t m, int64_t k, int64_t n) {
    for (int64_t i = 0; i < m; ++i)
        for (int64_t j = 0; j < n; ++j) {
            float acc = 0;
            for (int64_t p = 0; p < k; ++p) acc += a[i * k + p] * b[p * n + j];
            out[i * n + j] = acc;
        }
}

static void matmulF(const ForwardArgs& a) {
    matmul(a.in[0].data, a.in[1].data, a.out.data, a.in[0].shape.dim[0], a.in[0].shape.dim[1], a.in[1].shape.dim[1]);
}

static void matmulV(const VjpArgs& a) {
    int64_t m = a.in[0].shape.dim[0], k = a.in[0].shape.dim[1], n = a.in[1].shape.dim[1];
    const float *x = a.in[0].data, *w = a.in[1].data, *g = a.gout.data;
    if (a.need & 1)
        for (int64_t i = 0; i < m; ++i)
            for (int64_t p = 0; p < k; ++p) {
                float acc = 0;
                for (int64_t j = 0; j < n; ++j) acc += g[i * n + j] * w[p * n + j];
                a.gin[0].data[i * k + p] += acc;
            }
    if (a.need & 2)
        for (int64_t p = 0; p < k; ++p)
            for (int64_t j = 0; j < n; ++j) {
                float acc = 0;
                for (int64_t i = 0; i < m; ++i) acc += x[i * k + p] * g[i * n + j];
                a.gin[1].data[p * n + j] += acc;
            }
}

static void matmulJ(const JvpArgs& a) {
    int64_t m = a.in[0].shape.dim[0], k = a.in[0].shape.dim[1], n = a.in[1].shape.dim[1];
    for (int64_t i = 0; i < m; ++i)
        for (int64_t j = 0; j < n; ++j) {
            float acc = 0;
            for (int64_t p = 0; p < k; ++p) {
                if (a.tin[0].data) acc += a.tin[0].data[i * k + p] * a.in[1].data[p * n + j];
                if (a.tin[1].data) acc += a.in[0].data[i * k + p] * a.tin[1].data[p * n + j];
            }
            a.tout.data[i * n + j] = acc;
        }
}

// ── reductions ───────────────────────────────────────────────────────────────

static void reduceShape(const Shape& in, const Attrs& attrs, int64_t* outer, int64_t* len, int64_t* inner) {
    int32_t axis;
    if (attrs.axis == kAllAxes || !resolveAxis(attrs.axis, in.rank, &axis)) {
        *outer = *inner = 1;
        *len = numel(in);
        return;
    }
    around(in, axis, outer, len, inner);
}

static void reduce(const float* in, float* out, const Shape& s, const Attrs& attrs, bool mean) {
    int64_t outer, len, inner;
    reduceShape(s, attrs, &outer, &len, &inner);
    for (int64_t o = 0; o < outer; ++o)
        for (int64_t i = 0; i < inner; ++i) {
            double acc = 0;
            for (int64_t k = 0; k < len; ++k) acc += in[(o * len + k) * inner + i];
            out[o * inner + i] = toFloat(mean ? acc / double(len) : acc);
        }
}

static void reduceV(const VjpArgs& a, bool mean) {
    int64_t outer, len, inner;
    reduceShape(a.in[0].shape, a.attrs, &outer, &len, &inner);
    float scale = mean ? 1.0f / float(len) : 1.0f;
    for (int64_t o = 0; o < outer; ++o)
        for (int64_t i = 0; i < inner; ++i)
            for (int64_t k = 0; k < len; ++k) a.gin[0].data[(o * len + k) * inner + i] += a.gout.data[o * inner + i] * scale;
}

static void reduceJ(const JvpArgs& a, bool mean) {
    if (!a.tin[0].data) { fill(a.tout.data, a.tout.data + numel(a.tout.shape), 0.0f); return; }
    reduce(a.tin[0].data, a.tout.data, a.in[0].shape, a.attrs, mean);
}

static void sumF(const ForwardArgs& a) { reduce(a.in[0].data, a.out.data, a.in[0].shape, a.attrs, false); }
static void sumV(const VjpArgs& a) { reduceV(a, false); }
static void sumJ(const JvpArgs& a) { reduceJ(a, false); }
static void meanF(const ForwardArgs& a) { reduce(a.in[0].data, a.out.data, a.in[0].shape, a.attrs, true); }
static void meanV(const VjpArgs& a) { reduceV(a, true); }
static void meanJ(const JvpArgs& a) { reduceJ(a, true); }

// ── shape ops ────────────────────────────────────────────────────────────────
// contiguous storage means reshape is a copy and its derivative is the identity

static void reshapeF(const ForwardArgs& a) { copy_n(a.in[0].data, numel(a.out.shape), a.out.data); }

static void reshapeV(const VjpArgs& a) {
    int64_t n = numel(a.out.shape);
    for (int64_t i = 0; i < n; ++i) a.gin[0].data[i] += a.gout.data[i];
}

static void reshapeJ(const JvpArgs& a) {
    int64_t n = numel(a.out.shape);
    for (int64_t i = 0; i < n; ++i) a.tout.data[i] = a.tin[0].data ? a.tin[0].data[i] : 0.0f;
}

static void transposeF(const ForwardArgs& a) {
    int64_t r = a.in[0].shape.dim[0], c = a.in[0].shape.dim[1];
    for (int64_t i = 0; i < r; ++i)
        for (int64_t j = 0; j < c; ++j) a.out.data[j * r + i] = a.in[0].data[i * c + j];
}

static void transposeV(const VjpArgs& a) {
    int64_t r = a.in[0].shape.dim[0], c = a.in[0].shape.dim[1];
    for (int64_t i = 0; i < r; ++i)
        for (int64_t j = 0; j < c; ++j) a.gin[0].data[i * c + j] += a.gout.data[j * r + i];
}

static void transposeJ(const JvpArgs& a) {
    int64_t r = a.in[0].shape.dim[0], c = a.in[0].shape.dim[1];
    for (int64_t i = 0; i < r; ++i)
        for (int64_t j = 0; j < c; ++j) a.tout.data[j * r + i] = a.tin[0].data ? a.tin[0].data[i * c + j] : 0.0f;
}

// ── softmax ──────────────────────────────────────────────────────────────────
// along one axis (default the last). max is subtracted before exp.
// vjp: gx = y ⊙ (g − Σ g ⊙ y), jvp: tx ↦ y ⊙ (t − Σ t ⊙ y)

static void softmaxShape(const Shape& s, const Attrs& attrs, int64_t* outer, int64_t* len, int64_t* inner) {
    int32_t axis = s.rank - 1;
    if (attrs.axis != kAllAxes) resolveAxis(attrs.axis, s.rank, &axis);
    around(s, axis, outer, len, inner);
}

static void softmaxF(const ForwardArgs& a) {
    int64_t outer, len, inner;
    softmaxShape(a.in[0].shape, a.attrs, &outer, &len, &inner);
    const float* x = a.in[0].data;
    float* y = a.out.data;
    for (int64_t o = 0; o < outer; ++o)
        for (int64_t i = 0; i < inner; ++i) {
            auto idx = [&](int64_t k) { return (o * len + k) * inner + i; };
            float m = -INFINITY;
            for (int64_t k = 0; k < len; ++k) m = max(m, x[idx(k)]);
            double total = 0;
            for (int64_t k = 0; k < len; ++k) total += (y[idx(k)] = std::exp(x[idx(k)] - m));
            for (int64_t k = 0; k < len; ++k) y[idx(k)] = float(y[idx(k)] / total);
        }
}

// shared by vjp (v = g, accumulate) and jvp (v = t, overwrite)
static void softmaxBack(const Shape& s, const Attrs& attrs, const float* y, const float* v, float* out, bool accumulate) {
    int64_t outer, len, inner;
    softmaxShape(s, attrs, &outer, &len, &inner);
    for (int64_t o = 0; o < outer; ++o)
        for (int64_t i = 0; i < inner; ++i) {
            auto idx = [&](int64_t k) { return (o * len + k) * inner + i; };
            double dot = 0;
            for (int64_t k = 0; k < len; ++k) dot += double(v[idx(k)]) * y[idx(k)];
            for (int64_t k = 0; k < len; ++k) {
                float r = y[idx(k)] * (v[idx(k)] - toFloat(dot));
                out[idx(k)] = accumulate ? out[idx(k)] + r : r;
            }
        }
}

static void softmaxV(const VjpArgs& a) { softmaxBack(a.out.shape, a.attrs, a.out.data, a.gout.data, a.gin[0].data, true); }

static void softmaxJ(const JvpArgs& a) {
    if (!a.tin[0].data) { fill(a.tout.data, a.tout.data + numel(a.tout.shape), 0.0f); return; }
    softmaxBack(a.out.shape, a.attrs, a.out.data, a.tin[0].data, a.tout.data, false);
}

// ── losses ───────────────────────────────────────────────────────────────────
// mse = mean((p − t)²). both inputs get a gradient, so a learned target works

static void mseF(const ForwardArgs& a) {
    int64_t n = numel(a.in[0].shape);
    double acc = 0;
    for (int64_t i = 0; i < n; ++i) {
        double d = double(a.in[0].data[i]) - a.in[1].data[i];
        acc += d * d;
    }
    a.out.data[0] = toFloat(acc / double(n));
}

static void mseV(const VjpArgs& a) {
    int64_t n = numel(a.in[0].shape);
    float k = a.gout.data[0] * 2.0f / float(n);
    for (int64_t i = 0; i < n; ++i) {
        float d = a.in[0].data[i] - a.in[1].data[i];
        if (a.need & 1) a.gin[0].data[i] += k * d;
        if (a.need & 2) a.gin[1].data[i] -= k * d;
    }
}

static void mseJ(const JvpArgs& a) {
    int64_t n = numel(a.in[0].shape);
    double acc = 0;
    for (int64_t i = 0; i < n; ++i) {
        double d = double(a.in[0].data[i]) - a.in[1].data[i];
        double t = (a.tin[0].data ? a.tin[0].data[i] : 0.0) - (a.tin[1].data ? a.tin[1].data[i] : 0.0);
        acc += 2.0 * d * t;
    }
    a.tout.data[0] = toFloat(acc / double(n));
}

// cross-entropy of softmax(logits) against target probabilities along the last
// axis, averaged over rows. computed with log-sum-exp, never softmax then log.
// for row r: loss_r = Σ_c t_c (lse − z_c), d/dz_c = p_c Σt − t_c, d/dt_c = lse − z_c

static float rowLse(const float* z, int64_t c) {
    float m = -INFINITY;
    for (int64_t k = 0; k < c; ++k) m = max(m, z[k]);
    double total = 0;
    for (int64_t k = 0; k < c; ++k) total += std::exp(double(z[k]) - m);
    return m + float(std::log(total));
}

static void ceF(const ForwardArgs& a) {
    int64_t c = a.in[0].shape.dim[a.in[0].shape.rank - 1], rows = numel(a.in[0].shape) / c;
    double acc = 0;
    for (int64_t r = 0; r < rows; ++r) {
        const float *z = a.in[0].data + r * c, *t = a.in[1].data + r * c;
        float lse = rowLse(z, c);
        for (int64_t k = 0; k < c; ++k) acc += double(t[k]) * (lse - z[k]);
    }
    a.out.data[0] = toFloat(acc / double(rows));
}

static void ceV(const VjpArgs& a) {
    int64_t c = a.in[0].shape.dim[a.in[0].shape.rank - 1], rows = numel(a.in[0].shape) / c;
    float k0 = a.gout.data[0] / float(rows);
    for (int64_t r = 0; r < rows; ++r) {
        const float *z = a.in[0].data + r * c, *t = a.in[1].data + r * c;
        float lse = rowLse(z, c), tsum = 0;
        for (int64_t k = 0; k < c; ++k) tsum += t[k];
        for (int64_t k = 0; k < c; ++k) {
            if (a.need & 1) a.gin[0].data[r * c + k] += k0 * (std::exp(z[k] - lse) * tsum - t[k]);
            if (a.need & 2) a.gin[1].data[r * c + k] += k0 * (lse - z[k]);
        }
    }
}

static void ceJ(const JvpArgs& a) {
    int64_t c = a.in[0].shape.dim[a.in[0].shape.rank - 1], rows = numel(a.in[0].shape) / c;
    double acc = 0;
    for (int64_t r = 0; r < rows; ++r) {
        const float *z = a.in[0].data + r * c, *t = a.in[1].data + r * c;
        float lse = rowLse(z, c), tsum = 0;
        for (int64_t k = 0; k < c; ++k) tsum += t[k];
        for (int64_t k = 0; k < c; ++k) {
            if (a.tin[0].data) acc += double(std::exp(z[k] - lse) * tsum - t[k]) * a.tin[0].data[r * c + k];
            if (a.tin[1].data) acc += double(lse - z[k]) * a.tin[1].data[r * c + k];
        }
    }
    a.tout.data[0] = toFloat(acc / double(rows));
}

// ── the op table ─────────────────────────────────────────────────────────────
// rows are in the same order as enum Op; a test checks this.
// AG_OP(op, name, symbol, category, min, max, saves, infer, kernels, scalarize)
// where `kernels` names the F/V/J triple

#define AG_OP(op, name, symbol, category, lo, hi, saves, infer, k, scalar) \
    OpInfo{Op::op, name, symbol, Category::category, lo, hi, saves, infer, k##F, k##V, k##J, scalar}
#define AG_SOURCE(op, name, symbol, forward) \
    OpInfo{Op::op, name, symbol, Category::Source, 0, 0, kSavesNothing, inferSource, forward, nullptr, nullptr, nullptr}

static const OpInfo kOps[] = {
    AG_SOURCE(Input, "input", "x", sourceF),
    AG_SOURCE(Param, "param", "w", sourceF),
    AG_SOURCE(Const, "const", "c", constF),

    AG_OP(Add, "add", "+",   Arithmetic, 2, 2, kSavesNothing, inferBinary, add, scalarizeElementwise),
    AG_OP(Sub, "sub", "-",   Arithmetic, 2, 2, kSavesNothing, inferBinary, sub, scalarizeElementwise),
    AG_OP(Mul, "mul", "*",   Arithmetic, 2, 2, kSavesInputs,  inferBinary, mul, scalarizeElementwise),
    AG_OP(Div, "div", "/",   Arithmetic, 2, 2, kSavesInputs,  inferBinary, div, scalarizeElementwise),
    AG_OP(Neg, "neg", "-x",  Arithmetic, 1, 1, kSavesNothing, inferSame,   neg, scalarizeElementwise),
    AG_OP(Pow, "pow", "x^n", Arithmetic, 1, 1, kSavesInputs,  inferSame,   pow, scalarizeElementwise),
    AG_OP(Exp, "exp", "exp", Arithmetic, 1, 1, kSavesOutput,  inferSame,   exp, scalarizeElementwise),
    AG_OP(Log, "log", "log", Arithmetic, 1, 1, kSavesInputs,  inferSame,   log, scalarizeElementwise),

    AG_OP(MatMul, "matmul", "@", Linear, 2, 2, kSavesInputs, inferMatMul, matmul, scalarizeMatMul),

    AG_OP(Sum,  "sum",  "sum",  Reduce, 1, 1, kSavesNothing, inferReduce, sum,  scalarizeReduce),
    AG_OP(Mean, "mean", "mean", Reduce, 1, 1, kSavesNothing, inferReduce, mean, scalarizeReduce),

    AG_OP(Reshape,   "reshape",   "reshape", ShapeOp, 1, 1, kSavesNothing, inferReshape,   reshape,   scalarizeReshape),
    AG_OP(Transpose, "transpose", "T",       ShapeOp, 1, 1, kSavesNothing, inferTranspose, transpose, scalarizeTranspose),

    AG_OP(Relu,      "relu",       "relu",    Activation, 1, 1, kSavesOutput, inferSame,    relu,      scalarizeElementwise),
    AG_OP(LeakyRelu, "leaky_relu", "lrelu",   Activation, 1, 1, kSavesInputs, inferSame,    leakyRelu, scalarizeElementwise),
    AG_OP(Tanh,      "tanh",       "tanh",    Activation, 1, 1, kSavesOutput, inferSame,    tanh,      scalarizeElementwise),
    AG_OP(Sigmoid,   "sigmoid",    "sigmoid", Activation, 1, 1, kSavesOutput, inferSame,    sigmoid,   scalarizeElementwise),
    AG_OP(Softmax,   "softmax",    "softmax", Activation, 1, 1, kSavesOutput, inferSoftmax, softmax,   scalarizeSoftmax),

    AG_OP(MseLoss,             "mse_loss",              "mse", Loss, 2, 2, kSavesInputs, inferLoss,         mse, scalarizeMse),
    AG_OP(SoftmaxCrossEntropy, "softmax_cross_entropy", "ce",  Loss, 2, 2, kSavesInputs, inferCrossEntropy, ce,  scalarizeCrossEntropy),
};

#undef AG_OP
#undef AG_SOURCE

static_assert(sizeof(kOps) / sizeof(kOps[0]) == kOpCount, "op table and enum Op disagree");

const OpInfo& opInfo(Op op) { return kOps[static_cast<int>(op)]; }

span<const OpInfo> allOps() { return span<const OpInfo>(kOps, kOpCount); }

const OpInfo* findOp(string_view name) {
    for (const OpInfo& info : kOps)
        if (name == info.name) return &info;
    return nullptr;
}

} // namespace ag
