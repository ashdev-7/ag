#include "ag/data.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

using namespace std;

namespace ag {

// generators in the spirit of TensorFlow Playground's dataset.ts. inputs are
// scaled to roughly [-1, 1]; classification targets are one-hot over two
// classes. rows are shuffled once, then the last valFraction become the
// validation split

static constexpr float kPi = 3.14159265f;

// noise is passed in already drawn: two rng calls in one argument list run in
// an order that differs between compilers, and so would the dataset
static void push(Dataset* d, float x0, float x1, int cls, float n0 = 0, float n1 = 0) {
    d->x.push_back(x0 + n0);
    d->x.push_back(x1 + n1);
    d->y.push_back(cls == 0 ? 1.0f : 0.0f);
    d->y.push_back(cls == 1 ? 1.0f : 0.0f);
}

Status makeDataset(const DatasetConfig& cfg, Dataset* out) {
    if (!out) return makeError(Code::InvalidArgument, "makeDataset: null output");
    if (static_cast<int>(cfg.kind) > static_cast<int>(DatasetKind::None))
        return makeError(Code::InvalidArgument, "unknown dataset");
    if (cfg.kind != DatasetKind::None && (cfg.n < 4 || cfg.n > 20000))
        return makeError(Code::InvalidArgument, "dataset size must be between 4 and 20000");
    if (!(cfg.noise >= 0.0f && cfg.noise <= 2.0f)) return makeError(Code::InvalidArgument, "noise must be between 0 and 2");
    if (!(cfg.valFraction >= 0.0f && cfg.valFraction <= 0.5f))
        return makeError(Code::InvalidArgument, "validation fraction must be between 0 and 0.5");

    Dataset d;
    Rng rng;
    rngSeed(&rng, cfg.seed, 7);
    auto noise = [&] { return cfg.noise * rngNormal(&rng); };
    int32_t n = cfg.n, half = n / 2;

    switch (cfg.kind) {
        case DatasetKind::None:
            *out = Dataset{};
            return {};
        case DatasetKind::Moons:
            // two interleaved half circles, centred and scaled into [-1, 1]
            for (int32_t i = 0; i < n; ++i) {
                int cls = i < half ? 0 : 1;
                float t = kPi * rngUniform(&rng);
                float x0 = cls == 0 ? cos(t) : 1.0f - cos(t);
                float x1 = cls == 0 ? sin(t) : 0.5f - sin(t);
                float n0 = noise();
                float n1 = noise();
                push(&d, (x0 - 0.5f) / 1.6f, (x1 - 0.25f) / 1.6f, cls, n0 / 1.6f, n1 / 1.6f);
            }
            break;
        case DatasetKind::Circles:
            for (int32_t i = 0; i < n; ++i) {
                int cls = i < half ? 0 : 1;
                float r = cls == 0 ? rngUniform(&rng, 0.0f, 0.4f) : rngUniform(&rng, 0.6f, 0.9f);
                float a = 2 * kPi * rngUniform(&rng);
                float n0 = 0.5f * noise();
                float n1 = 0.5f * noise();
                push(&d, r * cos(a), r * sin(a), cls, n0, n1);
            }
            break;
        case DatasetKind::Spirals:
            for (int32_t i = 0; i < n; ++i) {
                int cls = i < half ? 0 : 1;
                float t = rngUniform(&rng, 0.0f, 1.0f);
                float r = 0.9f * t, a = 3.5f * kPi * t + (cls ? kPi : 0.0f);
                float n0 = 0.3f * noise();
                float n1 = 0.3f * noise();
                push(&d, r * cos(a), r * sin(a), cls, n0, n1);
            }
            break;
        case DatasetKind::Xor:
            for (int32_t i = 0; i < n; ++i) {
                float x0 = rngUniform(&rng, -1, 1);
                float x1 = rngUniform(&rng, -1, 1);
                int cls = x0 * x1 >= 0 ? 0 : 1;
                // keep a small margin around the axes so the classes are separable
                x0 += x0 > 0 ? 0.05f : -0.05f;
                x1 += x1 > 0 ? 0.05f : -0.05f;
                float n0 = 0.5f * noise();
                float n1 = 0.5f * noise();
                push(&d, x0, x1, cls, n0, n1);
            }
            break;
        case DatasetKind::Gaussians:
            for (int32_t i = 0; i < n; ++i) {
                int cls = i < half ? 0 : 1;
                float c = cls ? 0.45f : -0.45f;
                float n0 = (0.15f + cfg.noise) * rngNormal(&rng);
                float n1 = (0.15f + cfg.noise) * rngNormal(&rng);
                push(&d, c, c, cls, n0, n1);
            }
            break;
        case DatasetKind::Sine1D:
            for (int32_t i = 0; i < n; ++i) {
                float x = rngUniform(&rng, -1, 1);
                d.x.push_back(x);
                d.y.push_back(sin(kPi * x) + 0.5f * noise());
            }
            break;
    }

    bool regression = cfg.kind == DatasetKind::Sine1D;
    d.n = n;
    d.xDim = regression ? 1 : 2;
    d.yDim = regression ? 1 : 2;
    d.numClasses = regression ? 0 : 2;

    // shuffle rows once so both classes appear in both splits
    vector<int32_t> perm(n);
    iota(perm.begin(), perm.end(), 0);
    rngShuffle(&rng, perm);
    vector<float> x(d.x.size()), y(d.y.size());
    for (int32_t i = 0; i < n; ++i) {
        copy_n(d.x.begin() + perm[i] * d.xDim, d.xDim, x.begin() + i * d.xDim);
        copy_n(d.y.begin() + perm[i] * d.yDim, d.yDim, y.begin() + i * d.yDim);
    }
    d.x = std::move(x);
    d.y = std::move(y);
    d.nTrain = n - static_cast<int32_t>(lround(n * cfg.valFraction));
    *out = std::move(d);
    return {};
}

void batcherInit(Batcher* b, const Dataset& d, uint64_t seed) {
    b->perm.resize(d.nTrain);
    iota(b->perm.begin(), b->perm.end(), 0);
    rngSeed(&b->rng, seed, 11);
    rngShuffle(&b->rng, b->perm);
    b->pos = 0;
    b->epoch = 0;
}

Status nextBatch(Batcher* b, const Dataset& d, int32_t batchSize, span<float> xOut, span<float> yOut) {
    if (batchSize < 1 || batchSize > d.nTrain)
        return makeError(Code::InvalidArgument, "batch size must be between 1 and the number of training rows (" +
                                                to_string(d.nTrain) + ")");
    if (xOut.size() != size_t(batchSize) * d.xDim || yOut.size() != size_t(batchSize) * d.yDim)
        return makeError(Code::InvalidArgument, "batch buffers have the wrong size");
    for (int32_t r = 0; r < batchSize; ++r) {
        if (b->pos == d.nTrain) {
            rngShuffle(&b->rng, b->perm);
            b->pos = 0;
            ++b->epoch;
        }
        int32_t row = b->perm[b->pos++];
        copy_n(d.x.begin() + row * d.xDim, d.xDim, xOut.begin() + r * d.xDim);
        copy_n(d.y.begin() + row * d.yDim, d.yDim, yOut.begin() + r * d.yDim);
    }
    return {};
}

} // namespace ag
