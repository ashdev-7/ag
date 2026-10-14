#pragma once

// data — small synthetic datasets generated from a seed, and batching

#include "ag/core.hpp"
#include "ag/rng.hpp"

#include <vector>

namespace ag {

enum class DatasetKind : uint8_t { Moons, Circles, Spirals, Xor, Gaussians, Sine1D, None };
enum class Split : uint8_t { Train, Val };

struct DatasetConfig {
    DatasetKind kind = DatasetKind::Moons;
    int32_t     n = 500;
    float       noise = 0.1f;
    float       valFraction = 0.2f;
    uint64_t    seed = 0;
};

struct Dataset {
    int32_t            n = 0;
    int32_t            nTrain = 0;     // rows [0, nTrain) train, the rest val
    int32_t            xDim = 0;
    int32_t            yDim = 0;       // 1 for regression, numClasses for one-hot
    int32_t            numClasses = 0; // 0 for regression
    std::vector<float> x;              // n * xDim, row-major
    std::vector<float> y;              // n * yDim
};

Status makeDataset(const DatasetConfig& cfg, Dataset* out);

struct Batcher {
    std::vector<int32_t> perm;
    int32_t              pos = 0;
    int32_t              epoch = 0;
    Rng                  rng;
};

void batcherInit(Batcher* b, const Dataset& d, uint64_t seed);

// fill one training batch, reshuffling at each epoch boundary. batchSize must
// not exceed d.nTrain
Status nextBatch(Batcher* b, const Dataset& d, int32_t batchSize,
                 std::span<float> xOut, std::span<float> yOut);

} // namespace ag
