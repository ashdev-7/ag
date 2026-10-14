#pragma once

// rng — our own PCG32 so a seed gives the same stream in every build. the
// standard library's distributions differ between toolchains, so they are
// not used anywhere in the engine

#include <cstdint>
#include <span>

namespace ag {

struct Rng {
    uint64_t state = 0x853c49e6748fea9bULL;
    uint64_t inc   = 0xda3e39cb94b95bdbULL;
};

void     rngSeed(Rng* rng, uint64_t seed, uint64_t stream = 1);
uint32_t rngU32(Rng* rng);
uint32_t rngBelow(Rng* rng, uint32_t bound);     // unbiased, [0, bound)
float    rngUniform(Rng* rng);                   // [0, 1)
float    rngUniform(Rng* rng, float lo, float hi);
float    rngNormal(Rng* rng);                    // mean 0, std 1
void     rngShuffle(Rng* rng, std::span<int32_t> items);

} // namespace ag
