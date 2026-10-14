#include "ag/rng.hpp"

#include <cmath>
#include <utility>

namespace ag {

// PCG32 (O'Neill), XSH-RR variant

uint32_t rngU32(Rng* rng) {
    uint64_t old = rng->state;
    rng->state = old * 6364136223846793005ULL + rng->inc;
    uint32_t xorshifted = static_cast<uint32_t>(((old >> 18u) ^ old) >> 27u);
    uint32_t rot = static_cast<uint32_t>(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((32u - rot) & 31u));
}

void rngSeed(Rng* rng, uint64_t seed, uint64_t stream) {
    rng->state = 0;
    rng->inc = (stream << 1u) | 1u;
    rngU32(rng);
    rng->state += seed;
    rngU32(rng);
}

// rejection sampling so every value in [0, bound) is equally likely

uint32_t rngBelow(Rng* rng, uint32_t bound) {
    if (bound == 0) return 0;
    uint32_t threshold = (0u - bound) % bound;
    for (;;) {
        uint32_t r = rngU32(rng);
        if (r >= threshold) return r % bound;
    }
}

// top 24 bits: every result is exactly representable in a float

float rngUniform(Rng* rng) {
    return static_cast<float>(rngU32(rng) >> 8) * (1.0f / 16777216.0f);
}

float rngUniform(Rng* rng, float lo, float hi) {
    return lo + (hi - lo) * rngUniform(rng);
}

// Box-Muller. one of the pair is thrown away so the generator has no hidden
// state beyond PCG's; 1 - u keeps the log argument in (0, 1]

float rngNormal(Rng* rng) {
    float u1 = 1.0f - rngUniform(rng);
    float u2 = rngUniform(rng);
    return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.28318530718f * u2);
}

// Fisher-Yates

void rngShuffle(Rng* rng, std::span<int32_t> items) {
    for (size_t i = items.size(); i > 1; --i) {
        size_t j = rngBelow(rng, static_cast<uint32_t>(i));
        std::swap(items[i - 1], items[j]);
    }
}

} // namespace ag
