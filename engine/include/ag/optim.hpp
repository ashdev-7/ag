#pragma once

// optim — turns a gradient into a parameter update. state buffers (momentum,
// second moment) are laid out exactly like ParamStore::values so the UI can
// show them next to the parameter they belong to

#include "ag/exec.hpp"

namespace ag {

enum class OptimKind : uint8_t { Sgd, Momentum, Nesterov, RmsProp, Adam, AdamW };
enum class LrKind : uint8_t { Constant, StepDecay, Cosine, WarmupCosine };

struct LrSchedule {
    LrKind  kind = LrKind::Constant;
    float   baseLr = 0.03f;
    float   minLr = 0.0f;       // Cosine floor
    float   gamma = 0.5f;       // StepDecay factor
    int32_t period = 100;       // StepDecay interval
    int32_t warmupSteps = 0;
    int32_t totalSteps = 1000;  // Cosine horizon
};

struct OptimConfig {
    OptimKind  kind = OptimKind::Sgd;
    LrSchedule lr;
    float      momentum = 0.9f;
    float      beta1 = 0.9f;
    float      beta2 = 0.999f;
    float      eps = 1e-8f;
    float      weightDecay = 0.0f;   // L2 for all but AdamW, decoupled for AdamW
    float      clipNorm = 0.0f;      // 0 disables global-norm clipping
};

struct Optimizer {
    OptimConfig        cfg;
    int64_t            t = 0;        // completed updates
    std::vector<float> m;            // first moment / velocity
    std::vector<float> v;            // second moment
};

Status optimValidate(const OptimConfig& cfg);
Status optimInit(const OptimConfig& cfg, const ParamStore& ps, Optimizer* out);

float lrAt(const LrSchedule& s, int64_t step);

// scale factor in (0, 1] to apply to every gradient this step
float clipScale(float gradNorm, float clipNorm);

// one parameter tensor. called once per Update step so the stepper can show
// parameters changing one at a time
void optimUpdate(Optimizer* o, const ParamPlan& plan, std::span<float> value,
                 std::span<const float> grad, float lr, float scale);

// call once after the last Update step of a training step
void optimEndStep(Optimizer* o);

} // namespace ag
