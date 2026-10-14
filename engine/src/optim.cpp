#include "ag/optim.hpp"

#include <cmath>

using namespace std;

namespace ag {

static bool in01(float x) { return x >= 0.0f && x < 1.0f; }

Status optimValidate(const OptimConfig& c) {
    auto bad = [](const string& field, const string& why) { return makeError(Code::InvalidArgument, field + " " + why); };
    if (static_cast<int>(c.kind) > static_cast<int>(OptimKind::AdamW)) return bad("optimizer", "is unknown");
    if (static_cast<int>(c.lr.kind) > static_cast<int>(LrKind::WarmupCosine)) return bad("schedule", "is unknown");
    if (!(c.lr.baseLr > 0.0f && c.lr.baseLr <= 100.0f)) return bad("learning rate", "must be in (0, 100]");
    if (!(c.lr.minLr >= 0.0f && c.lr.minLr <= c.lr.baseLr)) return bad("minimum learning rate", "must be in [0, learning rate]");
    if (!(c.lr.gamma > 0.0f && c.lr.gamma <= 1.0f)) return bad("decay factor", "must be in (0, 1]");
    if (c.lr.period < 1) return bad("decay period", "must be at least 1");
    if (c.lr.totalSteps < 1) return bad("schedule length", "must be at least 1");
    if (c.lr.warmupSteps < 0) return bad("warmup", "cannot be negative");
    if (!in01(c.momentum)) return bad("momentum", "must be in [0, 1)");
    if (!in01(c.beta1)) return bad("beta1", "must be in [0, 1)");
    if (!in01(c.beta2)) return bad("beta2", "must be in [0, 1)");
    if (!(c.eps > 0.0f && c.eps < 1.0f)) return bad("epsilon", "must be in (0, 1)");
    if (!(c.weightDecay >= 0.0f && c.weightDecay < 1.0f)) return bad("weight decay", "must be in [0, 1)");
    if (!(c.clipNorm >= 0.0f) || !isfinite(c.clipNorm)) return bad("clip norm", "cannot be negative");
    return {};
}

Status optimInit(const OptimConfig& cfg, const ParamStore& ps, Optimizer* out) {
    AG_TRY(optimValidate(cfg));
    Optimizer o;
    o.cfg = cfg;
    o.m.assign(ps.values.size(), 0.0f);
    o.v.assign(ps.values.size(), 0.0f);
    *out = std::move(o);
    return {};
}

// warmup (if any) is linear from 0 and applies to every schedule

float lrAt(const LrSchedule& s, int64_t step) {
    if (s.warmupSteps > 0 && step < s.warmupSteps) return s.baseLr * float(step + 1) / float(s.warmupSteps);
    int64_t t = step - s.warmupSteps;
    switch (s.kind) {
        case LrKind::Constant:
            return s.baseLr;
        case LrKind::StepDecay:
            return s.baseLr * std::pow(s.gamma, float(t / s.period));
        case LrKind::Cosine:
        case LrKind::WarmupCosine: {
            int64_t horizon = max<int64_t>(1, s.totalSteps - s.warmupSteps);
            float frac = float(min(t, horizon)) / float(horizon);
            return s.minLr + 0.5f * (s.baseLr - s.minLr) * (1.0f + std::cos(3.14159265f * frac));
        }
    }
    return s.baseLr;
}

float clipScale(float gradNorm, float clipNorm) {
    if (clipNorm <= 0.0f || !(gradNorm > clipNorm)) return 1.0f;
    return clipNorm / gradNorm;
}

// the update rules follow torch.optim's conventions so the tests can compare
// against it: momentum buffers accumulate the raw gradient (no dampening),
// rmsprop's smoothing constant is beta2, adam corrects bias with t + 1.
// written out one optimizer at a time, no shared cleverness

void optimUpdate(Optimizer* o, const ParamPlan& plan, span<float> w, span<const float> grad, float lr, float scale) {
    const OptimConfig& c = o->cfg;
    float* m = o->m.data() + plan.offset;
    float* v = o->v.data() + plan.offset;
    float t = float(o->t + 1);
    for (size_t i = 0; i < w.size(); ++i) {
        float g = grad[i] * scale;
        if (c.kind != OptimKind::AdamW) g += c.weightDecay * w[i];
        switch (c.kind) {
            case OptimKind::Sgd:
                w[i] -= lr * g;
                break;
            case OptimKind::Momentum:
                m[i] = c.momentum * m[i] + g;
                w[i] -= lr * m[i];
                break;
            case OptimKind::Nesterov:
                m[i] = c.momentum * m[i] + g;
                w[i] -= lr * (g + c.momentum * m[i]);
                break;
            case OptimKind::RmsProp:
                v[i] = c.beta2 * v[i] + (1.0f - c.beta2) * g * g;
                w[i] -= lr * g / (std::sqrt(v[i]) + c.eps);
                break;
            case OptimKind::Adam:
            case OptimKind::AdamW: {
                if (c.kind == OptimKind::AdamW) w[i] -= lr * c.weightDecay * w[i];
                m[i] = c.beta1 * m[i] + (1.0f - c.beta1) * g;
                v[i] = c.beta2 * v[i] + (1.0f - c.beta2) * g * g;
                float mhat = m[i] / (1.0f - std::pow(c.beta1, t));
                float vhat = v[i] / (1.0f - std::pow(c.beta2, t));
                w[i] -= lr * mhat / (std::sqrt(vhat) + c.eps);
                break;
            }
        }
    }
}

void optimEndStep(Optimizer* o) { ++o->t; }

} // namespace ag
