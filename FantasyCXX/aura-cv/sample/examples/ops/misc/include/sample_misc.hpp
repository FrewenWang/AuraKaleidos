#ifndef AURA_SAMPLE_OPS_MISC_HPP__
#define AURA_SAMPLE_OPS_MISC_HPP__

#include "sample_ops.hpp"

aura::Status AdaptiveThresholdSampleTest(aura::Context *ctx, aura::TargetType type);

aura::Status HoughCirclesSampleTest(aura::Context *ctx, aura::TargetType type);

aura::Status HoughLinesSampleTest(aura::Context *ctx, aura::TargetType type);

aura::Status MipiSampleTest(aura::Context *ctx, aura::TargetType type);

aura::Status ThresholdSampleTest(aura::Context *ctx, aura::TargetType type);

#endif // AURA_SAMPLE_OPS_MISC_HPP__