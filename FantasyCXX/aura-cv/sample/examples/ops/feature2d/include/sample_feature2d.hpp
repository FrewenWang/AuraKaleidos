#ifndef AURA_SAMPLE_OPS_FEATURE2D_HPP__
#define AURA_SAMPLE_OPS_FEATURE2D_HPP__

#include "sample_ops.hpp"

aura::Status CannySampleTest(aura::Context *ctx, aura::TargetType type);

aura::Status FastSampleTest(aura::Context *ctx, aura::TargetType type);

aura::Status HarrisSampleTest(aura::Context *ctx, aura::TargetType type);

aura::Status TomasiSampleTest(aura::Context *ctx, aura::TargetType type);

#endif // AURA_SAMPLE_OPS_FEATURE2D_HPP__

