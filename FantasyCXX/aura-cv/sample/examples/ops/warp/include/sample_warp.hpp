#ifndef AURA_SAMPLE_OPS_WARP_HPP__
#define AURA_SAMPLE_OPS_WARP_HPP__

#include "sample_ops.hpp"

aura::Status RemapSampleTest(aura::Context *ctx, aura::TargetType type);

aura::Status WarpAffineSampleTest(aura::Context *ctx, aura::TargetType type);

aura::Status WarpPerspectiveSampleTest(aura::Context *ctx, aura::TargetType type);

#endif // AURA_SAMPLE_OPS_WARP_HPP__

