#ifndef AURA_SAMPLE_OPS_FILTER_HPP__
#define AURA_SAMPLE_OPS_FILTER_HPP__

#include "sample_ops.hpp"

aura::Status BilateralSampleTest(aura::Context *ctx, aura::TargetType type);

aura::Status BoxfilterSampleTest(aura::Context *ctx, aura::TargetType type);

aura::Status Filter2dSampleTest(aura::Context *ctx, aura::TargetType type);

aura::Status GaussianSampleTest(aura::Context *ctx, aura::TargetType type);

aura::Status LaplacianSampleTest(aura::Context *ctx, aura::TargetType type);

aura::Status MedianSampleTest(aura::Context *ctx, aura::TargetType type);

aura::Status SobelSampleTest(aura::Context *ctx, aura::TargetType type);

#endif // AURA_SAMPLE_OPS_FILTER_HPP__