#ifndef AURA_SAMPLE_OPS_MATRIX_HPP__
#define AURA_SAMPLE_OPS_MATRIX_HPP__

#include "sample_ops.hpp"

aura::Status ArithmScalarSampleTest(aura::Context *ctx, aura::TargetType type);

aura::Status ArithmeticSampleTest(aura::Context *ctx, aura::TargetType type);

aura::Status ConvertToSampleTest(aura::Context *ctx, aura::TargetType type);

aura::Status DctIDctSampleTest(aura::Context *ctx, aura::TargetType type);

aura::Status DftIDftSampleTest(aura::Context *ctx, aura::TargetType type);

aura::Status FlipSampleTest(aura::Context *ctx, aura::TargetType target_type);

aura::Status GemmSampleTest(aura::Context *ctx, aura::TargetType type);

aura::Status GridDftIDftSampleTest(aura::Context *ctx, aura::TargetType type);

aura::Status IntegralSampleTest(aura::Context *ctx, aura::TargetType type);

aura::Status MakeBorderSampleTest(aura::Context *ctx, aura::TargetType type);

aura::Status MeanStdDevSampleTest(aura::Context *ctx, aura::TargetType type);

aura::Status MergeSampleTest(aura::Context *ctx, aura::TargetType type);

aura::Status MinMaxLocSampleTest(aura::Context *ctx, aura::TargetType type);

aura::Status MinMaxSampleTest(aura::Context *ctx, aura::TargetType type);

aura::Status MulSpectrumsSampleTest(aura::Context *ctx, aura::TargetType type);

aura::Status NormSampleTest(aura::Context *ctx, aura::TargetType type);

aura::Status NormalizeSampleTest(aura::Context *ctx, aura::TargetType type);

aura::Status PsnrSampleTest(aura::Context *ctx, aura::TargetType type);

aura::Status RotateSampleTest(aura::Context *ctx, aura::TargetType target_type);

aura::Status SplitSampleTest(aura::Context *ctx, aura::TargetType type);

aura::Status SumMeanSampleTest(aura::Context *ctx, aura::TargetType type);

aura::Status TransposeSampleTest(aura::Context *ctx, aura::TargetType type);

#endif // AURA_SAMPLE_OPS_MATRIX_HPP__