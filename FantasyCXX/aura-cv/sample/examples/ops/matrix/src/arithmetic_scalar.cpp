#include "sample_matrix.hpp"

aura::Status ArithmScalarSampleTest(aura::Context *ctx, aura::TargetType type)
{
    AURA_LOGD(ctx, SAMPLE_TAG, "=================== ArithmScalarSampleTest Begin ===================\n");

    // ---------------------
    //     prepare data
    // ---------------------
    // create mat
    aura::Sizes3 size = {487, 487, 1};
    aura::Mat src(ctx, aura::ElemType::U8, size);
    aura::Mat dst(ctx, aura::ElemType::U8, size);
    if (!(src.IsValid() && dst.IsValid()))
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "src or dst mat is invalid.");
        return aura::Status::ERROR;
    }

    // load data
    aura::Status status = src.Load("data/comm/cameraman_487x487.gray");
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "ArithmScalar sample test load failed, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        return status;
    }

    DT_F32 scalar = 128.2;

    // ---------------------
    //     run interface
    // ---------------------
    // print info
    AURA_LOGD(ctx, SAMPLE_TAG, "Test arithmetic param detail:\n");
    AURA_LOGD(ctx, SAMPLE_TAG, "elem_type(%s), mat_size(%s), scalar(%f), target(%s)\n",
              ElemTypesToString(src.GetElemType()).c_str(), src.GetSizes().ToString().c_str(), scalar, TargetTypeToString(type).c_str());

    // run and time
    aura::Time start_time = aura::Time::Now();
    status = IDivide(ctx, scalar, src, dst, aura::OpTarget(type)); // see IAdd, ISubtract, IMultiply for more information
    aura::Time end_time = aura::Time::Now();

    // ---------------------
    //     check result
    // ---------------------
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "ArithmScalar Interface Execute Fail, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        AURA_LOGE(ctx, SAMPLE_TAG, "=================== ArithmScalarSampleTest: Test Failed ===================\n");
    }
    else
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "ArithmScalar running time = %s\n", (end_time - start_time).ToString().c_str());
        dst.Dump("./arithmscalar_test.raw");
        AURA_LOGD(ctx, SAMPLE_TAG, "result stored at ./arithmscalar_test.raw\n");
        AURA_LOGD(ctx, SAMPLE_TAG, "=================== ArithmScalarSampleTest: Test Succeeded ===================\n");
    }

    return status;
}