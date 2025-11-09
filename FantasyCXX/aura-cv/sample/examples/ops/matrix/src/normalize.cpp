#include "sample_matrix.hpp"

aura::Status NormalizeSampleTest(aura::Context *ctx, aura::TargetType type)
{
    AURA_LOGD(ctx, SAMPLE_TAG, "=================== NormalizeSampleTest Begin ===================\n");

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
        AURA_LOGE(ctx, SAMPLE_TAG, "Normalize sample test load failed, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        return status;
    }

    DT_F32 alpha = 3.2f;
    DT_F32 beta  = 1.0f;

    // ---------------------
    //     run interface
    // ---------------------
    // print info
    AURA_LOGD(ctx, SAMPLE_TAG, "Test normalize param detail:\n");
    AURA_LOGD(ctx, SAMPLE_TAG, "elem_type(%s), mat_size(%s), alpha(%f), beta(%f), norm_type(%s), target(%s)\n",
              ElemTypesToString(src.GetElemType()).c_str(), src.GetSizes().ToString().c_str(),
              alpha, beta, NormTypeToString(aura::NormType::NORM_L1).c_str(), TargetTypeToString(type).c_str());

    // run and time
    aura::Time start_time = aura::Time::Now();
    status = INormalize(ctx, src, dst, alpha, beta, aura::NormType::NORM_L1, aura::OpTarget(type));
    aura::Time end_time = aura::Time::Now();
    
    // ---------------------
    //     check result
    // ---------------------
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "Normalize Interface Execute Fail, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        AURA_LOGE(ctx, SAMPLE_TAG, "=================== NormalizeSampleTest: Test Failed ===================\n");
    }
    else
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "Normalize running time = %s\n", (end_time - start_time).ToString().c_str());
        dst.Dump("./normalize_test.raw");
        AURA_LOGD(ctx, SAMPLE_TAG, "result stored at ./normalize_test.raw\n");
        AURA_LOGD(ctx, SAMPLE_TAG, "=================== NormalizeSampleTest: Test Succeeded ===================\n");
    }

    return status;
}