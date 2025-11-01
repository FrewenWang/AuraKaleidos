#include "sample_misc.hpp"

aura::Status AdaptiveThresholdSampleTest(aura::Context *ctx, aura::TargetType type)
{
    AURA_LOGD(ctx, SAMPLE_TAG, "=================== AdaptiveThresholdSampleTest Begin ===================\n");

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
        AURA_LOGE(ctx, SAMPLE_TAG, "AdaptiveThreshold sample test load failed, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        return status;
    }

    // set interface parameters
    MI_F32 max_val     = 255.0f;
    MI_S32 thresh_type = AURA_THRESH_BINARY;
    MI_S32 blocksize   = 3;
    MI_F32 delta       = 5.0f;
    auto method        = aura::AdaptiveThresholdMethod::ADAPTIVE_THRESH_MEAN_C;

    // ---------------------
    //     run interface
    // ---------------------
    // print info
    AURA_LOGD(ctx, SAMPLE_TAG, "Test adaptive threshold param detail:\n");
    AURA_LOGD(ctx, SAMPLE_TAG, "elem_type(%s), mat_size(%s), max_val(%f), method(%s), thresh_type(%s), block_size(%d), delta(%f), target(%s)\n",
              ElemTypesToString(src.GetElemType()).c_str(), src.GetSizes().ToString().c_str(),
              max_val, AdaptiveThresholdMethodToString(method).c_str(), aura::ThresholdTypeToString(thresh_type).c_str(),
              blocksize, delta, TargetTypeToString(type).c_str());

    // run and time
    aura::Time start_time = aura::Time::Now();
    status = IAdaptiveThreshold(ctx, src, dst, max_val, method, thresh_type, blocksize, delta, aura::OpTarget(type));
    aura::Time end_time = aura::Time::Now();

    // ---------------------
    //     check result
    // ---------------------
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "AdaptiveThreshold Interface Execute Fail, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        AURA_LOGE(ctx, SAMPLE_TAG, "=================== AdaptiveThresholdSampleTest: Test Failed ===================\n");
    }
    else
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "AdaptiveThreshold running time = %s\n", (end_time - start_time).ToString().c_str());
        dst.Dump("./adaptive_threshold_test.raw");
        AURA_LOGD(ctx, SAMPLE_TAG, "result stored at ./adaptive_threshold_test.raw\n");
        AURA_LOGD(ctx, SAMPLE_TAG, "=================== AdaptiveThresholdSampleTest: Test Succeeded ===================\n");
    }

    return status;
}