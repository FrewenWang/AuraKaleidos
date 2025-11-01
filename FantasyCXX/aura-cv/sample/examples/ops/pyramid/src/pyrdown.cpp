#include "sample_pyramid.hpp"

aura::Status PyrDownSampleTest(aura::Context *ctx, aura::TargetType type)
{
    AURA_LOGD(ctx, SAMPLE_TAG, "=================== PyrDownSampleTest Begin ===================\n");

    // ---------------------
    //     prepare data
    // ---------------------
    // create mat
    aura::Sizes3 src_size = {487, 487, 1};
    aura::Sizes3 dst_size = {244, 244, 1};
    aura::Mat src(ctx, aura::ElemType::U8, src_size);
    aura::Mat dst(ctx, aura::ElemType::U8, dst_size);
    if (!(src.IsValid() && dst.IsValid()))
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "src or dst mat is invalid.");
        return aura::Status::ERROR;
    }

    // load data
    aura::Status status = src.Load("data/comm/cameraman_487x487.gray");
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "PyrDown sample test load failed, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        return status;
    }

    // ---------------------
    //     run interface
    // ---------------------
    // print info
    AURA_LOGD(ctx, SAMPLE_TAG, "Test pyrdown param detail:\n");
    AURA_LOGD(ctx, SAMPLE_TAG, "elem_type(%s), src_mat_size(%s), dst_mat_size(%s), kernel_param(ksize: %d, sigma: %f), border_type(%s), target(%s)\n",
              ElemTypesToString(src.GetElemType()).c_str(), src.GetSizes().ToString().c_str(), dst.GetSizes().ToString().c_str(),
              5, 1.0f, BorderTypeToString(aura::BorderType::REFLECT_101).c_str(), TargetTypeToString(type).c_str());

    // run and time
    aura::Time start_time = aura::Time::Now();
    status = IPyrDown(ctx, src, dst, 5, 1.0f, aura::BorderType::REFLECT_101, aura::OpTarget(type));
    aura::Time end_time = aura::Time::Now();

    // ---------------------
    //     check result
    // ---------------------
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "PyrDown Interface Execute Fail, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        AURA_LOGE(ctx, SAMPLE_TAG, "=================== PyrDownSampleTest: Test Failed ===================\n");
    }
    else
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "PyrDown running time = %s\n", (end_time - start_time).ToString().c_str());
        dst.Dump("./pyrdown_test.raw");
        AURA_LOGD(ctx, SAMPLE_TAG, "result stored at ./pyrdown_test.raw\n");
        AURA_LOGD(ctx, SAMPLE_TAG, "=================== PyrDownSampleTest: Test Succeeded ===================\n");
    }

    return status;
}