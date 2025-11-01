#include "sample_pyramid.hpp"

aura::Status PyrUpSampleTest(aura::Context *ctx, aura::TargetType type)
{
    AURA_LOGD(ctx, SAMPLE_TAG, "=================== PyrUpSampleTest Begin ===================\n");

    // ---------------------
    //     prepare data
    // ---------------------
    // create mat
    aura::Sizes3 src_size = {487, 487, 1};
    aura::Sizes3 dst_size = {974, 974, 1};
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
        AURA_LOGE(ctx, SAMPLE_TAG, "PyrUp sample test load failed, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        return status;
    }

    // ---------------------
    //     run interface
    // ---------------------
    // print info
    AURA_LOGD(ctx, SAMPLE_TAG, "Test pyrup param detail:\n");
    AURA_LOGD(ctx, SAMPLE_TAG, "elem_type(%s), src_mat_size(%s), dst_mat_size(%s), kernel_param(ksize: %d, sigma: %f), border_type(%s), target(%s)\n",
              ElemTypesToString(src.GetElemType()).c_str(), src.GetSizes().ToString().c_str(), dst.GetSizes().ToString().c_str(),
              5, 1.0f, BorderTypeToString(aura::BorderType::REFLECT_101).c_str(), TargetTypeToString(type).c_str());

    // run and time
    aura::Time start_time = aura::Time::Now();
    status = IPyrUp(ctx, src, dst, 5, 1.0f, aura::BorderType::REFLECT_101, aura::OpTarget(type));
    aura::Time end_time = aura::Time::Now();

    // ---------------------
    //     check result
    // ---------------------
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "PyrUp Interface Execute Fail, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        AURA_LOGE(ctx, SAMPLE_TAG, "=================== PyrUpSampleTest: Test Failed ===================\n");
    }
    else
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "PyrUp running time = %s\n", (end_time - start_time).ToString().c_str());
        dst.Dump("./pyrup_test.raw");
        AURA_LOGD(ctx, SAMPLE_TAG, "result stored at ./pyrup_test.raw\n");
        AURA_LOGD(ctx, SAMPLE_TAG, "=================== PyrUpSampleTest: Test Succeeded ===================\n");
    }

    return status;
}