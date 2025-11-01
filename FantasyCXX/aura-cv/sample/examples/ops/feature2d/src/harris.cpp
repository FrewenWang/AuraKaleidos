#include "sample_feature2d.hpp"

aura::Status HarrisSampleTest(aura::Context *ctx, aura::TargetType type)
{
    AURA_LOGD(ctx, SAMPLE_TAG, "=================== HarrisSampleTest Begin ===================\n");

    // ---------------------
    //     prepare data
    // ---------------------
    // create mat
    aura::Sizes3 size = {487, 487, 1};
    aura::Mat src(ctx, aura::ElemType::U8,  size);
    aura::Mat dst(ctx, aura::ElemType::F32, size);
    if (!(src.IsValid() && dst.IsValid()))
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "src or dst mat is invalid.");
        return aura::Status::ERROR;
    }

    // load data
    aura::Status status = src.Load("data/comm/cameraman_487x487.gray");
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "Harris sample test load failed, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        return status;
    }

    // ---------------------
    //     run interface
    // ---------------------
    // print info
    AURA_LOGD(ctx, SAMPLE_TAG, "Test harris param detail:\n");
    AURA_LOGD(ctx, SAMPLE_TAG, "elem_type(%s), mat_size(%s), block_size(%d), k_size(%d), k(%f), border_type(%s), target(%s)\n",
              ElemTypesToString(src.GetElemType()).c_str(), src.GetSizes().ToString().c_str(),
              5, 3, 0.04, BorderTypeToString(aura::BorderType::REFLECT_101).c_str(), TargetTypeToString(type).c_str());

    aura::Time start_time = aura::Time::Now();
    status = IHarris(ctx, src, dst, 5, 3, 0.04, aura::BorderType::REFLECT_101, aura::Scalar(0, 0, 0, 0), aura::OpTarget(type));
    aura::Time end_time = aura::Time::Now();

    // ---------------------
    //     check result
    // ---------------------
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "Harris Interface Execute Fail, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        AURA_LOGE(ctx, SAMPLE_TAG, "=================== HarrisSampleTest: Test Failed ===================\n");
    }
    else
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "Harris running time = %s\n", (end_time - start_time).ToString().c_str());
        dst.Dump("./harris_test.raw");
        AURA_LOGD(ctx, SAMPLE_TAG, "result stored at ./harris_test.raw\n");
        AURA_LOGD(ctx, SAMPLE_TAG, "=================== HarrisSampleTest: Test Succeeded ===================\n");
    }

    return status;
}