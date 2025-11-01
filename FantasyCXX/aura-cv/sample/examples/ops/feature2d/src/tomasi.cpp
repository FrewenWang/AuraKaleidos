#include "sample_feature2d.hpp"

aura::Status TomasiSampleTest(aura::Context *ctx, aura::TargetType type)
{
    AURA_LOGD(ctx, SAMPLE_TAG, "=================== TomasiSampleTest Begin ===================\n");

    // ---------------------
    //     prepare data
    // ---------------------
    // create mat
    aura::Sizes3 size = {487, 487, 1};
    aura::Mat src(ctx, aura::ElemType::U8, size);
    if (!(src.IsValid()))
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "src mat is invalid.");
        return aura::Status::ERROR;
    }

    // load data
    aura::Status status = src.Load("data/comm/cameraman_487x487.gray");
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "Tomasi sample test load failed, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        return status;
    }

    std::vector<aura::KeyPoint> corners;

    // set interface parameters
    MI_S32 max_corners   = 10000;
    MI_F32 quality_level = 0.05;
    MI_F32 min_distance  = 3.0;
    MI_S32 block_size    = 3;
    MI_S32 gradient_size = 3;
    MI_BOOL use_harris   = MI_FALSE;
    MI_F32 harris_k      = 0.04;

    // ---------------------
    //     run interface
    // ---------------------
    // print info
    AURA_LOGD(ctx, SAMPLE_TAG, "Test tomasi param detail:\n");
    AURA_LOGD(ctx, SAMPLE_TAG, "elem_type(%s), mat_size(%s)\n", ElemTypesToString(src.GetElemType()).c_str(), src.GetSizes().ToString().c_str());
    AURA_LOGD(ctx, SAMPLE_TAG, "max_corners(%d), quality_level(%f), min_distance(%f), block_size(%d), gradient_size(%d), use_harris(%d), target(%s)\n",
              max_corners, quality_level, min_distance, block_size, gradient_size, use_harris, harris_k, TargetTypeToString(type).c_str());

    // run and time
    aura::Time start_time = aura::Time::Now();
    status = GoodFeaturesToTrack(ctx, src, corners, max_corners, quality_level, min_distance,
                                 block_size, gradient_size, use_harris, harris_k, aura::OpTarget(type));
    aura::Time end_time = aura::Time::Now();

    // ---------------------
    //     check result
    // ---------------------
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "Tomasi Interface Execute Fail, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        AURA_LOGE(ctx, SAMPLE_TAG, "=================== TomasiSampleTest: Test Failed ===================\n");
    }
    else
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "Tomasi running time = %s\n", (end_time - start_time).ToString().c_str());
        AURA_LOGD(ctx, SAMPLE_TAG, "Corners num = %d\n", corners.size());
        AURA_LOGD(ctx, SAMPLE_TAG, "=================== TomasiSampleTest: Test Succeeded ===================\n");
    }

    return status;
}