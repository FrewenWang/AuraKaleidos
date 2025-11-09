#include "sample_feature2d.hpp"

aura::Status FastSampleTest(aura::Context *ctx, aura::TargetType type)
{
    AURA_LOGD(ctx, SAMPLE_TAG, "=================== FastSampleTest Begin ===================\n");

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
        AURA_LOGE(ctx, SAMPLE_TAG, "Fast sample test load failed, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        return status;
    }

    std::vector<aura::KeyPoint> key_points;

    // set interface parameters
    DT_S32 thresh = 40;
    DT_S32 max_num_corners = 3000;
    aura::FastDetectorType detect_type = aura::FastDetectorType::FAST_9_16;

    // ---------------------
    //     run interface
    // ---------------------
    // print info
    AURA_LOGD(ctx, SAMPLE_TAG, "Test fast param detail:\n");
    AURA_LOGD(ctx, SAMPLE_TAG, "elem_type(%s), mat_size(%s), thresh(%d), detect_type(%s), max_num_corners(%d), target(%s)\n",
              ElemTypesToString(src.GetElemType()).c_str(), src.GetSizes().ToString().c_str(),
              thresh, FastDetectorTypeToString(detect_type).c_str(), max_num_corners, TargetTypeToString(type).c_str());

    // run and time
    aura::Time start_time = aura::Time::Now();
    status = IFast(ctx, src, key_points, thresh, DT_TRUE, detect_type, max_num_corners, aura::OpTarget(type));
    aura::Time end_time = aura::Time::Now();

    // ---------------------
    //     check result
    // ---------------------
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "Fast Interface Execute Fail, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        AURA_LOGE(ctx, SAMPLE_TAG, "=================== FastSampleTest: Test Failed ===================\n");
    }
    else
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "Fast running time = %s\n", (end_time - start_time).ToString().c_str());
        AURA_LOGD(ctx, SAMPLE_TAG, "keyPoint num = %d\n", key_points.size());
        AURA_LOGD(ctx, SAMPLE_TAG, "=================== FastSampleTest: Test Succeeded ===================\n");
    }

    return status;
}