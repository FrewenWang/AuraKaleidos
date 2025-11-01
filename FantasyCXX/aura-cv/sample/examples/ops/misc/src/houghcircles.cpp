#include "sample_misc.hpp"

aura::Status HoughCirclesSampleTest(aura::Context *ctx, aura::TargetType type)
{
    AURA_LOGD(ctx, SAMPLE_TAG, "=================== HoughCirclesSampleTest Begin ===================\n");

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
        AURA_LOGE(ctx, SAMPLE_TAG, "HoughCircles sample test load failed, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        return status;
    }

    std::vector<aura::Scalar> dst;

    // set interface parameters
    MI_F64 dp           = 2;
    MI_F64 min_dist     = 115;
    MI_F64 canny_thresh = 100;
    MI_F64 acc_thresh   = 30;
    MI_S32 min_radius   = 1;
    MI_S32 max_radius   = 130;
    auto method         = aura::HoughCirclesMethod::HOUGH_GRADIENT;

    // ---------------------
    //     run interface
    // ---------------------
    // print info
    AURA_LOGD(ctx, SAMPLE_TAG, "Test houghcircles param detail:\n");
    AURA_LOGD(ctx, SAMPLE_TAG, "elem_type(%s), mat_size(%s), method(%s), dp(%f), min_dist(%f), canny_thresh(%f)\n",
              ElemTypesToString(src.GetElemType()).c_str(), src.GetSizes().ToString().c_str(),
              HoughCirclesMethodToString(method).c_str(), dp, min_dist, canny_thresh);
    AURA_LOGD(ctx, SAMPLE_TAG, "acc_thresh(%f), min_radius(%d), max_radius(%d), type(%s)\n",
              acc_thresh, min_radius, max_radius, TargetTypeToString(type).c_str());

    // run and time
    aura::Time start_time = aura::Time::Now();
    status = IHoughCircles(ctx, src, dst, method, dp, min_dist, canny_thresh, acc_thresh, min_radius, max_radius, aura::OpTarget(type));
    aura::Time end_time = aura::Time::Now();

    // ---------------------
    //     check result
    // ---------------------
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "HoughCircles Interface Execute Fail, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        AURA_LOGE(ctx, SAMPLE_TAG, "=================== HoughCirclesSampleTest: Test Failed ===================\n");
    }
    else
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "HoughCircles running time = %s\n", (end_time - start_time).ToString().c_str());
        AURA_LOGD(ctx, SAMPLE_TAG, "Circles num = %d\n", dst.size());
        AURA_LOGD(ctx, SAMPLE_TAG, "=================== HoughCirclesSampleTest: Test Succeeded ===================\n");
    }

    return status;
}