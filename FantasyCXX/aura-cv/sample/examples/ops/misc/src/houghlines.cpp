#include "sample_misc.hpp"

aura::Status HoughLinesSampleTest(aura::Context *ctx, aura::TargetType type)
{
    AURA_LOGD(ctx, SAMPLE_TAG, "=================== HoughLinesSampleTest Begin ===================\n");

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
    aura::Status status = src.Load("data/comm/lines_487x487.gray");
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "HoughLines sample test load failed, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        return status;
    }

    std::vector<aura::Scalar> dst;

    // set interface parameters
    DT_F64 rho       = 1;
    DT_F64 theta     = AURA_PI / 180.f;
    DT_S32 threshold = 100;
    DT_F64 srn       = 0;
    DT_F64 stn       = 0;
    DT_F64 min_theta = 0;
    DT_F64 max_theta = AURA_PI;
    auto line_type   = aura::LinesType::VEC2F;

    // ---------------------
    //     run interface
    // ---------------------
    // print info
    AURA_LOGD(ctx, SAMPLE_TAG, "Test houghlines param detail:\n");
    AURA_LOGD(ctx, SAMPLE_TAG, "elem_type(%s), mat_size(%s), line_type(%s), rho(%f), theta(%f), threshold(%d)\n",
              ElemTypesToString(src.GetElemType()).c_str(), src.GetSizes().ToString().c_str(),
              aura::LinesTypeToString(line_type).c_str(), rho, theta, threshold);
    AURA_LOGD(ctx, SAMPLE_TAG, "srn(%f), stn(%f), min_theta(%f), max_theta(%f), type(%s)\n",
              srn, stn, min_theta, max_theta, TargetTypeToString(type).c_str());

    // run and time
    aura::Time start_time = aura::Time::Now();
    status = IHoughLines(ctx, src, dst, line_type, rho, theta, threshold, srn, stn, min_theta, max_theta, aura::OpTarget(type));
    aura::Time end_time = aura::Time::Now();

    // ---------------------
    //     check result
    // ---------------------
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "HoughLines Interface Execute Fail, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        AURA_LOGE(ctx, SAMPLE_TAG, "=================== HoughLinesSampleTest: Test Failed ===================\n");
    }
    else
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "HoughLines running time = %s\n", (end_time - start_time).ToString().c_str());
        AURA_LOGD(ctx, SAMPLE_TAG, "Lines num = %d\n", dst.size());
        AURA_LOGD(ctx, SAMPLE_TAG, "=================== HoughLinesSampleTest: Test Succeeded ===================\n");
    }

    return status;
}