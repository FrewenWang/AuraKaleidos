#include "sample_warp.hpp"

aura::Status WarpAffineSampleTest(aura::Context *ctx, aura::TargetType type)
{
    AURA_LOGD(ctx, SAMPLE_TAG, "=================== WarpAffineSampleTest Begin ===================\n");

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
        AURA_LOGE(ctx, SAMPLE_TAG, "WarpAffine sample test load failed, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        return status;
    }

    aura::Point2 center = aura::Point2(size.m_width / 2, size.m_height / 2);
    aura::Mat matrix = GetRotationMatrix2D(ctx, center, -10.0, 1.1);

    // ---------------------
    //     run interface
    // ---------------------
    // print info
    AURA_LOGD(ctx, SAMPLE_TAG, "Test warp affine param detail:\n");
    AURA_LOGD(ctx, SAMPLE_TAG, "elem_type(%s), mat_size(%s), interp_type(%s), border_type(%s), target(%s)\n",
              ElemTypesToString(src.GetElemType()).c_str(), src.GetSizes().ToString().c_str(), InterpTypeToString(aura::InterpType::NEAREST).c_str(),
              BorderTypeToString(aura::BorderType::REPLICATE).c_str(), TargetTypeToString(type).c_str());

    // run and time
    aura::Time start_time = aura::Time::Now();
    status = IWarpAffine(ctx, src, matrix, dst, aura::InterpType::NEAREST, aura::BorderType::REPLICATE, aura::Scalar(), aura::OpTarget(type));
    aura::Time end_time = aura::Time::Now();

    // ---------------------
    //     check result
    // ---------------------
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "WarpAffine Interface Execute Fail, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        AURA_LOGE(ctx, SAMPLE_TAG, "=================== WarpAffineSampleTest: Test Failed ===================\n");
    }
    else
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "WarpAffine running time = %s\n", (end_time - start_time).ToString().c_str());
        dst.Dump("./warp_affine_test.raw");
        AURA_LOGD(ctx, SAMPLE_TAG, "result stored at ./warp_affine_test.raw\n");
        AURA_LOGD(ctx, SAMPLE_TAG, "=================== WarpAffineSampleTest: Test Succeeded ===================\n");
    }

    return status;
}