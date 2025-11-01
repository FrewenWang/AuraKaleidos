#include "sample_warp.hpp"

aura::Status WarpPerspectiveSampleTest(aura::Context *ctx, aura::TargetType type)
{
    AURA_LOGD(ctx, SAMPLE_TAG, "=================== WarpPerspectiveSampleTest Begin ===================\n");

    // ---------------------
    //     prepare data
    // ---------------------
    // create mat
    aura::Sizes3 size = {487, 487, 1};
    aura::Mat src(ctx, aura::ElemType::U8, size);
    aura::Mat dst(ctx, aura::ElemType::U8, size);
    aura::Mat matrix(ctx, aura::ElemType::F64, aura::Sizes(3, 3));
    if (!(src.IsValid() && dst.IsValid() && matrix.IsValid()))
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "src, matrix or dst mat is invalid.");
        return aura::Status::ERROR;
    }

    // load data
    aura::Status status = src.Load("data/comm/cameraman_487x487.gray");
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "WarpPerspective sample test load failed, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        return status;
    }

    // set matrix mat value
    MI_F64 *data = reinterpret_cast<MI_F64*>(matrix.GetData());
    MI_S32 n = matrix.GetTotalBytes() / sizeof(MI_F64);
    for (MI_S32 i = 0; i < n; i++)
    {
        data[i] = 0.3;
    }

    matrix.At<MI_F64>(2, 2, 0) = 1.0;

    // ---------------------
    //     run interface
    // ---------------------
    // print info
    AURA_LOGD(ctx, SAMPLE_TAG, "Test warp perspective param detail:\n");
    AURA_LOGD(ctx, SAMPLE_TAG, "elem_type(%s), mat_size(%s), interp_type(%s), border_type(%s), target(%s)\n",
              ElemTypesToString(src.GetElemType()).c_str(), src.GetSizes().ToString().c_str(), InterpTypeToString(aura::InterpType::NEAREST).c_str(),
              BorderTypeToString(aura::BorderType::REFLECT_101).c_str(), TargetTypeToString(type).c_str());

    // run and time
    aura::Time start_time = aura::Time::Now();
    status = IWarpPerspective(ctx, src, matrix, dst, aura::InterpType::NEAREST, aura::BorderType::REFLECT_101, aura::Scalar(), aura::OpTarget(type));
    aura::Time end_time = aura::Time::Now();

    // ---------------------
    //     check result
    // ---------------------
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "WarpPerspective Interface Execute Fail, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        AURA_LOGE(ctx, SAMPLE_TAG, "=================== WarpPerspectiveSampleTest: Test Failed ===================\n");
    }
    else
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "WarpPerspective running time = %s\n", (end_time - start_time).ToString().c_str());
        dst.Dump("./warp_perspective_test.raw");
        AURA_LOGD(ctx, SAMPLE_TAG, "result stored at ./warp_perspective_test.raw\n");
        AURA_LOGD(ctx, SAMPLE_TAG, "=================== WarpPerspectiveSampleTest: Test Succeeded ===================\n");
    }

    return status;
}