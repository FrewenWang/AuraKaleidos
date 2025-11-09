#include "sample_filter.hpp"

aura::Status Filter2dSampleTest(aura::Context *ctx, aura::TargetType type)
{
    AURA_LOGD(ctx, SAMPLE_TAG, "=================== Filter2dSampleTest Begin ===================\n");

    // ---------------------
    //     prepare data
    // ---------------------
    // create mat
    aura::Sizes3 size = {487, 487, 1};
    aura::Mat src(ctx, aura::ElemType::U8, size);
    aura::Mat dst(ctx, aura::ElemType::U8, size);
    aura::Mat kernel(ctx, aura::ElemType::F32, {3, 3, 1});
    if (!(src.IsValid() && dst.IsValid() && kernel.IsValid()))
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "src, dst or kernel mat is invalid.");
        return aura::Status::ERROR;
    }

    // load data
    aura::Status status = src.Load("data/comm/cameraman_487x487.gray");
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "Filter2d sample test load failed, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        return status;
    }

    // set kernel mat value
    DT_F32 *data = reinterpret_cast<DT_F32*>(kernel.GetData());
    DT_S32 n = kernel.GetTotalBytes() / sizeof(DT_F32);
    for (DT_S32 i = 0; i < n; i++)
    {
        data[i] = 0.3f;
    }

    // ---------------------
    //     run interface
    // ---------------------
    // print info
    AURA_LOGD(ctx, SAMPLE_TAG, "Test filter2d param detail:\n");
    AURA_LOGD(ctx, SAMPLE_TAG, "elem_type(%s), mat_size(%s), kernel_param(ksize: %d), border_type(%s), target(%s)\n",
              ElemTypesToString(src.GetElemType()).c_str(), src.GetSizes().ToString().c_str(),
              3, BorderTypeToString(aura::BorderType::REFLECT_101).c_str(), TargetTypeToString(type).c_str());

    // run and time
    aura::Time start_time = aura::Time::Now();
    status = IFilter2d(ctx, src, dst, kernel, aura::BorderType::REFLECT_101, aura::Scalar(0, 0, 0, 0), aura::OpTarget(type));
    aura::Time end_time = aura::Time::Now();

    // ---------------------
    //     check result
    // ---------------------
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "Filter2d Interface Execute Fail, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        AURA_LOGE(ctx, SAMPLE_TAG, "=================== Filter2dSampleTest: Test Failed ===================\n");
    }
    else
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "Filter2d running time = %s\n", (end_time - start_time).ToString().c_str());
        dst.Dump("./filter2d_test.raw");
        AURA_LOGD(ctx, SAMPLE_TAG, "result stored at ./filter2d_test.raw\n");
        AURA_LOGD(ctx, SAMPLE_TAG, "=================== Filter2dSampleTest: Test Successed ===================\n");
    }

    return status;
}