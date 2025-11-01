#include "sample_filter.hpp"

aura::Status SobelSampleTest(aura::Context *ctx, aura::TargetType type)
{
    AURA_LOGD(ctx, SAMPLE_TAG, "=================== SobelSampleTest Begin ===================\n");

    // ---------------------
    //     prepare data
    // ---------------------
    // create mat
    aura::Sizes3 size = {487, 487, 1};
    aura::Mat src(ctx, aura::ElemType::U8, size);
    aura::Mat dst(ctx, aura::ElemType::S16, size);
    if (!(src.IsValid() && dst.IsValid()))
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "src or dst mat is invalid.");
        return aura::Status::ERROR;
    }

    // load data
    aura::Status status = src.Load("data/comm/cameraman_487x487.gray");
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "Sobel sample test load failed, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        return status;
    }

    // ---------------------
    //     run interface
    // ---------------------
    // call interface
    AURA_LOGD(ctx, SAMPLE_TAG, "Test sobel param detail:\n");
    AURA_LOGD(ctx, SAMPLE_TAG, "elem_type(%s), mat_size(%s), kernel_param(dx = %d, dy = %d, ksize: %d, scale: %f), border_type(%s), target(%s)\n",
              ElemTypesToString(src.GetElemType()).c_str(), src.GetSizes().ToString().c_str(),
              1, 0, -1, 1.f, BorderTypeToString(aura::BorderType::REFLECT_101).c_str(), TargetTypeToString(type).c_str());

    // run and time
    aura::Time start_time = aura::Time::Now();
    status = ISobel(ctx, src, dst, 1, 0, -1, 1.f, aura::BorderType::REFLECT_101, aura::Scalar(0, 0, 0, 0), aura::OpTarget(type));
    aura::Time end_time = aura::Time::Now();

    // ---------------------
    //     check result
    // ---------------------
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "Sobel Interface Execute Fail, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        AURA_LOGE(ctx, SAMPLE_TAG, "=================== SobelSampleTest: Test Failed ===================\n");
    }
    else
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "Sobel running time = %s\n", (end_time - start_time).ToString().c_str());
        dst.Dump("./sobel_test.raw");
        AURA_LOGD(ctx, SAMPLE_TAG, "result stored at ./sobel_test.raw\n");
        AURA_LOGD(ctx, SAMPLE_TAG, "=================== SobelSampleTest: Test Successed ===================\n");
    }

    return status;
}