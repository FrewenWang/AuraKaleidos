#include "sample_filter.hpp"

aura::Status LaplacianSampleTest(aura::Context *ctx, aura::TargetType type)
{
    AURA_LOGD(ctx, SAMPLE_TAG, "=================== LaplacianSampleTest Begin ===================\n");

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
        AURA_LOGE(ctx, SAMPLE_TAG, "Laplacian sample test load failed, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        return status;
    }

    // ---------------------
    //     run interface
    // ---------------------
    // call interface
    AURA_LOGD(ctx, SAMPLE_TAG, "Test laplacian param detail:\n");
    AURA_LOGD(ctx, SAMPLE_TAG, "elem_type(%s), mat_size(%s), kernel_param(ksize: %d), border_type(%s), target(%s)\n",
              ElemTypesToString(src.GetElemType()).c_str(), src.GetSizes().ToString().c_str(),
              3, BorderTypeToString(aura::BorderType::REFLECT_101).c_str(), TargetTypeToString(type).c_str());

    // run and time
    aura::Time start_time = aura::Time::Now();
    status = ILaplacian(ctx, src, dst, 3, aura::BorderType::REFLECT_101, aura::Scalar(0, 0, 0, 0), aura::OpTarget(type));
    aura::Time end_time = aura::Time::Now();

    // ---------------------
    //     check result
    // ---------------------
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "Laplacian Interface Execute Fail, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        AURA_LOGE(ctx, SAMPLE_TAG, "=================== LaplacianSampleTest: Test Failed ===================\n");
    }
    else
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "Laplacian running time = %s\n", (end_time - start_time).ToString().c_str());
        dst.Dump("./laplacian_test.raw");
        AURA_LOGD(ctx, SAMPLE_TAG, "result stored at ./laplacian_test.raw\n");
        AURA_LOGD(ctx, SAMPLE_TAG, "=================== LaplacianSampleTest: Test Successed ===================\n");
    }

    return status;
}