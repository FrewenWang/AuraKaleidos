#include "sample_matrix.hpp"

aura::Status PsnrSampleTest(aura::Context *ctx, aura::TargetType type)
{
    AURA_LOGD(ctx, SAMPLE_TAG, "=================== PsnrSampleTest Begin ===================\n");

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
        AURA_LOGE(ctx, SAMPLE_TAG, "Psnr sample test load failed, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        return status;
    }

    DT_F64 res_psnr = 0.0;

    // ---------------------
    //     run interface
    // ---------------------
    // print info
    AURA_LOGD(ctx, SAMPLE_TAG, "Test psnr param detail:\n");
    AURA_LOGD(ctx, SAMPLE_TAG, "elem_type(%s), mat_size(%s), target(%s)\n",
              ElemTypesToString(src.GetElemType()).c_str(), src.GetSizes().ToString().c_str(), TargetTypeToString(type).c_str());

    // run and time
    aura::Time start_time = aura::Time::Now();
    status = IPsnr(ctx, src, src, 255.0, &res_psnr, aura::OpTarget(type));
    aura::Time end_time = aura::Time::Now();
    
    // ---------------------
    //     check result
    // ---------------------
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "Psnr Interface Execute Fail, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        AURA_LOGE(ctx, SAMPLE_TAG, "=================== PsnrSampleTest: Test Failed ===================\n");
    }
    else
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "Psnr running time = %s\n", (end_time - start_time).ToString().c_str());
        AURA_LOGD(ctx, SAMPLE_TAG, "res = %f\n", res_psnr);
        AURA_LOGD(ctx, SAMPLE_TAG, "=================== PsnrSampleTest: Test Succeeded ===================\n");
    }

    return status;
}