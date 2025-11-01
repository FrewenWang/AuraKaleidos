#include "sample_matrix.hpp"

aura::Status MulSpectrumsSampleTest(aura::Context *ctx, aura::TargetType type)
{
    AURA_LOGD(ctx, SAMPLE_TAG, "=================== MulSpectrumsSampleTest Begin ===================\n");

    // ---------------------
    //     prepare data
    // ---------------------
    // create mat
    aura::Sizes3 size = {256, 256, 2};
    aura::Mat src_u8(ctx,  aura::ElemType::U8,  size);
    aura::Mat src_f32(ctx, aura::ElemType::F32, size);
    aura::Mat dst_f32(ctx, aura::ElemType::F32, size);
    if (!(src_u8.IsValid() && src_f32.IsValid() && dst_f32.IsValid()))
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "src_u8, src_f32 or dst_f32 mat is invalid.");
        return aura::Status::ERROR;
    }

    // load data
    aura::Status status = src_u8.Load("data/comm/lena_256x256x2.uv");
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "MulSpectrums sample test load failed, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        return status;
    }

    // convert src u8 to f32
    status = IConvertTo(ctx, src_u8, src_f32);
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "MulSpectrums convert failed, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        return status;
    }

    // ---------------------
    //     run interface
    // ---------------------
    // print info
    AURA_LOGD(ctx, SAMPLE_TAG, "Test mulspectrums param detail:\n");
    AURA_LOGD(ctx, SAMPLE_TAG, "elem_type(%s), mat_size(%s), target(%s)\n",
              ElemTypesToString(src_f32.GetElemType()).c_str(), src_f32.GetSizes().ToString().c_str(),
              TargetTypeToString(type).c_str());

    // run and time
    aura::Time start_time = aura::Time::Now();
    status = IMulSpectrums(ctx, src_f32, src_f32, dst_f32, MI_FALSE, aura::OpTarget(type));
    aura::Time end_time = aura::Time::Now();
    
    // ---------------------
    //     check result
    // ---------------------
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "MulSpectrums Interface Execute Fail, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        AURA_LOGE(ctx, SAMPLE_TAG, "=================== MulSpectrumsSampleTest: Test Failed ===================\n");
    }
    else
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "MulSpectrums running time = %s\n", (end_time - start_time).ToString().c_str());
        dst_f32.Dump("./mulspectrums_test.raw");
        AURA_LOGD(ctx, SAMPLE_TAG, "result stored at ./mulspectrums_test.raw\n");
        AURA_LOGD(ctx, SAMPLE_TAG, "=================== MulSpectrumsSampleTest: Test Succeeded ===================\n");
    }

    return status;
}