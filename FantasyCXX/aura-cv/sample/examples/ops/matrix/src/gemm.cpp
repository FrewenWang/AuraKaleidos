#include "sample_matrix.hpp"

aura::Status GemmSampleTest(aura::Context *ctx, aura::TargetType type)
{
    AURA_LOGD(ctx, SAMPLE_TAG, "=================== GemmSampleTest Begin ===================\n");

    // ---------------------
    //     prepare data
    // ---------------------
    // create mat
    aura::Sizes3 size = {487, 487, 1};
    aura::Mat src_u8(ctx,  aura::ElemType::U8,  size);
    aura::Mat src_f32(ctx, aura::ElemType::F32, size);
    aura::Mat dst_f32(ctx, aura::ElemType::F32, size);
    if (!(src_u8.IsValid() && src_f32.IsValid() && dst_f32.IsValid()))
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "src_u8, src_f32 or dst_f32 mat is invalid.");
        return aura::Status::ERROR;
    }

    // load data
    aura::Status status = src_u8.Load("data/comm/cameraman_487x487.gray");
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "Gemm sample test load failed, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        return status;
    }

    // convert src from u8 to f32
    status = IConvertTo(ctx, src_u8, src_f32);
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "Gemm convert failed, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        return status;
    }

    // ---------------------
    //     run interface
    // ---------------------
    // print info
    AURA_LOGD(ctx, SAMPLE_TAG, "Test gemm param detail:\n");
    AURA_LOGD(ctx, SAMPLE_TAG, "elem_type(%s), src_size(%s), target(%s)\n",
              ElemTypesToString(src_f32.GetElemType()).c_str(), src_f32.GetSizes().ToString().c_str(), TargetTypeToString(type).c_str());

    // run and time
    aura::Time start_time = aura::Time::Now();
    status = IGemm(ctx, src_f32, src_f32, dst_f32, aura::OpTarget(type));
    aura::Time end_time = aura::Time::Now();
    
    // ---------------------
    //     check result
    // ---------------------
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "Gemm Interface Execute Fail, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        AURA_LOGE(ctx, SAMPLE_TAG, "=================== GemmSampleTest: Test Failed ===================\n");
    }
    else
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "Gemm running time = %s\n", (end_time - start_time).ToString().c_str());
        dst_f32.Dump("./gemm_test.raw");
        AURA_LOGD(ctx, SAMPLE_TAG, "result stored at ./gemm_test.raw\n");
        AURA_LOGD(ctx, SAMPLE_TAG, "=================== GemmSampleTest: Test Succeeded ===================\n");
    }

    return status;
}