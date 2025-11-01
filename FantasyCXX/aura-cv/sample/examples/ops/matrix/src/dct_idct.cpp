#include "sample_matrix.hpp"

aura::Status DctIDctSampleTest(aura::Context *ctx, aura::TargetType type)
{
    AURA_LOGD(ctx, SAMPLE_TAG, "=================== DctIDctSampleTest Begin ===================\n");

    // ---------------------
    //     prepare data
    // ---------------------
    // create mat
    aura::Sizes3 size = {512, 512, 1};
    aura::Mat src(ctx,  aura::ElemType::U8,  size);
    aura::Mat dst0(ctx, aura::ElemType::F32, size);
    aura::Mat dst1(ctx, aura::ElemType::U8,  size);
    if (!(src.IsValid() && dst0.IsValid() && dst1.IsValid()))
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "src, dst0 or dst1 mat is invalid.");
        return aura::Status::ERROR;
    }

    // load data
    aura::Status status = src.Load("data/comm/lena_512x512.y");
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "Dct sample test load failed, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        return status;
    }

    // ---------------------
    //     run interface
    // ---------------------
    // print info
    AURA_LOGD(ctx, SAMPLE_TAG, "Test dct and idct param detail:\n");
    AURA_LOGD(ctx, SAMPLE_TAG, "elem_type(%s), src_size(%s), target(%s)\n",
              ElemTypesToString(src.GetElemType()).c_str(), src.GetSizes().ToString().c_str(), TargetTypeToString(type).c_str());

    // dct run and time
    aura::Time start_time = aura::Time::Now();
    status = IDct(ctx, src, dst0, aura::OpTarget(type));
    aura::Time end_time = aura::Time::Now();
    
    // ---------------------
    //     check result
    // ---------------------
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "Dct Interface Execute Fail, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        AURA_LOGE(ctx, SAMPLE_TAG, "=================== DctIDctSampleTest: Test Failed ===================\n");
    }
    else
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "Dct running time = %s\n", (end_time - start_time).ToString().c_str());
        dst0.Dump("./dct_test.raw");
        AURA_LOGD(ctx, SAMPLE_TAG, "result stored at ./dct_test.raw\n");
        AURA_LOGD(ctx, SAMPLE_TAG, "=================== DctIDctSampleTest: Test Succeeded ===================\n");
    }

    // idct run and time
    start_time = aura::Time::Now();
    status = IInverseDct(ctx, dst0, dst1, aura::OpTarget(type));
    end_time = aura::Time::Now();
    
    // ---------------------
    //     check result
    // ---------------------
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "InverseDct Interface Execute Fail, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        AURA_LOGE(ctx, SAMPLE_TAG, "=================== DctIDctSampleTest: Test Failed ===================\n");
    }
    else
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "InverseDct running time = %s\n", (end_time - start_time).ToString().c_str());
        dst1.Dump("./idct_test.raw");
        AURA_LOGD(ctx, SAMPLE_TAG, "result stored at ./idct_test.raw\n");
        AURA_LOGD(ctx, SAMPLE_TAG, "=================== DctIDctSampleTest: Test Succeeded ===================\n");
    }

    return status;
}