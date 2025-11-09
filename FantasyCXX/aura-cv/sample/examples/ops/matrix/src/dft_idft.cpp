#include "sample_matrix.hpp"

aura::Status DftIDftSampleTest(aura::Context *ctx, aura::TargetType type)
{
    AURA_LOGD(ctx, SAMPLE_TAG, "=================== DftIDftSampleTest Begin ===================\n");

    // ---------------------
    //     prepare data
    // ---------------------
    // create mat
    aura::Sizes3 size0 = {512, 512, 1};
    aura::Sizes3 size1 = {512, 512, 2};
    aura::Mat src(ctx,  aura::ElemType::U8,  size0);
    aura::Mat dst0(ctx, aura::ElemType::F32, size1);
    aura::Mat dst1(ctx, aura::ElemType::U8,  size0);
    if (!(src.IsValid() && dst0.IsValid() && dst1.IsValid()))
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "src dst0, or dst1 mat is invalid.");
        return aura::Status::ERROR;
    }

    // load data
    aura::Status status = src.Load("data/comm/lena_512x512.y");
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "Dft sample test load failed, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        return status;
    }

    // ---------------------
    //     run interface
    // ---------------------
    // print info
    AURA_LOGD(ctx, SAMPLE_TAG, "Test dft and idft param detail:\n");
    AURA_LOGD(ctx, SAMPLE_TAG, "elem_type(%s), src_size(%s), target(%s)\n",
              ElemTypesToString(src.GetElemType()).c_str(), src.GetSizes().ToString().c_str(), TargetTypeToString(type).c_str());

    // run and time
    aura::Time start_time = aura::Time::Now();
    status = IDft(ctx, src, dst0, aura::OpTarget(type));
    aura::Time end_time = aura::Time::Now();
    
    // ---------------------
    //     check result
    // ---------------------
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "Dft Interface Execute Fail, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        AURA_LOGE(ctx, SAMPLE_TAG, "=================== DftIDftSampleTest: Test Failed ===================\n");
        return status;
    }
    else
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "Dft running time = %s\n", (end_time - start_time).ToString().c_str());
        dst0.Dump("./dft_test.raw");
        AURA_LOGD(ctx, SAMPLE_TAG, "result stored at ./dft_test.raw\n");
        AURA_LOGD(ctx, SAMPLE_TAG, "=================== DftIDftSampleTest: Test Succeeded ===================\n");
    }

    // run and time
    start_time = aura::Time::Now();
    status = IInverseDft(ctx, dst0, dst1, DT_TRUE, aura::OpTarget(type));
    end_time = aura::Time::Now();
    
    // ---------------------
    //     check result
    // ---------------------
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "InverseDft Interface Execute Fail, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        AURA_LOGE(ctx, SAMPLE_TAG, "=================== DftIDftSampleTest: Test Failed ===================\n");
        return status;
    }
    else
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "InverseDft running time = %s\n", (end_time - start_time).ToString().c_str());
        dst1.Dump("./idft_test.raw");
        AURA_LOGD(ctx, SAMPLE_TAG, "result stored at ./idft_test.raw\n");
        AURA_LOGD(ctx, SAMPLE_TAG, "=================== DftIDftSampleTest: Test Succeeded ===================\n");
    }

    return status;
}