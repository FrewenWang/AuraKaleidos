#include "sample_matrix.hpp"

aura::Status IntegralSampleTest(aura::Context *ctx, aura::TargetType type)
{
    AURA_LOGD(ctx, SAMPLE_TAG, "=================== IntegralSampleTest Begin ===================\n");

    // ---------------------
    //     prepare data
    // ---------------------
    // create mat
    aura::Sizes3 size = {487, 487, 1};
    aura::Mat src(ctx,  aura::ElemType::U8,  size);
    aura::Mat dst0(ctx, aura::ElemType::U32, size);
    aura::Mat dst1(ctx, aura::ElemType::F64, size);
    if (!(src.IsValid() && dst0.IsValid() && dst1.IsValid()))
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "src, dst0 or dst1 mat is invalid.");
        return aura::Status::ERROR;
    }

    // load data
    aura::Status status = src.Load("data/comm/cameraman_487x487.gray");
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "Integral sample test load failed, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        return status;
    }

    // ---------------------
    //     run interface
    // ---------------------
    // print info
    AURA_LOGD(ctx, SAMPLE_TAG, "Test integral param detail:\n");
    AURA_LOGD(ctx, SAMPLE_TAG, "elem_type(%s), src_size(%s), target(%s)\n",
              ElemTypesToString(src.GetElemType()).c_str(), src.GetSizes().ToString().c_str(), TargetTypeToString(type).c_str());

    // run and time
    aura::Time start_time = aura::Time::Now();
    status = IIntegral(ctx, src, dst0, dst1, aura::OpTarget(type));
    aura::Time end_time = aura::Time::Now();
    
    // ---------------------
    //     check result
    // ---------------------
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "Integral Interface Execute Fail, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        AURA_LOGE(ctx, SAMPLE_TAG, "=================== IntegralSampleTest: Test Failed ===================\n");
    }
    else
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "Integral running time = %s\n", (end_time - start_time).ToString().c_str());
        dst0.Dump("./integral_test.raw");
        AURA_LOGD(ctx, SAMPLE_TAG, "result stored at ./integral_test.raw\n");
        AURA_LOGD(ctx, SAMPLE_TAG, "=================== IntegralSampleTest: Test Succeeded ===================\n");
    }

    return status;
}