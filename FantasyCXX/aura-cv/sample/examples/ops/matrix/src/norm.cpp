#include "sample_matrix.hpp"

aura::Status NormSampleTest(aura::Context *ctx, aura::TargetType type)
{
    AURA_LOGD(ctx, SAMPLE_TAG, "=================== NormSampleTest Begin ===================\n");

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
        AURA_LOGE(ctx, SAMPLE_TAG, "Norm sample test load failed, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        return status;
    }

    MI_F64 res_norm = 0.0;

    // ---------------------
    //     run interface
    // ---------------------
    // print info
    AURA_LOGD(ctx, SAMPLE_TAG, "Test norm param detail:\n");
    AURA_LOGD(ctx, SAMPLE_TAG, "elem_type(%s), mat_size(%s), norm_type(%s), target(%s)\n",
              ElemTypesToString(src.GetElemType()).c_str(), src.GetSizes().ToString().c_str(), 
              NormTypeToString(aura::NormType::NORM_L2SQR).c_str(), TargetTypeToString(type).c_str());

    // run and time
    aura::Time start_time = aura::Time::Now();
    status = INorm(ctx, src, &res_norm, aura::NormType::NORM_L2SQR, aura::OpTarget(type));
    aura::Time end_time = aura::Time::Now();
    
    // ---------------------
    //     check result
    // ---------------------
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "Norm Interface Execute Fail, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        AURA_LOGE(ctx, SAMPLE_TAG, "=================== NormSampleTest: Test Failed ===================\n");
    }
    else
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "Norm running time = %s\n", (end_time - start_time).ToString().c_str());
        AURA_LOGD(ctx, SAMPLE_TAG, "res_norm = %f\n", res_norm);
        AURA_LOGD(ctx, SAMPLE_TAG, "=================== NormSampleTest: Test Succeeded ===================\n");
    }

    return status;
}