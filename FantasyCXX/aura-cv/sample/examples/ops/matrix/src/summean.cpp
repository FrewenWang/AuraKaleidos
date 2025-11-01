#include "sample_matrix.hpp"

aura::Status SumMeanSampleTest(aura::Context *ctx, aura::TargetType type)
{
    AURA_LOGD(ctx, SAMPLE_TAG, "=================== SumMeanSampleTest Begin ===================\n");

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
        AURA_LOGE(ctx, SAMPLE_TAG, "SumMean sample test load failed, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        return status;
    }

    aura::Scalar dst_sum;
    aura::Scalar dst_mean;

    // ---------------------
    //     run interface
    // ---------------------
    // print info
    AURA_LOGD(ctx, SAMPLE_TAG, "Test sum mean param detail:\n");
    AURA_LOGD(ctx, SAMPLE_TAG, "elem_type(%s), mat_size(%s), target(%s)\n",
              ElemTypesToString(src.GetElemType()).c_str(), src.GetSizes().ToString().c_str(), TargetTypeToString(type).c_str());

    // run and time
    aura::Time start_time = aura::Time::Now();
    status  = ISum(ctx, src, dst_sum, aura::OpTarget(type));
    status |= IMean(ctx, src, dst_mean, aura::OpTarget(type));
    aura::Time end_time = aura::Time::Now();
    
    // ---------------------
    //     check result
    // ---------------------
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "Sum and Mean Interface Execute Fail, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        AURA_LOGE(ctx, SAMPLE_TAG, "=================== SumMeanSampleTest: Test Failed ===================\n");
    }
    else
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "Sum and Mean running time = %s\n", (end_time - start_time).ToString().c_str());
        AURA_LOGD(ctx, SAMPLE_TAG, "Sum = %s\n", dst_sum.ToString().c_str());
        AURA_LOGD(ctx, SAMPLE_TAG, "Mean = %s\n", dst_mean.ToString().c_str());
        AURA_LOGD(ctx, SAMPLE_TAG, "=================== SumMeanSampleTest: Test Succeeded ===================\n");
    }

    return status;
}