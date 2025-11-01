#include "sample_matrix.hpp"

aura::Status MinMaxSampleTest(aura::Context *ctx, aura::TargetType type)
{
    AURA_LOGD(ctx, SAMPLE_TAG, "=================== MinMaxSampleTest Begin ===================\n");

    // ---------------------
    //     prepare data
    // ---------------------
    // create mat
    aura::Sizes3 size = {487, 487, 1};
    aura::Mat src0(ctx, aura::ElemType::U8, size);
    aura::Mat src1(ctx, aura::ElemType::U8, size);
    aura::Mat dst(ctx,  aura::ElemType::U8, size);
    if (!(src0.IsValid() && src1.IsValid() && dst.IsValid()))
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "src0, src1 or dst mat is invalid.");
        return aura::Status::ERROR;
    }

    // load data
    aura::Status status = src0.Load("data/comm/cameraman_487x487.gray");
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "MinMax sample test load failed, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        return status;
    }

    status = src1.Load("data/comm/lines_487x487.gray");
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "MinMax sample test load failed, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        return status;
    }

    // ---------------------
    //     run interface
    // ---------------------
    // print info
    AURA_LOGD(ctx, SAMPLE_TAG, "Test minmax param detail:\n");
    AURA_LOGD(ctx, SAMPLE_TAG, "elem_type(%s), mat_size(%s), target(%s)\n",
              ElemTypesToString(src0.GetElemType()).c_str(), src1.GetSizes().ToString().c_str(),
              TargetTypeToString(type).c_str());

    // run and time
    aura::Time start_time = aura::Time::Now();
    status = IMin(ctx, src0, src1, dst, aura::OpTarget(type));
    aura::Time end_time = aura::Time::Now();
    
    // ---------------------
    //     check result
    // ---------------------
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "MinMax Interface Execute Fail, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        AURA_LOGE(ctx, SAMPLE_TAG, "=================== MinMaxSampleTest: Test Failed ===================\n");
    }
    else
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "MinMax running time = %s\n", (end_time - start_time).ToString().c_str());
        dst.Dump("./minmax_test.raw");
        AURA_LOGD(ctx, SAMPLE_TAG, "result stored at ./minmax_test.raw\n");
        AURA_LOGD(ctx, SAMPLE_TAG, "=================== MinMaxSampleTest: Test Succeeded ===================\n");
    }

    return status;
}