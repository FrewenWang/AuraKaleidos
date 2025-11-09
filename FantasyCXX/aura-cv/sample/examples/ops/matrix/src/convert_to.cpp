#include "sample_matrix.hpp"

aura::Status ConvertToSampleTest(aura::Context *ctx, aura::TargetType type)
{
    AURA_LOGD(ctx, SAMPLE_TAG, "=================== ConvertToSampleTest Begin ===================\n");

    // ---------------------
    //     prepare data
    // ---------------------
    // create mat
    aura::Sizes3 size = {487, 487, 1};
    aura::Mat src(ctx, aura::ElemType::U8, size);
    aura::Mat dst(ctx, aura::ElemType::F32, size);
    if (!(src.IsValid() && dst.IsValid()))
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "src or dst mat is invalid.");
        return aura::Status::ERROR;
    }

    // load data
    aura::Status status = src.Load("data/comm/cameraman_487x487.gray");
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "ConvertTo sample test load failed, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        return status;
    }

    // ---------------------
    //     run interface
    // ---------------------
    // print info
    DT_F32 alpha = 1.2f, beta = 0.5f;
    AURA_LOGD(ctx, SAMPLE_TAG, "Test convert_to param detail:\n");
    AURA_LOGD(ctx, SAMPLE_TAG, "src_elem_type(%s), dst_elem_type(%s), mat_size(%s), alpha(%f), beta(%f), target(%s)\n",
              ElemTypesToString(src.GetElemType()).c_str(), ElemTypesToString(dst.GetElemType()).c_str(), 
              src.GetSizes().ToString().c_str(), alpha, beta, TargetTypeToString(type).c_str());

    // run and time
    aura::Time start_time = aura::Time::Now();
    status = IConvertTo(ctx, src, dst, alpha, beta, aura::OpTarget(type));
    aura::Time end_time = aura::Time::Now();

    // ---------------------
    //     check result
    // ---------------------
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "ConvertTo Interface Execute Fail, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        AURA_LOGE(ctx, SAMPLE_TAG, "=================== ConvertToSampleTest: Test Failed ===================\n");
    }
    else
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "ConvertTo running time = %s\n", (end_time - start_time).ToString().c_str());
        dst.Dump("./convert_test.raw");
        AURA_LOGD(ctx, SAMPLE_TAG, "result stored at ./convert_test.raw\n");
        AURA_LOGD(ctx, SAMPLE_TAG, "=================== ConvertToSampleTest: Test Succeeded ===================\n");
    }

    return status;
}