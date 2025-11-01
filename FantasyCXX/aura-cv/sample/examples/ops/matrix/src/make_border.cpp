#include "sample_matrix.hpp"

aura::Status MakeBorderSampleTest(aura::Context *ctx, aura::TargetType type)
{
    AURA_LOGD(ctx, SAMPLE_TAG, "=================== MakeBorderSampleTest Begin ===================\n");

    MI_S32 top    = 5;
    MI_S32 bottom = 5;
    MI_S32 left   = 5;
    MI_S32 right  = 5;
    
    aura::Scalar border_value(1, 2, 3, 4);

    // ---------------------
    //     prepare data
    // ---------------------
    // create mat
    aura::Sizes3 src_size = {487, 487, 1};
    aura::Sizes3 dst_size = src_size + aura::Sizes3(top + bottom, left + right, 0);
    aura::Mat src(ctx, aura::ElemType::U8, src_size);
    aura::Mat dst(ctx, aura::ElemType::U8, dst_size);
    if (!(src.IsValid() && dst.IsValid()))
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "src or dst mat is invalid.");
        return aura::Status::ERROR;
    }

    // load data
    aura::Status status = src.Load("data/comm/cameraman_487x487.gray");
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "MakeBorder sample test load failed, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        return status;
    }

    // ---------------------
    //     run interface
    // ---------------------
    // print info
    AURA_LOGD(ctx, SAMPLE_TAG, "Test makeborder param detail:\n");
    AURA_LOGD(ctx, SAMPLE_TAG, "elem_type(%s), src_size(%s), dst_size(%s), target(%s)\n",
              ElemTypesToString(src.GetElemType()).c_str(), src.GetSizes().ToString().c_str(),
              dst.GetSizes().ToString().c_str(), TargetTypeToString(type).c_str());

    // run and time
    aura::Time start_time = aura::Time::Now();
    status = IMakeBorder(ctx, src, dst, top, bottom, left, right, aura::BorderType::CONSTANT, border_value, aura::OpTarget(type));
    aura::Time end_time = aura::Time::Now();
    
    // ---------------------
    //     check result
    // ---------------------
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "MakeBorder Interface Execute Fail, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        AURA_LOGE(ctx, SAMPLE_TAG, "=================== MakeBorderSampleTest: Test Failed ===================\n");
    }
    else
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "MakeBorder running time = %s\n", (end_time - start_time).ToString().c_str());
        dst.Dump("./makeborder_test.raw");
        AURA_LOGD(ctx, SAMPLE_TAG, "result stored at ./makeborder_test.raw\n");
        AURA_LOGD(ctx, SAMPLE_TAG, "=================== MakeBorderSampleTest: Test Succeeded ===================\n");
    }

    return status;
}