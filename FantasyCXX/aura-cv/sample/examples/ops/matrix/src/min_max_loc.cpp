#include "sample_matrix.hpp"

aura::Status MinMaxLocSampleTest(aura::Context *ctx, aura::TargetType type)
{
    AURA_LOGD(ctx, SAMPLE_TAG, "=================== MinMaxLocSampleTest Begin ===================\n");

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
        AURA_LOGE(ctx, SAMPLE_TAG, "MinMaxLoc sample test load failed, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        return status;
    }

    MI_F64 res_min = 0.0;
    MI_F64 res_max = 0.0;

    aura::Point3i res_min_pos = {-1, -1, -1};
    aura::Point3i res_max_pos = {-1, -1, -1};

    // ---------------------
    //     run interface
    // ---------------------
    // print info
    AURA_LOGD(ctx, SAMPLE_TAG, "Test min and max loc param detail:\n");
    AURA_LOGD(ctx, SAMPLE_TAG, "elem_type(%s), mat_size(%s), target(%s)\n",
              ElemTypesToString(src.GetElemType()).c_str(), src.GetSizes().ToString().c_str(), TargetTypeToString(type).c_str());

    // run and time
    aura::Time start_time = aura::Time::Now();
    status = IMinMaxLoc(ctx, src, &res_min, &res_max, &res_min_pos, &res_max_pos, aura::OpTarget(type));
    aura::Time end_time = aura::Time::Now();
    
    // ---------------------
    //     check result
    // ---------------------
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "MinMaxLoc Interface Execute Fail, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        AURA_LOGE(ctx, SAMPLE_TAG, "=================== MinMaxLocSampleTest: Test Failed ===================\n");
    }
    else
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "MinMaxLoc running time = %s\n", (end_time - start_time).ToString().c_str());
        AURA_LOGD(ctx, SAMPLE_TAG, "Min = %f, Max = %f, \n", res_min, res_max);
        AURA_LOGD(ctx, SAMPLE_TAG, "MinLoc = %s, MaxLoc = %s\n", res_min_pos.ToString().c_str(), res_max_pos.ToString().c_str());
        AURA_LOGD(ctx, SAMPLE_TAG, "=================== MinMaxLocSampleTest: Test Succeeded ===================\n");
    }

    return status;
}