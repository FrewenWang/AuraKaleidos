#include "sample_warp.hpp"

aura::Status RemapSampleTest(aura::Context *ctx, aura::TargetType type)
{
    AURA_LOGD(ctx, SAMPLE_TAG, "===================RemapSampleTest Begin ===================\n");

    // ---------------------
    //     prepare data
    // ---------------------
    // create mat
    aura::Sizes3 size    = {487, 487, 1};
    aura::Sizes3 mapsize = {487, 487, 2};
    aura::Mat src(ctx, aura::ElemType::U8,  size);
    aura::Mat dst(ctx, aura::ElemType::U8,  size);
    aura::Mat map(ctx, aura::ElemType::F32, mapsize);
    if (!(src.IsValid() && dst.IsValid() && map.IsValid()))
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "src, map or dst mat is invalid.");
        return aura::Status::ERROR;
    }

    // load data
    aura::Status status = src.Load("data/comm/cameraman_487x487.gray");
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "Remap sample test load failed, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        return status;
    }

    // set map value
    DT_S32 rows = size.m_height;
    DT_S32 cols = size.m_width;
    for (DT_S32 i = 0; i < rows; i++)
    {
        for (DT_S32 j = 0;j < cols; j++)
        {
            map.At<DT_F32>(i, j, 0) = static_cast<DT_F32>(cols - j);
            map.At<DT_F32>(i, j, 1) = static_cast<DT_F32>(rows - i);
        }
    }

    // ---------------------
    //     run interface
    // ---------------------
    // print info
    AURA_LOGD(ctx, SAMPLE_TAG, "Test remap param detail:\n");
    AURA_LOGD(ctx, SAMPLE_TAG, "elem_type(%s), mat_size(%s), interp_type(%s), border_type(%s), target(%s)\n",
              ElemTypesToString(src.GetElemType()).c_str(), src.GetSizes().ToString().c_str(), InterpTypeToString(aura::InterpType::NEAREST).c_str(),
              BorderTypeToString(aura::BorderType::REFLECT_101).c_str(), TargetTypeToString(type).c_str());

    // run and time
    aura::Time start_time = aura::Time::Now();
    status = IRemap(ctx, src, map, dst, aura::InterpType::NEAREST, aura::BorderType::REFLECT_101, aura::Scalar(0, 0, 0, 0), aura::OpTarget(type));
    aura::Time end_time = aura::Time::Now();

    // ---------------------
    //     check result
    // ---------------------
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "Remap Interface Execute Fail, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        AURA_LOGE(ctx, SAMPLE_TAG, "===================RemapSampleTest: Test Failed ===================\n");
    }
    else
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "Remap running time = %s\n", (end_time - start_time).ToString().c_str());
        dst.Dump("./remap_test.raw");
        AURA_LOGD(ctx, SAMPLE_TAG, "result stored at ./remap_test.raw\n");
        AURA_LOGD(ctx, SAMPLE_TAG, "===================RemapSampleTest: Test Succeeded ===================\n");
    }

    return status;
}