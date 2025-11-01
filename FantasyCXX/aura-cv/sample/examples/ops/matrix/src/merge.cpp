#include "sample_matrix.hpp"

aura::Status MergeSampleTest(aura::Context *ctx, aura::TargetType type)
{
    AURA_LOGD(ctx, SAMPLE_TAG, "=================== MergeSampleTest Begin ===================\n");

    // ---------------------
    //     prepare data
    // ---------------------
    // create mat
    aura::Sizes3 src_size = {487, 487, 1};
    aura::Sizes3 dst_size = {487, 487, 2};
    aura::Mat src0(ctx, aura::ElemType::U8, src_size);
    aura::Mat src1(ctx, aura::ElemType::U8, src_size);
    aura::Mat dst(ctx,  aura::ElemType::U8, dst_size);
    if (!(src0.IsValid() && src1.IsValid() && dst.IsValid()))
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "src0, src1 or dst mat is invalid.");
        return aura::Status::ERROR;
    }

    // load data
    aura::Status status = src0.Load("data/comm/cameraman_487x487.gray");
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "Merge sample test load failed, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        return status;
    }

    status = src1.Load("data/comm/lines_487x487.gray");
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "Merge sample test load failed, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        return status;
    }

    std::vector<aura::Mat> srcs;
    srcs.push_back(src0);
    srcs.push_back(src1);

    // ---------------------
    //     run interface
    // ---------------------
    // print info
    AURA_LOGD(ctx, SAMPLE_TAG, "Test merge param detail:\n");
    AURA_LOGD(ctx, SAMPLE_TAG, "elem_type(%s), src_mat_size(%s), dst_mat_size(%s), target(%s)\n",
              ElemTypesToString(src0.GetElemType()).c_str(), src1.GetSizes().ToString().c_str(),
              dst.GetSizes().ToString().c_str(), TargetTypeToString(type).c_str());

    // run and time
    aura::Time start_time = aura::Time::Now();
    status = IMerge(ctx, srcs, dst, aura::OpTarget(type));
    aura::Time end_time = aura::Time::Now();
    
    // ---------------------
    //     check result
    // ---------------------
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "Merge Interface Execute Fail, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        AURA_LOGE(ctx, SAMPLE_TAG, "=================== MergeSampleTest: Test Failed ===================\n");
    }
    else
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "Merge running time = %s\n", (end_time - start_time).ToString().c_str());
        dst.Dump("./merge_test.raw");
        AURA_LOGD(ctx, SAMPLE_TAG, "result stored at ./merge_test.raw\n");
        AURA_LOGD(ctx, SAMPLE_TAG, "=================== MergeSampleTest: Test Succeeded ===================\n");
    }

    return status;
}