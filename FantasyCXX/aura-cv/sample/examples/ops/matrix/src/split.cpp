#include "sample_matrix.hpp"

aura::Status SplitSampleTest(aura::Context *ctx, aura::TargetType type)
{
    AURA_LOGD(ctx, SAMPLE_TAG, "=================== SplitSampleTest Begin ===================\n");

    // ---------------------
    //     prepare data
    // ---------------------
    // create mat
    aura::Sizes3 src_size = {512, 512, 3};
    aura::Sizes3 dst_size = {512, 512, 1};
    aura::Mat src(ctx,  aura::ElemType::U8, src_size);
    aura::Mat dst0(ctx, aura::ElemType::U8, dst_size);
    aura::Mat dst1(ctx, aura::ElemType::U8, dst_size);
    aura::Mat dst2(ctx, aura::ElemType::U8, dst_size);
    if (!(src.IsValid() && dst0.IsValid() && dst1.IsValid() && dst2.IsValid()))
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "src, dst0, dst1 or dst2 mat is invalid.");
        return aura::Status::ERROR;
    }

    // load data
    aura::Status status = src.Load("data/comm/baboon_512x512.rgb");
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "Split sample test load failed, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        return status;
    }

    std::vector<aura::Mat> dsts;
    dsts.emplace_back(dst0);
    dsts.emplace_back(dst1);
    dsts.emplace_back(dst2);

    // ---------------------
    //     run interface
    // ---------------------
    // print info
    AURA_LOGD(ctx, SAMPLE_TAG, "Test split param detail:\n");
    AURA_LOGD(ctx, SAMPLE_TAG, "elem_type(%s), mat_size(%s), target(%s)\n",
              ElemTypesToString(src.GetElemType()).c_str(), src.GetSizes().ToString().c_str(),
              TargetTypeToString(type).c_str());

    // run and time
    aura::Time start_time = aura::Time::Now();
    status = ISplit(ctx, src, dsts, aura::OpTarget(type));
    aura::Time end_time = aura::Time::Now();
    
    // ---------------------
    //     check result
    // ---------------------
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "Split Interface Execute Fail, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        AURA_LOGE(ctx, SAMPLE_TAG, "=================== SplitSampleTest: Test Failed ===================\n");
    }
    else
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "Split running time = %s\n", (end_time - start_time).ToString().c_str());
        dsts[0].Dump("./split_test0.raw");
        dsts[1].Dump("./split_test1.raw");
        dsts[2].Dump("./split_test2.raw");
        AURA_LOGD(ctx, SAMPLE_TAG, "result stored at ./split_test0.raw\n");
        AURA_LOGD(ctx, SAMPLE_TAG, "result stored at ./split_test1.raw\n");
        AURA_LOGD(ctx, SAMPLE_TAG, "result stored at ./split_test2.raw\n");
        AURA_LOGD(ctx, SAMPLE_TAG, "=================== SplitSampleTest: Test Succeeded ===================\n");
    }

    return status;
}