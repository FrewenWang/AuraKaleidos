#include "sample_ops.hpp"

static std::string help_info = R"(
Usage:
    Usage: sample_cvtcolor [Operators] [TargetType]

Example usage:
    Usage: ./sample_cvtcolor cvtcolor none

Operators:
    This module contains convert color operators, for example:
    cvtcolor             Performs color space conversion on a batch of input iauras.

TargetType:
    These operators run on different hardware(CPU, GPU and DSP),
    you can choose the target type to run the operator, for example:

    none                 Run on CPU. (Supported by all devices, Android/Linux/Windows)
    neon                 Run on Android  CPU with NEON   support.
    opencl               Run on Android  GPU with OpenCL support.
    hvx                  Run on Qualcomm DSP with HVX    support.
)";

aura::Status CvtColorSampleTest(aura::Context *ctx, aura::TargetType type)
{
    AURA_LOGD(ctx, SAMPLE_TAG, "=================== CvtColorSampleTest Begin ===================\n");

    // ---------------------
    //     prepare data
    // ---------------------
    // create mat
    aura::Sizes3 src_size = {487, 487, 1};
    aura::Sizes3 dst_size = {487, 487, 3};

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
        AURA_LOGE(ctx, SAMPLE_TAG, "CvtColor sample test load failed, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        return status;
    }

    std::vector<aura::Mat> vec_src = {src};
    std::vector<aura::Mat> vec_dst = {dst};

    // ---------------------
    //     run interface
    // ---------------------
    // print info
    AURA_LOGD(ctx, SAMPLE_TAG, "Test cvtcolor param detail:\n");
    AURA_LOGD(ctx, SAMPLE_TAG, "elem_type(%s), mat_size(%s), cvtcolor_type(%s), target(%s)\n",
              ElemTypesToString(src.GetElemType()).c_str(), src.GetSizes().ToString().c_str(),
              CvtColorTypeToString(aura::CvtColorType::GRAY2RGB).c_str(), TargetTypeToString(type).c_str());

    // run and time
    aura::Time start_time = aura::Time::Now();
    status = ICvtColor(ctx, vec_src, vec_dst, aura::CvtColorType::GRAY2RGB, aura::OpTarget(type));
    aura::Time end_time = aura::Time::Now();

    // ---------------------
    //     check result
    // ---------------------
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "CvtColor Interface Execute Fail, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        AURA_LOGE(ctx, SAMPLE_TAG, "=================== CvtColorSampleTest: Test Failed ===================\n");
    }
    else
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "CvtColor running time = %s\n", (end_time - start_time).ToString().c_str());
        dst.Dump("./cvtcolor_test.raw");
        AURA_LOGD(ctx, SAMPLE_TAG, "result stored at ./cvtcolor_test.raw\n");
        AURA_LOGD(ctx, SAMPLE_TAG, "=================== CvtColorSampleTest: Test Succeeded ===================\n");
    }

    return status;
}

const static std::map<std::string, SampleOpsFunc> g_func_map = {{"cvtcolor", CvtColorSampleTest}};

MI_S32 main(MI_S32 argc, MI_CHAR *argv[])
{
    SampleOpsFunc sample_func;
    aura::TargetType type;

    // create context for sample
    std::shared_ptr<aura::Context> ctx = CreateContext();
    if (nullptr == ctx)
    {
        return -1;
    }

    // parse inputs
    if (InputParser(argc, argv, help_info, g_func_map, sample_func, type) != aura::Status::OK)
    {
        AURA_LOGE(ctx.get(), SAMPLE_TAG, "InputParser failed\n");
        return -1;
    }

    // run cvtcolor sample
    aura::Status ret = sample_func(ctx.get(), type);
    if (aura::Status::OK == ret)
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "=================== sample_cvtcolor execution succeeded ===================");
        return 0;
    }
    else
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "=================== sample_cvtcolor execution failed ===================");
        return -1;
    }
}