#include "sample_ops.hpp"

static std::string help_info = R"(
Usage:
    Usage: sample_morph [Operators] [TargetType]

Example usage:
    Usage: ./sample_morph morph none

Operators:
    This module contains morph operators, for example:
    morph             In this sample, we will run morph operator.

TargetType:
    These operators run on different hardware(CPU, GPU and DSP),
    you can choose the target type to run the operator, for example:

    none                 Run on CPU. (Supported by all devices, Android/Linux/Windows)
    neon                 Run on Android  CPU with NEON   support.
    opencl               Run on Android  GPU with OpenCL support.
    hvx                  Run on Qualcomm DSP with HVX    support.
)";

aura::Status MorphSampleTest(aura::Context *ctx, aura::TargetType type)
{
    AURA_LOGD(ctx, SAMPLE_TAG, "=================== MorphSampleTest Begin ===================\n");

    // ---------------------
    //     prepare data
    // ---------------------
    // create mat
    aura::Sizes3 size = {487, 487, 1};
    aura::Mat src(ctx, aura::ElemType::U8, size);
    aura::Mat dst(ctx, aura::ElemType::U8, size);
    if (!(src.IsValid() && dst.IsValid()))
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "src or dst mat is invalid.");
        return aura::Status::ERROR;
    }

    // load data
    aura::Status status = src.Load("data/comm/cameraman_487x487.gray");
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "Morph sample test load failed, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        return status;
    }

    MI_S32 ksize = 3;
    MI_S32 iter  = 10;
    // ---------------------
    //     run interface
    // ---------------------
    // print info
    AURA_LOGD(ctx, SAMPLE_TAG, "Test morph param detail:\n");
    AURA_LOGD(ctx, SAMPLE_TAG, "elem_type(%s), mat_size(%s), morph_type(%s), ksize(%d), morph_shape(%s), iter(%d), target(%s)\n",
              ElemTypesToString(src.GetElemType()).c_str(), src.GetSizes().ToString().c_str(),
              MorphTypeToString(aura::MorphType::ERODE).c_str(), ksize, MorphShapeToString(aura::MorphShape::RECT).c_str(),
              iter, TargetTypeToString(type).c_str());

    // run and time
    aura::Time start_time = aura::Time::Now();
    status = IMorphologyEx(ctx, src, dst, aura::MorphType::ERODE, ksize, aura::MorphShape::RECT, iter, aura::OpTarget(type));
    aura::Time end_time = aura::Time::Now();

    // ---------------------
    //     check result
    // ---------------------
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "Morph Interface Execute Fail, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        AURA_LOGE(ctx, SAMPLE_TAG, "=================== MorphSampleTest: Test Failed ===================\n");
    }
    else
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "Morph running time = %s\n", (end_time - start_time).ToString().c_str());
        dst.Dump("./morph_test.raw");
        AURA_LOGD(ctx, SAMPLE_TAG, "result stored at ./morph_test.raw\n");
        AURA_LOGD(ctx, SAMPLE_TAG, "=================== MorphSampleTest: Test Succeeded ===================\n");
    }

    return status;
}

const static std::map<std::string, SampleOpsFunc> g_func_map = {{"morph", MorphSampleTest}};

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

    // run morph sample
    aura::Status ret = sample_func(ctx.get(), type);
    if (aura::Status::OK == ret)
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "=================== sample_morph execution succeeded ===================");
        return 0;
    }
    else
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "=================== sample_morph execution failed ===================");
        return -1;
    }
}