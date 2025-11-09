#include "sample_feature2d.hpp"

static std::string help_info = R"(
Usage:
    Usage: sample_feature2d [Operators] [TargetType]

Example usage:
    Usage: ./sample_feature2d canny none 

Operators:
    This module contains common used feature detector operators, for example:
    canny                        Performs Canny edge detection on a single-channel iaura.
    fast                         This sample applies the FAST corner detector to the input iaura.
    harris                       This sample applies the Harris corner detector to the input iaura.
    tomasi                       Determines strong corners on an iaura.

TargetType:
    These operators run on different hardware(CPU, GPU and DSP),
    you can choose the target type to run the operator, for example:

    none                 Run on CPU. (Supported by all devices, Android/Linux/Windows)
    neon                 Run on Android  CPU with NEON   support.
    opencl               Run on Android  GPU with OpenCL support.
    hvx                  Run on Qualcomm DSP with HVX    support.
)";

const static std::map<std::string, SampleOpsFunc> g_func_map = {
    {"canny",  CannySampleTest },
    {"fast",   FastSampleTest  },
    {"harris", HarrisSampleTest},
    {"tomasi", TomasiSampleTest}
};

DT_S32 main(DT_S32 argc, DT_CHAR *argv[])
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

    // run feature2d sample
    aura::Status ret = sample_func(ctx.get(), type);
    if (aura::Status::OK == ret)
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "=================== sample_feature2d execution succeeded ===================");
        return 0;
    }
    else
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "=================== sample_feature2d execution failed ===================");
        return -1;
    }
}