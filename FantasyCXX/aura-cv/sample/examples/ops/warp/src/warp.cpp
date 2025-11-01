#include "sample_warp.hpp"

static std::string help_info = R"(
Usage:
    Usage: sample_warp [Operators] [TargetType]

Example usage:
    Usage: ./sample_warp remap none 

Operators:
    This module contains warp operators, for example:
    remap                  Apply a remap operation to the source matrix using the provided mapping matrix.
    warpaffine             Apply an affine warp to the src matrix using the provided affine transformation matrix.
    warpperspective        Apply a perspective warp to the src matrix using the provided perspective transformation matrix

TargetType:
    These operators run on different hardware(CPU, GPU and DSP),
    you can choose the target type to run the operator, for example:

    none                 Run on CPU. (Supported by all devices, Android/Linux/Windows)
    neon                 Run on Android  CPU with NEON   support.
    opencl               Run on Android  GPU with OpenCL support.
    hvx                  Run on Qualcomm DSP with HVX    support.
)";

const static std::map<std::string, SampleOpsFunc> g_func_map = {
    {"remap",           RemapSampleTest},
    {"warpaffine",      WarpAffineSampleTest},
    {"warpperspective", WarpPerspectiveSampleTest}
};

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

    // run warp sample
    aura::Status ret = sample_func(ctx.get(), type);
    if (aura::Status::OK == ret)
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "=================== sample_warp execution succeeded ===================");
        return 0;
    }
    else
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "=================== sample_warp execution failed ===================");
        return -1;
    }
}