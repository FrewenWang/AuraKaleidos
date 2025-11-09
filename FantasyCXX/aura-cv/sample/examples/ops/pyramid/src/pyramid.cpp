#include "sample_pyramid.hpp"

static std::string help_info = R"(
Usage:
    Usage: sample_pyramid [Operators] [TargetType]

Example usage:
    Usage: ./sample_pyramid pyrdown none 

Operators:
    This module contains pyramid operators, for example:
    pyrdown                Apply a pyramid down operation to the src matrix.
    pyrup                  Apply a pyramid up   operation to the src matrix.

TargetType:
    These operators run on different hardware(CPU, GPU and DSP),
    you can choose the target type to run the operator, for example:

    none                 Run on CPU. (Supported by all devices, Android/Linux/Windows)
    neon                 Run on Android  CPU with NEON   support.
    opencl               Run on Android  GPU with OpenCL support.
    hvx                  Run on Qualcomm DSP with HVX    support.
)";

const static std::map<std::string, SampleOpsFunc> g_func_map = {
    {"pyrdown", PyrDownSampleTest},
    {"pyrup",   PyrUpSampleTest  }
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

    // run pyramid sample
    aura::Status ret = sample_func(ctx.get(), type);
    if (aura::Status::OK == ret)
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "=================== sample_pyramid execution succeeded ===================");
        return 0;
    }
    else
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "=================== sample_pyramid execution failed ===================");
        return -1;
    }
}