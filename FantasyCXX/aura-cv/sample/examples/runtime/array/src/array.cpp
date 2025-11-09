#include "sample_array.hpp"

static std::string help_info = R"(
Usage:
    Usage: sample_array [MatType]

Example usage:
    Usage: ./sample_array mat

MatType:
    This module contains two MatType below:
    mat                      Introducing usage of Mat class.
    cl_mat                   Introducing usage of CLMat class.
)";

const static std::map<std::string, SampleRuntimeFunc> g_func_map = {
    {"mat",    MatSampleTest},
    {"cl_mat", CLMemSampleTest},
};

static aura::Status InputParser(DT_S32 argc, DT_CHAR *argv[], SampleRuntimeFunc &func)
{
    if (argc != 2)
    {
        PrintHelpInfo(help_info);
        return aura::Status::ERROR;
    }

    // find sample function
    std::string op = std::string(argv[1]);
    std::transform(op.begin(), op.end(), op.begin(), ::tolower);
    auto it = g_func_map.find(op);
    if (it == g_func_map.end())
    {
        PrintHelpInfo(help_info);
        return aura::Status::ERROR;
    }
    func = it->second;

    return aura::Status::OK;
}

DT_S32 main(DT_S32 argc, DT_CHAR *argv[])
{
    SampleRuntimeFunc sample_func;

    // create context for sample
    std::shared_ptr<aura::Context> ctx = CreateContext();
    if (nullptr == ctx)
    {
        return -1;
    }

    // parse inputs
    if (InputParser(argc, argv, sample_func) != aura::Status::OK)
    {
        AURA_LOGE(ctx.get(), SAMPLE_TAG, "InputParser failed\n");
        return -1;
    }

    // run array sample
    aura::Status ret = sample_func(ctx.get());
    if (aura::Status::OK == ret)
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "=================== sample_array execution succeeded ===================\n");
        return 0;
    }
    else
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "=================== sample_array execution failed ===================\n");
        return -1;
    }
}