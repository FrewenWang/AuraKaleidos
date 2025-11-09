#ifndef AURA_SAMPLE_HPP__
#define AURA_SAMPLE_HPP__

#include "aura/runtime.h"
#include "aura/config.h"
#include "aura/ops/core.h"

#define SAMPLE_TAG                                "aura_sample"
#define SPLIT_WIDTH                               (86)

AURA_INLINE DT_VOID PrintHeading(const std::string &heading)
{
    std::cout << std::endl << std::string(SPLIT_WIDTH, '=') << std::endl;
    std::cout << heading << std::endl;
    std::cout << std::string(SPLIT_WIDTH, '=') << std::endl;
}

AURA_INLINE DT_VOID PrintHelpInfo(const std::string &help_info)
{
    PrintHeading("Aura Sample Run Help Info");
    std::cout << help_info <<std::endl;
    std::cout << std::string(SPLIT_WIDTH, '=') << std::endl;
}

AURA_INLINE std::shared_ptr<aura::Context> CreateContext()
{
    aura::Config config;
    config.SetLog(aura::LogOutput::STDOUT, aura::LogLevel::DEBUG, std::string("log"));
    config.SetWorkerPool("AuraSample", aura::CpuAffinity::BIG, aura::CpuAffinity::LITTLE);

#if defined(AURA_ENABLE_OPENCL)
    config.SetCLConf(DT_TRUE, "/data/local/tmp", "aura_unit_test");
#endif

#if defined(AURA_ENABLE_HEXAGON)
    config.SetHexagonConf(DT_TRUE, DT_TRUE, "aura_hexagon");
#endif

#if defined(AURA_ENABLE_XTENSA)
    config.SetXtensaConf(DT_TRUE, "aura_xtensa_pil.so", XtensaPriorityLevel::PRIORITY_HIGH);
#endif

    std::shared_ptr<aura::Context> ctx = std::make_shared<aura::Context>(config);
    if (DT_NULL == ctx)
    {
        return DT_NULL;
    }

    if (ctx->Initialize() != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "aura::Context::Initialize() failed\n");
        return DT_NULL;
    }

    return ctx;
}

AURA_INLINE aura::Status GetTargetType(std::string target, aura::TargetType &type)
{
    // to lower
    std::transform(target.begin(), target.end(), target.begin(), ::tolower);

    // find target type by string compare
    if (target == "none")
    {
        type = aura::TargetType::NONE;
    }
    else if (target == "neon")
    {
        type = aura::TargetType::NEON;
    }
    else if (target == "opencl")
    {
        type = aura::TargetType::OPENCL;
    }
    else if (target == "hvx")
    {
        type = aura::TargetType::HVX;
    }
    else if (target == "vdsp")
    {
        type = aura::TargetType::VDSP;
    }
    else
    {
        std::cout << "Invalid target type, please try other TargetType." << std::endl;
        return aura::Status::ERROR;
    }

    // check type based on platform
#if defined(ANDROID)
#  if !defined(AURA_ENABLE_NEON)
    if (type == aura::TargetType::NEON)
    {
        std::cout << "NEON is not supported on this platform, please try other TargetType." << std::endl;
        return aura::Status::ERROR;
    }
#  endif

#  if !defined(AURA_ENABLE_OPENCL)
    if (type == aura::TargetType::OPENCL)
    {
        std::cout << "OpenCL is not supported on this platform, please try other TargetType." << std::endl;
        return aura::Status::ERROR;
    }
#  endif

#  if !defined(AURA_ENABLE_HEXAGON)
    if (type == aura::TargetType::HVX)
    {
        std::cout << "HVX is not supported on this platform, please try other TargetType." << std::endl;
        return aura::Status::ERROR;
    }
#  endif

#  if !defined(AURA_ENABLE_XTENSA)
    if (type == aura::TargetType::VDSP)
    {
        std::cout << "VDSP is not supported on this platform, please try other TargetType." << std::endl;
        return aura::Status::ERROR;
    }
#  endif

#else // non-Android platform, only support none method
    if (type != aura::TargetType::NONE)
    {
        std::cout << "Non-Android platform only support TargetType of none." << std::endl;
        return aura::Status::ERROR;
    }
#endif

    return aura::Status::OK;
}

#endif // AURA_SAMPLE_HPP__