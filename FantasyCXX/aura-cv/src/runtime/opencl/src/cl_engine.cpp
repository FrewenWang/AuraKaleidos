#include "aura/runtime/opencl/cl_engine.hpp"
#include "cl_runtime_impl.hpp"
#include "cl_program_string_register.hpp"
#include "aura/runtime/logger.h"

namespace aura
{

CLEngine::CLEngine(Context *ctx, MI_BOOL enable_cl, 
                   const std::string &cache_path,
                   const std::string &cache_prefix,
                   CLPrecompiledType cl_precompiled_type,
                   const std::string &precompiled_sources,
                   const std::string &external_version,
                   CLPerfLevel cl_perf_level,
                   CLPriorityLevel cl_priority_level)
                   : m_cl_runtime(),
                     m_cl_config(enable_cl, cache_path, cache_prefix, cl_precompiled_type,
                     precompiled_sources, external_version, cl_perf_level, cl_priority_level)
{
    do
    {
        // check ctx
        if (MI_NULL == ctx)
        {
            break;
        }

        if (MI_FALSE == enable_cl)
        {
            break;
        }

        CLProgramStringRegister();

        // get opencl cl_platform & cl_device
        std::shared_ptr<cl::Platform> cl_platform;
        std::shared_ptr<cl::Device>   cl_device;
        if (FindFirstGPU(cl_platform, cl_device) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "opencl find cl_device failed");
            break;
        }

        // create runtime by vendor info
        std::string device_name = cl_device->getInfo<CL_DEVICE_NAME>();
        if (device_name.find("Adreno") != std::string::npos)
        {
            m_cl_runtime.reset(new AdrenoCLRuntime(ctx, cl_platform, cl_device, m_cl_config));
        }
        else if (device_name.find("Mali") != std::string::npos)
        {
            m_cl_runtime.reset(new MaliCLRuntime(ctx, cl_platform, cl_device, m_cl_config));
        }
        else
        {
            AURA_ADD_ERROR_STRING(ctx, "opencl device only suppose Adreno/Mali");
            break;
        }

        // check runtime validity
        if (m_cl_runtime && m_cl_runtime->Initialize() != Status::OK)
        {
            m_cl_runtime.reset();
        }
    } while (0);
}

CLEngine::~CLEngine()
{
    if (m_cl_config.m_enable_cl)
    {
        m_cl_runtime.reset();
    }
}

std::shared_ptr<CLRuntime> CLEngine::GetCLRuntime()
{
    return m_cl_runtime;
}

std::shared_ptr<CLRuntime> CLEngine::GetCLRuntime() const
{
    return m_cl_runtime;
}

} // namespace aura