#include "host/context_impl.hpp"
#include "aura/version.h"

namespace aura
{

Context::Impl::Impl(const Config &config)
                    : m_logger(), m_mem_pool(), m_wp(),
#if defined(AURA_BUILD_ANDROID)
                    m_systrace(),
#endif // AURA_BUILD_ANDROID
#if defined(AURA_ENABLE_OPENCL)
                    m_cl_engine(),
#endif // AURA_ENABLE_OPENCL
                    m_config(config),
                    m_flag(MI_FALSE)
{}

Context::Impl::~Impl()
{}

Status Context::Impl::Initialize(Context *ctx)
{
    if (MI_FALSE == m_flag)
    {
        m_logger.reset(new Logger(m_config.m_log_output, m_config.m_log_level, m_config.m_log_file));
        if (MI_NULL == m_logger.get())
        {
            AURA_ADD_ERROR_STRING(ctx, "m_logger null ptr");
            return Status::ERROR;
        }

        m_mem_pool.reset(new MemPool(ctx));
        if (MI_NULL == m_mem_pool.get())
        {
            AURA_ADD_ERROR_STRING(ctx, "m_mem_pool null ptr");
            return Status::ERROR;
        }

#if !defined(AURA_BUILD_XPLORER)
        m_wp.reset(new WorkerPool(ctx, m_config.m_thread_tag, m_config.m_compute_affinity, m_config.m_async_affinity,
                                  m_config.m_compute_threads, m_config.m_async_threads));
        if (MI_NULL == m_wp.get())
        {
            AURA_ADD_ERROR_STRING(ctx, "m_wp null ptr");
            return Status::ERROR;
        }
#endif

#if defined(AURA_BUILD_ANDROID)
        m_systrace.reset(new Systrace(m_config.m_enable_systrace));
        if (MI_NULL == m_systrace.get())
        {
            AURA_ADD_ERROR_STRING(ctx, "m_systrace null ptr");
            return Status::ERROR;
        }
#endif // AURA_BUILD_ANDROID

#if defined(AURA_ENABLE_OPENCL)
        m_cl_engine.reset(new CLEngine(ctx, m_config.m_enable_cl, m_config.m_cl_cache_path, m_config.m_cl_cache_prefix,
                                       m_config.m_cl_precompiled_type, m_config.m_cl_precompiled_sources,
                                       m_config.m_cl_external_version, m_config.m_cl_perf_level,
                                       m_config.m_cl_priority_level));
        if (MI_NULL == m_cl_engine.get())
        {
            AURA_ADD_ERROR_STRING(ctx, "m_cl_engine null ptr");
            return Status::ERROR;
        }
#endif // AURA_ENABLE_OPENCL

#if defined(AURA_ENABLE_HEXAGON)
        m_hexagon_engine.reset(new HexagonEngine(ctx, m_config.m_enable_hexagon, m_config.m_unsigned_pd, m_config.m_hexagon_lib_prefix, m_config.m_async_call, 
                                                 m_config.m_hexagon_log_output, m_config.m_hexagon_log_level, m_config.m_hexagon_log_file));
        if (MI_NULL == m_hexagon_engine.get())
        {
            AURA_ADD_ERROR_STRING(ctx, "m_hexagon_engine null ptr");
            return Status::ERROR;
        }
#endif // AURA_ENABLE_HEXAGON

#if defined(AURA_ENABLE_NN)
        m_nn_engine.reset(new NNEngine(ctx, m_config.m_enable_nn));
        if (MI_NULL == m_nn_engine.get())
        {
            AURA_ADD_ERROR_STRING(ctx, "m_nn_engine null ptr");
            return Status::ERROR;
        }
#endif // AURA_ENABLE_NN

#if defined(AURA_ENABLE_XTENSA)
        m_xtensa_engine.reset(new XtensaEngine(ctx, m_config.m_enable_xtensa, m_config.m_pil_name, m_config.m_xtensa_priority_level));
        if (MI_NULL == m_xtensa_engine.get())
        {
            AURA_ADD_ERROR_STRING(ctx, "m_xtensa_engine null ptr");
            return Status::ERROR;
        }
#endif // AURA_ENABLE_XTENSA

        m_flag = MI_TRUE;
    }

    return Status::OK;
}

std::string Context::Impl::GetVersion()
{
    std::string aura_version(AURA_VERSION);
    aura_version.erase(aura_version.find_last_not_of(' ') + 1);
    return aura_version;
}

Logger* Context::Impl::GetLogger()
{
    return m_logger.get();
}

MemPool* Context::Impl::GetMemPool()
{
    return m_mem_pool.get();
}

WorkerPool* Context::Impl::GetWorkerPool()
{
    return m_wp.get();
}

#if defined(AURA_BUILD_ANDROID)
Systrace* Context::Impl::GetSystrace()
{
    return m_systrace.get();
}
#endif // AURA_BUILD_ANDROID

#if defined(AURA_ENABLE_OPENCL)
CLEngine* Context::Impl::GetCLEngine()
{
    return m_cl_engine.get();
}
#endif // AURA_ENABLE_OPENCL

#if defined(AURA_ENABLE_HEXAGON)
/// 获取对应的HVX 的引擎
HexagonEngine* Context::Impl::GetHexagonEngine()
{
    return m_hexagon_engine.get();
}
#endif // AURA_ENABLE_HEXAGON

#if defined(AURA_ENABLE_NN)
NNEngine* Context::Impl::GetNNEngine()
{
    return m_nn_engine.get();
}
#endif // AURA_ENABLE_NN

#if defined(AURA_ENABLE_XTENSA)
XtensaEngine* Context::Impl::GetXtensaEngine()
{
    return m_xtensa_engine.get();
}
#endif // AURA_ENABLE_XTENSA

Context::Context(const Config &config)
{
    m_impl.reset(new Context::Impl(config));
}

Context::~Context()
{}

MI_BOOL Context::IsPlatformSupported()
{
    return CpuInfo::Get().IsAtomicsSupported();
}

Status Context::Initialize()
{
    Status sts = Status::ERROR;
    if (m_impl)
    {
        sts = m_impl->Initialize(this);
    }
    return sts;
}

std::string Context::GetVersion() const
{
    if (m_impl)
    {
        return m_impl->GetVersion();
    }
    return "INVALID";
}

Logger* Context::GetLogger() const
{
    if (m_impl)
    {
        return m_impl->GetLogger();
    }
    return MI_NULL;
}

MemPool* Context::GetMemPool() const
{
    if (m_impl)
    {
        return m_impl->GetMemPool();
    }
    return MI_NULL;
}

WorkerPool* Context::GetWorkerPool() const
{
    if (m_impl)
    {
        return m_impl->GetWorkerPool();
    }
    return MI_NULL;
}

#if defined(AURA_BUILD_ANDROID)
Systrace* Context::GetSystrace() const
{
    if (m_impl)
    {
        return m_impl->GetSystrace();
    }
    return MI_NULL;
}
#endif // AURA_BUILD_ANDROID

#if defined(AURA_ENABLE_OPENCL)
CLEngine* Context::GetCLEngine() const
{
    if (m_impl)
    {
        return m_impl->GetCLEngine();
    }
    return MI_NULL;
}
#endif // AURA_ENABLE_OPENCL

#if defined(AURA_ENABLE_HEXAGON)
HexagonEngine* Context::GetHexagonEngine() const
{
    if (m_impl)
    {
        return m_impl->GetHexagonEngine();
    }
    return MI_NULL;
}
#endif // AURA_ENABLE_HEXAGON

#if defined(AURA_ENABLE_NN)
NNEngine* Context::GetNNEngine() const
{
    if (m_impl)
    {
        return m_impl->GetNNEngine();
    }
    return MI_NULL;
}
#endif // AURA_ENABLE_NN

#if defined(AURA_ENABLE_XTENSA)
XtensaEngine* Context::GetXtensaEngine() const
{
    if (m_impl)
    {
        return m_impl->GetXtensaEngine();
    }
    return MI_NULL;
}
#endif // AURA_ENABLE_XTENSA

} // namespace aura