#ifndef AURA_RUNTIME_CONTEXT_HOST_CONTEXT_IMPL_HPP__
#define AURA_RUNTIME_CONTEXT_HOST_CONTEXT_IMPL_HPP__

#include "aura/runtime/context/host/context.hpp"
#include "aura/runtime/logger.h"
#include "aura/runtime/memory.h"
#include "aura/runtime/worker_pool.h"
#if defined(AURA_BUILD_ANDROID)
#  include "aura/runtime/systrace.h"
#endif // AURA_BUILD_ANDROID
#if defined(AURA_ENABLE_HEXAGON)
#  include "aura/runtime/hexagon.h"
#endif // AURA_ENABLE_HEXAGON
#if defined(AURA_ENABLE_OPENCL)
#  include "aura/runtime/opencl.h"
#endif // AURA_ENABLE_OPENCL
#if defined(AURA_ENABLE_NN)
#  include "aura/runtime/nn.h"
#endif // AURA_ENABLE_NN
#if defined(AURA_ENABLE_XTENSA)
#  include "aura/runtime/xtensa.h"
#endif // AURA_ENABLE_XTENSA

#include <memory>

namespace aura
{

class Context::Impl
{
public:
    Impl(const Config &config);
    ~Impl();

    Status Initialize(Context *ctx);

    std::string GetVersion();

    Logger* GetLogger();

    MemPool* GetMemPool();

    WorkerPool* GetWorkerPool();

#if defined(AURA_BUILD_ANDROID)
    Systrace* GetSystrace();
#endif // AURA_BUILD_ANDROID

#if defined(AURA_ENABLE_OPENCL)
    CLEngine* GetCLEngine();
#endif // AURA_ENABLE_OPENCL

#if defined(AURA_ENABLE_HEXAGON)
    HexagonEngine* GetHexagonEngine();
#endif // AURA_ENABLE_HEXAGON

#if defined(AURA_ENABLE_NN)
    NNEngine* GetNNEngine();
#endif // AURA_ENABLE_NN

#if defined(AURA_ENABLE_XTENSA)
    XtensaEngine* GetXtensaEngine();
#endif // AURA_ENABLE_XTENSA

private:
    std::shared_ptr<Logger>     m_logger;
    std::shared_ptr<MemPool>    m_mem_pool;
    std::shared_ptr<WorkerPool> m_wp;

#if defined(AURA_BUILD_ANDROID)
    std::shared_ptr<Systrace> m_systrace;
#endif // AURA_BUILD_ANDROID

#if defined(AURA_ENABLE_OPENCL)
    std::shared_ptr<CLEngine> m_cl_engine;
#endif // AURA_ENABLE_OPENCL

#if defined(AURA_ENABLE_HEXAGON)
    std::shared_ptr<HexagonEngine>   m_hexagon_engine;
#endif // AURA_ENABLE_HEXAGON

#if defined(AURA_ENABLE_NN)
    std::shared_ptr<NNEngine> m_nn_engine;
#endif

#if defined(AURA_ENABLE_XTENSA)
    std::shared_ptr<XtensaEngine>   m_xtensa_engine;
#endif // AURA_ENABLE_XTENSA

    Config m_config;
    MI_BOOL m_flag;
};

} // namespace aura

#endif // AURA_RUNTIME_CONTEXT_HOST_CONTEXT_IMPL_HPP__