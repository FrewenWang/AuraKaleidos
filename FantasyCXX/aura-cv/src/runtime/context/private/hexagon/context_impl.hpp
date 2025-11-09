#ifndef AURA_RUNTIME_CONTEXT_HEXAGON_CONTEXT_IMPL_HPP__
#define AURA_RUNTIME_CONTEXT_HEXAGON_CONTEXT_IMPL_HPP__

#include "aura/runtime/context/hexagon/context.hpp"
#include "aura/runtime/logger.h"
#include "aura/runtime/memory.h"
#include "aura/runtime/worker_pool.h"
#if defined(AURA_ENABLE_NN)
#  include "aura/runtime/nn.h"
#endif // AURA_ENABLE_NN
#include <unordered_map>

namespace aura
{

class Context::Impl
{
public:
    Impl();
    ~Impl();

    Status Initialize(Context *ctx, LogOutput output, LogLevel level, const std::string &file);

    std::string GetVersion();

    Logger* GetLogger();

    MemPool* GetMemPool();

    WorkerPool* GetWorkerPool();

#if defined(AURA_ENABLE_NN)
    NNEngine* GetNNEngine();
#endif // AURA_ENABLE_NN

    Status AddShareBuffer(const std::string &name, const Buffer &buffer);

    Buffer GetShareBuffer(const std::string &name);

    Status RemoveShareBuffer(const std::string &name);

private:
    std::shared_ptr<Logger>      m_logger;
    std::shared_ptr<MemPool>     m_mem_pool;
    std::shared_ptr<WorkerPool>  m_wp;
#if defined(AURA_ENABLE_NN)
    std::shared_ptr<NNEngine> m_nn_engine;
#endif // AURA_ENABLE_NN
    std::unordered_map<std::string, Buffer> m_share_buffer_map;

    DT_BOOL m_flag;
};

} // namespace aura

#endif // AURA_RUNTIME_CONTEXT_HEXAGON_CONTEXT_IMPL_HPP__