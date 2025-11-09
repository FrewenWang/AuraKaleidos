#include "hexagon/context_impl.hpp"
#include "aura/version.h"

namespace aura
{

Context::Impl::Impl() : m_flag(DT_FALSE)
{}

Context::Impl::~Impl()
{}

Status Context::Impl::Initialize(Context *ctx, LogOutput output, LogLevel level, const std::string &file)
{
    if (m_flag)
    {
        AURA_ADD_ERROR_STRING(ctx, "repeat initialization");
        return Status::ERROR;
    }

    m_logger.reset(new Logger(output, level, file));
    if (DT_NULL == m_logger.get())
    {
        AURA_ADD_ERROR_STRING(ctx, "m_logger null ptr");
        return Status::ERROR;
    }

    m_mem_pool.reset(new MemPool(ctx));
    if (DT_NULL == m_mem_pool.get())
    {
        AURA_ADD_ERROR_STRING(ctx, "m_mem_pool null ptr");
        return Status::ERROR;
    }

    m_wp.reset(new WorkerPool(ctx));
    if (DT_NULL == m_wp.get())
    {
        AURA_ADD_ERROR_STRING(ctx, "m_wp null ptr");
        return Status::ERROR;
    }

#if defined(AURA_ENABLE_NN)
    m_nn_engine.reset(new NNEngine(ctx, DT_TRUE));
    if (DT_NULL == m_nn_engine.get())
    {
        AURA_ADD_ERROR_STRING(ctx, "m_nn_engine null ptr");
        return Status::ERROR;
    }
#endif // AURA_ENABLE_NN

    m_flag = DT_TRUE;

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

#if defined(AURA_ENABLE_NN)
NNEngine* Context::Impl::GetNNEngine()
{
    return m_nn_engine.get();
}
#endif // AURA_ENABLE_NN

Status Context::Impl::AddShareBuffer(const std::string &name, const Buffer &buffer)
{
    if (!buffer.IsValid() || m_share_buffer_map.count(name) > 0)
    {
        return Status::ERROR;
    }

    m_share_buffer_map[name] = buffer;

    return Status::OK;
}

Buffer Context::Impl::GetShareBuffer(const std::string &name)
{
    if (0 == m_share_buffer_map.count(name))
    {
        return Buffer();
    }
    else
    {
        return m_share_buffer_map[name];
    }
}

Status Context::Impl::RemoveShareBuffer(const std::string &name)
{
    if (0 == m_share_buffer_map.count(name))
    {
        return Status::ERROR;
    }
    else
    {
        m_share_buffer_map.erase(name);
        return Status::OK;
    }
}

Context::Context()
{
    m_impl.reset(new Context::Impl());
}

Context::~Context()
{}

Status Context::Initialize(LogOutput output, LogLevel level, const std::string &file)
{
    Status sts = Status::ERROR;
    if (m_impl)
    {
        sts = m_impl->Initialize(this, output, level, file);
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
    return DT_NULL;
}

MemPool* Context::GetMemPool() const
{
    if (m_impl)
    {
        return m_impl->GetMemPool();
    }
    return DT_NULL;
}

WorkerPool* Context::GetWorkerPool() const
{
    if (m_impl)
    {
        return m_impl->GetWorkerPool();
    }
    return DT_NULL;
}

#if defined(AURA_ENABLE_NN)
NNEngine* Context::GetNNEngine() const
{
    if (m_impl)
    {
        return m_impl->GetNNEngine();
    }
    return DT_NULL;
}
#endif // AURA_ENABLE_NN

Status Context::AddShareBuffer(const std::string &name, const Buffer &buffer)
{
    if (m_impl)
    {
        return m_impl->AddShareBuffer(name, buffer);
    }
    return Status::ERROR;
}

Buffer Context::GetShareBuffer(const std::string &name)
{
    if (m_impl)
    {
        return m_impl->GetShareBuffer(name);
    }
    return Buffer();
}

Status Context::RemoveShareBuffer(const std::string &name)
{
    if (m_impl)
    {
        return m_impl->RemoveShareBuffer(name);
    }
    return Status::ERROR;
}

} // namespace aura