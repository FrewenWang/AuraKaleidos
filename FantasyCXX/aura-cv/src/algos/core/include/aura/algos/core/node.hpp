
#ifndef AURA_ALGOS_CORE_NODE_HPP__
#define AURA_ALGOS_CORE_NODE_HPP__

#include "aura/algos/core/graph.hpp"

namespace aura
{

class AURA_EXPORTS Node
{
public:
    Node();
    Node(Context *ctx, Op *op, NodeType type, Graph *graph, const std::string &name);
    AURA_DISABLE_COPY_AND_ASSIGN(Node);

    template <typename Tp, typename ...ArgsType>
    Status SyncInitialize(ArgsType &&...args);
    Status SyncRun();
    Status SyncDeInitialize();
    template <typename Tp, typename ...ArgsType>
    Status SyncCall(ArgsType &&...args);

#if defined(AURA_BUILD_HOST)
    template <typename Tp, typename ...ArgsType>
    std::shared_future<Status> AsyncInitialize(ArgsType &&...args);
    std::shared_future<Status> AsyncRun();
    std::shared_future<Status> AsyncDeInitialize();
    template <typename Tp, typename ...ArgsType>
    std::shared_future<Status> AsyncCall(ArgsType &&...args);
#endif // AURA_BUILD_HOST

    Op* GetOp();
    NodeType GetType() const;
    std::string GetName() const;

    AURA_VOID SetPrint(const std::vector<std::string> &print_props);
    AURA_VOID SetDump(const std::vector<std::string> &dump_props);

    template <typename ...ArgsType>
    Status BindDump(ArgsType &&...args);

    template <typename ...ArgsType>
    Status SetInputArrays(ArgsType &&...args);

    template <typename ...ArgsType>
    Status SetOutputArrays(ArgsType &&...args);

private:
    MI_BOOL IsValid() const;
    MI_BOOL IsMatch(const std::vector<std::string> &props) const;

    Context     *m_ctx;
    NodeType    m_type;
    Op          *m_op;
    std::string m_name;
    Graph       *m_graph;
    MI_BOOL     m_enable_print;
    MI_BOOL     m_enable_dump;
};

template <typename Tp, typename ...ArgsType>
Status Node::SyncInitialize(ArgsType &&...args)
{
    Status ret = Status::ERROR;

    if (!IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid node");
        return Status::ERROR;
    }

    if (m_graph->m_timer->IsTimedOut())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "timed out");
        return Status::ERROR;
    }

    Tp *op = dynamic_cast<Tp*>(m_op);
    if (MI_NULL == op)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "null ptr");
        return Status::ERROR;
    }

    ret = op->SetArgs(std::forward<ArgsType>(args)...);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArgs failed");
        return Status::ERROR;
    }

    Time start = m_graph->m_timer->Now();
    ret = op->Initialize();
    if (Status::ERROR == ret)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Initialize failed");
    }
    Time end = m_graph->m_timer->Now();

    if (m_enable_print)
    {
        AURA_LOGD(m_ctx, AURA_TAG, "[%s] : Initialize(%s) result(%s)\n", GetName().c_str(),
                  (end - start).ToString().c_str(), StatusToString(ret).c_str());
    }

    ret |= m_graph->m_profiler->AddNodeProfiling(this, start, end, ret, NodeExecType::INITIALIZE);

    AURA_RETURN(m_ctx, ret);
}

template <typename Tp, typename ...ArgsType>
Status Node::SyncCall(ArgsType &&...args)
{
    Status ret = Status::ERROR;

    if (!IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid node");
        return Status::ERROR;
    }

    ret = SyncInitialize<Tp, ArgsType...>(std::forward<ArgsType>(args)...);
    if (ret != Status::OK)
    {
        if (ret == Status::ERROR)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "SyncInitialize failed");
        }
        return ret;
    }

    ret = SyncRun();
    if (ret != Status::OK)
    {
        if (ret == Status::ERROR)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "SyncRun failed");
        }
        return ret;
    }

    ret = SyncDeInitialize();
    if (ret != Status::OK)
    {
        if (ret == Status::ERROR)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "SyncDeInitialize failed");
        }
        return ret;
    }

    AURA_RETURN(m_ctx, ret);
}

template <typename ...ArgsType>
Status Node::BindDump(ArgsType &&...args)
{
    if (m_type != NodeType::FUNCTION)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Unsupport node type");
        return Status::ABORT;
    }

    Function *func = dynamic_cast<Function*>(m_op);
    return func->BindDump(std::forward<ArgsType>(args)...);
}

template <typename ...ArgsType>
Status Node::SetInputArrays(ArgsType &&...args)
{
    if (m_type != NodeType::FUNCTION)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Unsupport node type");
        return Status::ABORT;
    }

    Function *func = dynamic_cast<Function*>(m_op);
    return func->SetInputArrays(std::forward<ArgsType>(args)...);
}

template <typename ...ArgsType>
Status Node::SetOutputArrays(ArgsType &&...args)
{
    if (m_type != NodeType::FUNCTION)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Unsupport node type");
        return Status::ABORT;
    }

    Function *func = dynamic_cast<Function*>(m_op);
    return func->SetOutputArrays(std::forward<ArgsType>(args)...);
}

#if defined(AURA_BUILD_HOST)
template <typename Tp, typename ...ArgsType>
std::shared_future<Status> Node::AsyncInitialize(ArgsType &&...args)
{
    if (!IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid node");
        return std::shared_future<Status>();
    }

    auto func = [=] () mutable {return SyncInitialize<Tp, ArgsType...>(std::forward<ArgsType>(args)...);};
    std::shared_future<Status> token = m_ctx->GetWorkerPool()->AsyncRun(func);
    if (!token.valid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "WorkerPool::AsyncRun failed");
        return std::shared_future<Status>();
    }

    MI_U64 thread_id = std::hash<std::thread::id>()(std::this_thread::get_id());
    m_graph->m_tokens[thread_id].push_back(token);

    return token;
}

template <typename Tp, typename ...ArgsType>
std::shared_future<Status> Node::AsyncCall(ArgsType &&...args)
{
    if (!IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid node");
        return std::shared_future<Status>();
    }

    auto func = [=] () mutable {return SyncCall<Tp, ArgsType...>(std::forward<ArgsType>(args)...);};
    std::shared_future<Status> token = m_ctx->GetWorkerPool()->AsyncRun(func);
    if (!token.valid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "WorkerPool::AsyncRun failed");
        return std::shared_future<Status>();
    }

    MI_U64 thread_id = std::hash<std::thread::id>()(std::this_thread::get_id());
    m_graph->m_tokens[thread_id].push_back(token);

    return token;
}
#endif // AURA_BUILD_HOST

} // namespace aura

#endif // AURA_ALGOS_CORE_NODE_HPP__