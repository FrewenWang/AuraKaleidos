#include "aura/algos/core/node.hpp"

namespace aura
{

Node::Node() : m_ctx(MI_NULL), m_type(NodeType::INVALID), m_op(MI_NULL),
               m_graph(MI_NULL), m_enable_print(MI_FALSE), m_enable_dump(MI_FALSE)
{}

Node::Node(Context *ctx, Op *op, NodeType type, Graph *graph, const std::string &name)
           : m_ctx(ctx), m_type(type), m_op(op), m_name(name), m_graph(graph),
             m_enable_print(MI_FALSE), m_enable_dump(MI_FALSE)
{}

Status Node::SyncRun()
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

    Time start = m_graph->m_timer->Now();
    ret = m_op->Run();
    if (Status::ERROR == ret)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Run failed");
    }
    Time end = m_graph->m_timer->Now();

    if (m_enable_print)
    {
        AURA_LOGD(m_ctx, AURA_TAG, "[%s] : Run(%s) result(%s) info(%s)\n", GetName().c_str(),
                  (end - start).ToString().c_str(), StatusToString(ret).c_str(), m_op->ToString().c_str());
    }

    if (m_enable_dump)
    {
        Time now = m_graph->m_timer->Now();
        std::stringstream os;
        os << now.sec << std::setfill('0') << std::setw(3) << now.ms << std::setw(3) << now.us;

        std::string dump_prefix = m_graph->m_config->m_output_dir + m_graph->m_config->m_output_prefix;
        dump_prefix += GetName() + "_" + os.str();

        m_op->Dump(dump_prefix);
    }

    ret |= m_graph->m_profiler->AddNodeProfiling(this, start, end, ret, NodeExecType::RUN);

    AURA_RETURN(m_ctx, ret);
}

Status Node::SyncDeInitialize()
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

    Time start = m_graph->m_timer->Now();
    ret = m_op->DeInitialize();
    if (Status::ERROR == ret)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "DeInitialize failed");
    }
    Time end = m_graph->m_timer->Now();

    if (m_enable_print)
    {
        AURA_LOGD(m_ctx, AURA_TAG, "[%s] : DeInitialize(%s) result(%s)\n", GetName().c_str(),
                  (end - start).ToString().c_str(), StatusToString(ret).c_str());
    }

    ret |= m_graph->m_profiler->AddNodeProfiling(this, start, end, ret, NodeExecType::DEINITIALIZE);

    AURA_RETURN(m_ctx, ret);
}

#if defined(AURA_BUILD_HOST)
std::shared_future<Status> Node::AsyncRun()
{
    if (!IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid node");
        return std::shared_future<Status>();
    }

    std::shared_future<Status> token = m_ctx->GetWorkerPool()->AsyncRun(std::bind(&Node::SyncRun, this));
    if (!token.valid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "WorkerPool::AsyncRun failed");
        return std::shared_future<Status>();
    }

    MI_U64 thread_id = std::hash<std::thread::id>()(std::this_thread::get_id());
    m_graph->m_tokens[thread_id].push_back(token);

    return token;
}

std::shared_future<Status> Node::AsyncDeInitialize()
{
    if (!IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid node");
        return std::shared_future<Status>();
    }

    std::shared_future<Status> token = m_ctx->GetWorkerPool()->AsyncRun(std::bind(&Node::SyncDeInitialize, this));
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

Op* Node::GetOp()
{
    return m_op;
}

NodeType Node::GetType() const
{
    return m_type;
}

std::string Node::GetName() const
{
    if (!IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid node");
        return std::string();
    }

    if (m_graph->m_name.empty())
    {
        return m_name;
    }
    return m_graph->m_name + "." + m_name;
}

AURA_VOID Node::SetPrint(const std::vector<std::string> &print_props)
{
    m_enable_print = IsMatch(print_props);
}

AURA_VOID Node::SetDump(const std::vector<std::string> &dump_props)
{
    m_enable_dump = IsMatch(dump_props);
}

MI_BOOL Node::IsValid() const
{
    return (m_ctx && (m_type != NodeType::INVALID) && m_op && m_graph);
}

MI_BOOL Node::IsMatch(const std::vector<std::string> &props) const
{
    for (const auto &prop : props)
    {
        if (prop.back() == '*')
        {
            std::string str = prop.substr(0, prop.size() - 1);
            if (GetName().substr(0, str.size()) == str)
            {
                return MI_TRUE;
            }
        }
        else
        {
            if (GetName() == prop)
            {
                return MI_TRUE;
            }
        }
    }
    return MI_FALSE;
}

} // namespace aura