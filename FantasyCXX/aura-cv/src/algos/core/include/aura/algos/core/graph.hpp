
#ifndef AURA_ALGOS_CORE_GRAPH_HPP__
#define AURA_ALGOS_CORE_GRAPH_HPP__

#include "aura/algos/core/function.hpp"
#include "aura/algos/core/profiler.hpp"
#include "aura/algos/core/timer.hpp"
#include "aura/runtime/mat.h"
#include "aura/runtime/worker_pool.h"
#if defined(AURA_ENABLE_OPENCL)
#  include "aura/runtime/cl_mem.h"
#endif // AURA_ENABLE_OPENCL
#include "aura/ops/core.h"

#include <future>
#include <unordered_map>

namespace aura
{

enum class NodeType
{
    INVALID = 0,
    OP,
    ALGO,
    FUNCTION,
};

AURA_INLINE std::string NodeTypesToString(NodeType node_type)
{
    switch (node_type)
    {
        case NodeType::OP:
        {
            return "OP";
        }

        case NodeType::ALGO:
        {
            return "ALGO";
        }

        case NodeType::FUNCTION:
        {
            return "FUNCTION";
        }

        case NodeType::INVALID:
        {
            return "INVALID";
        }
    }

    return "Invalid node_type";
}

class Algo;
class Node;


class AURA_EXPORTS Graph
{
public:
    template <typename Tp>
    struct OpTraits
    {
        static constexpr DT_BOOL value = std::is_base_of<Op, Tp>::value &&
                                         !std::is_base_of<Algo, Tp>::value &&
                                         !std::is_base_of<Function, Tp>::value;
    };

    template <typename Tp>
    struct FunctionTraits
    {
        static constexpr DT_BOOL value = std::is_base_of<Function, Tp>::value;
    };

    template <typename Tp>
    struct AlgoTraits
    {
        static constexpr DT_BOOL value = std::is_base_of<Algo, Tp>::value;
    };

    Graph(Context *ctx);

#if defined(AURA_BUILD_HOST)
    Graph(Context *ctx, const std::unordered_map<std::string, std::vector<std::string>> &props);
#elif defined(AURA_BUILD_HEXAGON)
    Graph(Context *ctx, const std::string &name, const Time &end_time, const Time &host_base_time,
          const std::unordered_map<std::string, std::vector<std::string>> &props);
#endif

    ~Graph();
    AURA_DISABLE_COPY_AND_ASSIGN(Graph);

    template <typename Tp, typename ...ArgsType, typename std::enable_if<OpTraits<Tp>::value>::type* = DT_NULL>
    Node* MakeNode(const std::string &name, ArgsType &&...args);

    template <typename Tp, typename ...ArgsType, typename std::enable_if<FunctionTraits<Tp>::value>::type* = DT_NULL>
    Node* MakeNode(const std::string &name, ArgsType &&...args);

    template <typename Tp, typename ...ArgsType, typename std::enable_if<AlgoTraits<Tp>::value>::type* = DT_NULL>
    Node* MakeNode(const std::string &name, ArgsType &&...args);

    DT_VOID MakeNodes(Node *node, const std::vector<std::string> &names);

    Node& operator[](const std::string &name);
    Status Finalize();

#if defined(AURA_BUILD_HOST)
    Status SetTimeout(DT_S32 timeout_ms);
    Status SaveProfiling();
#endif // AURA_BUILD_HOST
    Status SetOutputPath(const std::string &output_dir, const std::string &output_prefix = std::string());

#if defined(AURA_BUILD_HOST)
    template <typename FuncType, typename ...ArgsType>
    std::shared_future<Status> AsyncRun(FuncType &&f, ArgsType &&...args);
    Status Barrier();
#elif defined(AURA_BUILD_HEXAGON)
    Profiler* GetProfiler();
#endif // AURA_BUILD_HEXAGON

    Context* GetContext();

#if defined(AURA_ENABLE_HEXAGON)
    Status CallHexagon(const std::string &package, const std::string &module, OpImpl *op_impl,
                       HexagonRpcParam &rpc_param, HexagonProfiling *profiling = DT_NULL);
#endif // AURA_ENABLE_HEXAGON

    // create mat
    Mat* CreateMat(const std::string &name, ElemType elem_type, const Sizes3 &sizes,
                   DT_S32 mem_type = AURA_MEM_DEFAULT, const Sizes &strides = Sizes());
    Mat* CreateMat(const std::string &name, ElemType elem_type, const Sizes3 &sizes,
                   const Buffer &buffer, const Sizes &strides = Sizes());
    Mat* CreateMat(const std::string &name, const Mat *src, const Rect &roi);
    Mat* CloneMat(const std::string &name, const Mat *src, const Rect &roi = Rect(), const Sizes &strides = Sizes());

#if defined(AURA_ENABLE_OPENCL)
    // create clmem
    CLMem* CreateClMem(const std::string &name, const CLMemParam &cl_param, ElemType elem_type,
                       const Sizes3 &sizes, const Sizes &strides = Sizes());
    CLMem* CreateClMem(const std::string &name, const CLMemParam &cl_param, ElemType elem_type,
                       const Sizes3 &sizes, const Buffer &buffer, const Sizes &strides = Sizes());
    CLMem* CreateClMem(const std::string &name, const CLMemParam &cl_param, const Mat *mat);
#endif // AURA_ENABLE_OPENCL

    template <typename Tp>
    DT_VOID DeleteArray(Tp **array);

    template <typename Tp, typename ...Tpn>
    DT_VOID DeleteArray(Tp **array, Tpn **...arrays)
    {
        DeleteArray(array);
        DeleteArray(arrays...);
    }

    // create buffer
    Buffer CreateBuffer(const std::string &name, DT_S64 size, DT_S32 type = AURA_MEM_DEFAULT, DT_S32 align = 0);
    DT_VOID DeleteBuffer(Buffer &buffer);

    Buffer AddExternalMem(const std::string &name, DT_S32 type, DT_S64 size, DT_VOID *data, DT_S32 property);

    Status AddExternalArray(const std::string &name, const Array *array);

    template <typename Tp, typename std::enable_if<std::is_base_of<Array, Tp>::value>::type* = DT_NULL>
    Status AddExternalArray(const std::vector<std::string> &names, const std::vector<Tp*> &arrays);

private:
    friend class Node;

    class Config
    {
    public:
        Config(const std::unordered_map<std::string, std::vector<std::string>> &props);
#if defined(AURA_BUILD_ANDROID)
        Config(const std::string &root_name);
#endif // AURA_BUILD_ANDROID
        DT_VOID SetOutputPath(const std::string &output_dir, const std::string &output_prefix);

        std::string m_output_dir;
        std::string m_output_prefix;
        std::unordered_map<std::string, std::vector<std::string>> m_props;

    private:
#if defined(AURA_BUILD_ANDROID)
        DT_VOID GetPropsFromFile(const std::string &config_key, const std::unordered_map<std::string, std::string> &keys_map);
        DT_VOID GetPropsFromShell(const std::unordered_map<std::string, std::string> &keys_map);
#endif // AURA_BUILD_ANDROID

        std::vector<std::string> ParseProps(const std::string &props);
    };

    template <typename Tp, typename ...ArgsType>
    Tp* CreateArrayImpl(const std::string &name, ArgsType &&...args);

    template <typename Tp, typename ...ArgsType>
    Node *MakeNodeImpl(const std::string &name, NodeType type, ArgsType &&...args);

    DT_BOOL CheckValid();
    Status UpdateNodes();

    Context *m_ctx;
    std::string m_name;
    DT_BOOL m_is_valid;
    std::vector<Op*> m_ops;
    std::shared_ptr<Config> m_config;
    std::shared_ptr<Profiler> m_profiler;
    std::shared_ptr<Timer> m_timer;
    std::unordered_map<std::string, Node*> m_nodes;
#if defined(AURA_BUILD_HOST)
    std::unordered_map<DT_U64, std::vector<std::shared_future<Status>>> m_tokens;
#endif // AURA_BUILD_HOST
    static Node m_dummy_node;
};

template <typename Tp, typename ...ArgsType, typename std::enable_if<Graph::OpTraits<Tp>::value>::type*>
Node* Graph::MakeNode(const std::string &name, ArgsType &&...args)
{
    return MakeNodeImpl<Tp>(name, NodeType::OP, std::forward<ArgsType>(args)...);
}

template <typename Tp, typename ...ArgsType, typename std::enable_if<Graph::FunctionTraits<Tp>::value>::type*>
Node* Graph::MakeNode(const std::string &name, ArgsType &&...args)
{
    return MakeNodeImpl<Tp>(name, NodeType::FUNCTION, std::forward<ArgsType>(args)...);
}

template <typename Tp, typename ...ArgsType, typename std::enable_if<Graph::AlgoTraits<Tp>::value>::type*>
Node* Graph::MakeNode(const std::string &name, ArgsType &&...args)
{
    return MakeNodeImpl<Tp>(name, NodeType::ALGO, std::forward<ArgsType>(args)...);
}

template <typename Tp, typename ...ArgsType>
Node* Graph::MakeNodeImpl(const std::string &name, NodeType type, ArgsType &&...args)
{
    if (!m_is_valid)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid graph");
        return DT_NULL;
    }

    if (m_nodes.find(name) != m_nodes.end() || name.find('.') != std::string::npos)
    {
        AURA_ADD_ERROR_STRING(m_ctx, ("invalid name " + name).c_str());
        m_is_valid = DT_FALSE;
        return DT_NULL;
    }

    Op *op = Create<Tp>(m_ctx, args...);
    m_ops.push_back(op);
    Node *node = Create<Node>(m_ctx, op, type, this, name);
    m_nodes[name] = node;

    if (DT_NULL == op || DT_NULL == node)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "null ptr");
        m_is_valid = DT_FALSE;
        return DT_NULL;
    }

    return node;
}

#if defined(AURA_BUILD_HOST)
template <typename FuncType, typename ...ArgsType>
std::shared_future<Status> Graph::AsyncRun(FuncType &&f, ArgsType &&...args)
{
    if (!m_is_valid)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid graph");
        return std::shared_future<Status>();
    }

    std::shared_future<Status> token = m_ctx->GetWorkerPool()->AsyncRun(std::forward<FuncType>(f), std::forward<ArgsType>(args)...);
    if (!token.valid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "WorkerPool::AsyncRun failed");
        return std::shared_future<Status>();
    }

    DT_U64 thread_id = std::hash<std::thread::id>()(std::this_thread::get_id());
    m_tokens[thread_id].push_back(token);

    return token;
}
#endif // AURA_BUILD_HOST

template <typename Tp, typename ...ArgsType>
Tp* Graph::CreateArrayImpl(const std::string &name, ArgsType &&...args)
{
    if (!m_is_valid)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid graph");
        return DT_NULL;
    }

    if (m_timer && m_timer->IsTimedOut())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "timed out");
        return DT_NULL;
    }

    if (name.find('.') != std::string::npos)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid name");
        return DT_NULL;
    }

    Time start, end;
    if (m_timer)
    {
        start = m_timer->Now();
    }
    Tp *array = Create<Tp>(m_ctx, args...);
    if (DT_NULL == array)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "null ptr");
        return DT_NULL;
    }
    if (m_timer)
    {
        end = m_timer->Now();
    }

    if (m_profiler)
    {
        std::string full_name = m_name.empty() ? name : m_name + "." + name;
        Status ret = m_profiler->AddCreateArrayProfiling(full_name, array, start, end);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "AddCreateArrayProfiling failed");
            Delete<Tp>(m_ctx, &array);
            return DT_NULL;
        }
    }

    return array;
}

template <typename Tp>
DT_VOID Graph::DeleteArray(Tp **array)
{
    if (!m_is_valid)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid graph");
        return;
    }

    Array *array_backup = DT_NULL;
    Buffer buffer_backup;
    if (array != DT_NULL && *array != DT_NULL)
    {
        array_backup = *array;
        buffer_backup = (*array)->GetBuffer();
    }

    Time start, end;
    if (m_timer)
    {
        start = m_timer->Now();
    }
    Delete<Tp>(m_ctx, array);
    if (m_timer)
    {
        end = m_timer->Now();
    }

    if (m_profiler)
    {
        m_profiler->AddDeleteArrayProfiling(array_backup, buffer_backup, start, end);
    }
}

template <typename Tp, typename std::enable_if<std::is_base_of<Array, Tp>::value>::type*>
Status Graph::AddExternalArray(const std::vector<std::string> &names, const std::vector<Tp*> &arrays)
{
    Status ret = Status::OK;

    if (names.size() != arrays.size())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "The length of names does not match the length of arrays");
        return Status::ERROR;
    }

    for (size_t index = 0; index < names.size(); ++index)
    {
        ret |= AddExternalArray(names[index], arrays[index]);
    }

    return ret;
}

} // namespace aura

#endif // AURA_ALGOS_CORE_GRAPH_HPP__