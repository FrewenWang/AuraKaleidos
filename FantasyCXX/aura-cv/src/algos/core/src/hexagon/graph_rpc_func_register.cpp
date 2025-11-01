#include "aura/algos/core/hexagon/graph_rpc_func_register.hpp"
#include "graph_rpc_impl.hpp"

namespace aura
{

std::unordered_map<std::string, GraphRpcFunc>& GetGraphRpcFuncMap()
{
    static std::unordered_map<std::string, GraphRpcFunc> graph_rpc_func_map;
    return graph_rpc_func_map;
}

GraphRpcFuncRegister::GraphRpcFuncRegister(const std::string &name, GraphRpcFunc func)
{
    auto& graph_rpc_func_map = GetGraphRpcFuncMap();

    if (graph_rpc_func_map.find(name) == graph_rpc_func_map.end())
    {
        graph_rpc_func_map[name] = func;
    }
}

Status GraphRpc(Context *ctx, HexagonRpcParam &rpc_param)
{
    Status ret = Status::ERROR;

    std::string graph_name;
    std::string node_name;
    std::string output_dir;
    std::string output_prefix;
    std::unordered_map<std::string, std::vector<std::string>> props;
    Time end_time;
    Time host_base_time;
    Buffer buffer;
    std::string func_name;

    Graph *graph = NULL;
    std::unordered_map<std::string, GraphRpcFunc>& func_map = GetGraphRpcFuncMap();

    ret = rpc_param.Get(graph_name, node_name, output_dir, output_prefix, props, end_time, host_base_time, buffer, func_name);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get failed");
        goto EXIT;
    }

    if (func_map.find(func_name) == func_map.end())
    {
        AURA_ADD_ERROR_STRING(ctx, "invalid name");
        ret = Status::ERROR;
        goto EXIT;
    }

    graph = Create<Graph>(ctx, graph_name, end_time, host_base_time, props);
    if (MI_NULL == graph)
    {
        AURA_ADD_ERROR_STRING(ctx, "null ptr");
        ret = Status::ERROR;
        goto EXIT;
    }

    ret = graph->SetOutputPath(output_dir, output_prefix);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "SetOutputPath failed");
        goto EXIT;
    }

    ret = func_map[func_name](graph, node_name, rpc_param);
    if (Status::ERROR == ret)
    {
        AURA_ADD_ERROR_STRING(ctx, "run failed");
    }

EXIT:
    if (ret != Status::OK)
    {
        if (graph != NULL && graph->GetProfiler() != NULL && graph->GetProfiler()->IsEnablePerf())
        {
            rpc_param.ResetBuffer();
            ret |= rpc_param.Set(graph_rpc_magic_number);
            ret |= graph->GetProfiler()->Serialize(rpc_param);
        }
    }
    else
    {
        if (graph->GetProfiler()->IsEnablePerf())
        {
            if (rpc_param.m_rpc_param.m_size <= buffer.m_size)
            {
                AuraMemCopy(buffer.m_data, rpc_param.m_rpc_param.m_origin, rpc_param.m_rpc_param.m_size);
            }
            else
            {
                buffer.Clear();
            }

            rpc_param.ResetBuffer();
            ret |= graph->GetProfiler()->Serialize(rpc_param);
            if (Status::OK == ret)
            {
                if (buffer.IsValid() && (rpc_param.m_rpc_param.m_capacity - rpc_param.m_rpc_param.m_size >= buffer.m_size))
                {
                    AuraMemCopy(rpc_param.m_rpc_param.m_data, buffer.m_data, buffer.m_size);
                }
                else
                {
                    AURA_ADD_ERROR_STRING(ctx, "invalid buffer or size not satisfied");
                    rpc_param.ResetBuffer();
                    rpc_param.Set((MI_S32)0);
                    ret = Status::ERROR;
                }
            }
        }
    }

    Delete<Graph>(ctx, &graph);
    AURA_RETURN(ctx, ret);
}

AURA_HEXAGON_RPC_FUNC_REGISTER(AURA_ALGOS_GRAPH_PACKAGE_NAME, AURA_ALGOS_GRAPH_MODULE_NAME, GraphRpc);

} // namespace aura