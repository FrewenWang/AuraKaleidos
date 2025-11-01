#include "aura/algos/core/graph.hpp"
#include "aura/algos/core/node.hpp"
#include "aura/algos/core/algo.hpp"
#include "graph_rpc_impl.hpp"

#include <sstream>
#include <fstream>

#if defined(AURA_BUILD_ANDROID)
#  include "sys/system_properties.h"
#endif // AURA_BUILD_ANDROID

namespace aura
{

Graph::Config::Config(const std::unordered_map<std::string, std::vector<std::string>> &props)
                      : m_props(props)
{}

#if defined(AURA_BUILD_ANDROID)
Graph::Config::Config(const std::string &root_name)
{
    std::string config_key = "xiaomi.aura.config."  + root_name;
    std::unordered_map<std::string, std::string> keys_map =
    {
        { "print", "xiaomi.aura.print." + root_name },
        { "dump", "xiaomi.aura.dump." + root_name },
        { "perf", "xiaomi.aura.perf." + root_name },
    };

    GetPropsFromFile(config_key, keys_map);
    GetPropsFromShell(keys_map);
}
#endif // AURA_BUILD_ANDROID

AURA_VOID Graph::Config::SetOutputPath(const std::string &output_dir, const std::string &output_prefix)
{
    m_output_dir = output_dir + "/";
    m_output_prefix = output_prefix;
}

#if defined AURA_BUILD_ANDROID
AURA_VOID Graph::Config::GetPropsFromFile(const std::string &config_key, const std::unordered_map<std::string, std::string> &keys_map)
{
    MI_CHAR config_path[2048];
    memset(config_path, 0, sizeof(config_path));
    __system_property_get(config_key.c_str(), config_path);

    std::ifstream ifs(config_path);

    if (ifs.fail())
    {
        return;
    }

    std::string line = "";

    while (std::getline(ifs, line))
    {
        std::istringstream iss(line);
        std::vector<std::string> key_prop;
        std::string word = "";

        while (iss >> word)
        {
            key_prop.push_back(word);
        }

        if (key_prop.size() < 2)
        {
            continue;
        }

        for (const auto &key_pair : keys_map)
        {
            if (key_pair.second == key_prop[0])
            {
                m_props[key_pair.first] = ParseProps(key_prop[1]);
            }
        }
    }

    ifs.close();
}

AURA_VOID Graph::Config::GetPropsFromShell(const std::unordered_map<std::string, std::string> &keys_map)
{
    MI_CHAR prop_value[2048];

    for (const auto &key_pair : keys_map)
    {
        if (m_props.find(key_pair.first) == m_props.end())
        {
            memset(prop_value, 0, sizeof(prop_value));
            __system_property_get(key_pair.second.c_str(), prop_value);
            m_props[key_pair.first] = ParseProps(prop_value);
        }
    }
}
#endif // AURA_BUILD_ANDROID

std::vector<std::string> Graph::Config::ParseProps(const std::string &props)
{
    std::vector<std::string> v_props;

    std::stringstream ss(props);
    std::string token;
    while (getline(ss, token, ','))
    {
        v_props.push_back(token);
    }

    return v_props;
}

Node Graph::m_dummy_node;

Graph::Graph(Context *ctx) : m_ctx(ctx), m_is_valid(MI_TRUE)
{}

#if defined(AURA_BUILD_HOST)
Graph::Graph(Context *ctx, const std::unordered_map<std::string, std::vector<std::string>> &props)
             : m_ctx(ctx), m_is_valid(MI_TRUE)
{
    do
    {
#  if defined(AURA_BUILD_ANDROID)
        if (!props.empty())
        {
            m_config = std::make_shared<Config>(props);
            if (MI_NULL == m_config)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "null ptr");
                m_is_valid = MI_FALSE;
                break;
            }
        }
#  else
        m_config = std::make_shared<Config>(props);
        if (MI_NULL == m_config)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "null ptr");
            m_is_valid = MI_FALSE;
            break;
        }
#  endif

        m_profiler = std::make_shared<Profiler>(m_ctx);
        m_timer = std::make_shared<Timer>();
        if (MI_NULL == m_profiler || MI_NULL == m_timer)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "null ptr");
            m_is_valid = MI_FALSE;
        }
    } while (0);
}
#endif // AURA_BUILD_HOST

#if defined(AURA_BUILD_HEXAGON)
Graph::Graph(Context *ctx, const std::string &name, const Time &end_time, const Time &host_base_time,
             const std::unordered_map<std::string, std::vector<std::string>> &props)
             : m_ctx(ctx), m_name(name), m_is_valid(MI_TRUE)
{
    m_config = std::make_shared<Config>(props);
    m_profiler = std::make_shared<Profiler>(m_ctx);
    m_timer = std::make_shared<Timer>(end_time, host_base_time);
    if (MI_NULL == m_config || MI_NULL == m_profiler || MI_NULL == m_timer)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "null ptr");
        m_is_valid = MI_FALSE;
    }
}
#endif // AURA_BUILD_HEXAGON

Graph::~Graph()
{
    for (auto op : m_ops)
    {
        Delete<Op>(m_ctx, &op);
    }

    for (auto node : m_nodes)
    {
        Delete<Node>(m_ctx, &(node.second));
    }
}

AURA_VOID Graph::MakeNodes(Node *node, const std::vector<std::string> &names)
{
    if (!m_is_valid)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid graph");
        return;
    }

    if (MI_NULL == node)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "null ptr");
        m_is_valid = MI_FALSE;
        return;
    }

    for (const auto &name : names)
    {
        if (m_nodes.find(name) != m_nodes.end() || name.find('.') != std::string::npos)
        {
            AURA_ADD_ERROR_STRING(m_ctx, ("invalid name " + name).c_str());
            m_is_valid = MI_FALSE;
            return;
        }

        Node *node_copy = Create<Node>(m_ctx, node->GetOp(), node->GetType(), this, name);
        if (MI_NULL == node_copy)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "null ptr");
            m_is_valid = MI_FALSE;
            return;
        }
        m_nodes[name] = node_copy;
    }
}

Node& Graph::operator[](const std::string &name)
{
    if (!m_is_valid)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid graph");
        return m_dummy_node;
    }

    if (m_nodes.find(name) == m_nodes.end())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid node name");
        return m_dummy_node;
    }
    else
    {
        if (m_nodes[name] != MI_NULL)
        {
            return *(m_nodes[name]);
        }
        else
        {
            AURA_ADD_ERROR_STRING(m_ctx, "null ptr");
            return m_dummy_node;
        }
    }
}

Status Graph::Finalize()
{
    if (!CheckValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "CheckValid failed");
        return Status::ERROR;
    }

    if (m_nodes.size() != 1)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "node size must be 1");
        return Status::ERROR;
    }

#if defined(AURA_BUILD_ANDROID)
    if (MI_NULL == m_config)
    {
        m_config = std::make_shared<Config>(m_nodes.begin()->first);
        if (MI_NULL == m_config)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "null ptr");
            m_is_valid = MI_FALSE;
            return Status::ERROR;
        }
    }
#endif // AURA_BUILD_ANDROID

    if (MI_NULL == m_profiler)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "null ptr");
        m_is_valid = MI_FALSE;
        return Status::ERROR;
    }

    const std::vector<std::string> &perf_props = m_config->m_props["perf"];
    MI_BOOL enable_perf = (perf_props.size() == 1 && perf_props[0] == "1");
    m_profiler->Initialize(enable_perf);

    Node *node = m_nodes.begin()->second;
    const std::vector<std::string> &print_props = m_config->m_props["print"];
    node->SetPrint(print_props);
#if defined(AURA_BUILD_HOST)
    const std::vector<std::string> &dump_props = m_config->m_props["dump"];
    node->SetDump(dump_props);
#endif

    Status ret = m_profiler->AddNewNode(node);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "AddNewNode failed");
        return ret;
    }

    if (node->GetType() == NodeType::ALGO)
    {
        Graph *graph = dynamic_cast<Algo*>(node->GetOp())->GetGraph();
        if (MI_NULL == graph)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "null ptr");
            return Status::ERROR;
        }

        graph->m_config = m_config;
        graph->m_profiler = m_profiler;
        graph->m_timer = m_timer;
        graph->m_name = node->GetName();

        ret = graph->UpdateNodes();
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "UpdateNodes failed");
            return ret;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

#if defined(AURA_BUILD_HOST)
Status Graph::SetTimeout(MI_S32 timeout_ms)
{
    if (!m_is_valid)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid graph");
        return Status::ERROR;
    }

    m_timer->SetTimeout(timeout_ms);
    return Status::OK;
}

Status Graph::SaveProfiling()
{
    if (!m_is_valid)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid graph");
        return Status::ERROR;
    }

    Time now = m_timer->Now();
    std::stringstream os;
    os << now.sec << std::setfill('0') << std::setw(3) << now.ms << std::setw(3) << now.us;

    std::string prefix = m_config->m_output_dir + m_config->m_output_prefix;
    prefix += m_nodes.begin()->second->GetName() + "_" + os.str();

    return m_profiler->Save(prefix);
}
#endif // AURA_BUILD_HOST

Status Graph::SetOutputPath(const std::string &output_dir, const std::string &output_prefix)
{
    if (!m_is_valid)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid graph");
        return Status::ERROR;
    }

    m_config->SetOutputPath(output_dir, output_prefix);
    return Status::OK;
}

#if defined(AURA_BUILD_HOST)
Status Graph::Barrier()
{
    if (!m_is_valid)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid graph");
        return Status::ERROR;
    }

    Status ret = Status::OK;
    MI_U64 thread_id = std::hash<std::thread::id>()(std::this_thread::get_id());
    for (auto &token : m_tokens[thread_id])
    {
        if (token.valid())
        {
            ret |= token.get();
        }
    }
    m_tokens.erase(thread_id);

    return ret;
}
#elif defined(AURA_BUILD_HEXAGON)
Profiler* Graph::GetProfiler()
{
    if (!m_is_valid)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid graph");
        return MI_NULL;
    }

    return m_profiler.get();
}
#endif // AURA_BUILD_HEXAGON

Context* Graph::GetContext()
{
    if (!m_is_valid)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid graph");
        return MI_NULL;
    }

    return m_ctx;
}

#if defined(AURA_ENABLE_HEXAGON)
Status Graph::CallHexagon(const std::string &package, const std::string &module, OpImpl *op_impl,
                          HexagonRpcParam &rpc_param, HexagonProfiling *profiling)
{
    if (!m_is_valid)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid graph");
        return Status::ERROR;
    }

    Status ret = Status::OK;

    HexagonEngine *engine = m_ctx->GetHexagonEngine();
    if (MI_NULL == engine)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "null ptr");
        return Status::ERROR;
    }

    if (MI_NULL == op_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "op_impl is null");
        return Status::ERROR;
    }

    MI_S32 size = 0;
    Buffer buffer;
    if (m_profiler->IsEnablePerf())
    {
        size = rpc_param.m_rpc_param.m_capacity + 128 * 1024;
        buffer = m_ctx->GetMemPool()->GetBuffer(AURA_ALLOC(m_ctx, rpc_param.m_rpc_param.m_capacity));
        if (!buffer.IsValid())
        {
            AURA_ADD_ERROR_STRING(m_ctx, "invalid buffer");
            return Status::ERROR;
        }
    }
    else
    {
        size = rpc_param.m_rpc_param.m_capacity + 4 * 1024;
    }

    std::string graph_name;
    std::string node_name = m_name;

    auto pos = m_name.find_last_of('.');
    if (pos != std::string::npos)
    {
        graph_name = m_name.substr(0, pos);
        node_name = m_name.substr(pos + 1);
    }

    HexagonRpcParam graph_rpc_param(m_ctx, size);
    graph_rpc_param.ResetBuffer();

    for (auto &mem : rpc_param.m_rpc_mem)
    {
        graph_rpc_param.m_rpc_mem.push_back(mem);
    }

    ret = graph_rpc_param.Set(graph_name, node_name, m_config->m_output_dir, m_config->m_output_prefix,
                              m_config->m_props, m_timer->m_end_time, m_timer->Now(), buffer, package + "." + module);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Set failed");
        goto EXIT;
    }

    memcpy(graph_rpc_param.m_rpc_param.m_data, rpc_param.m_rpc_param.m_origin, rpc_param.m_rpc_param.m_size);
    ret = engine->Run(AURA_ALGOS_GRAPH_PACKAGE_NAME, AURA_ALGOS_GRAPH_MODULE_NAME, graph_rpc_param, profiling);

    if (ret != Status::OK)
    {
        if (Status::ERROR == ret)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "HexagonEngine::Run failed");
        }

        if (m_profiler->IsEnablePerf())
        {
            Status ret_status = Status::OK;
            MI_S32 magic_num = 0;

            ret_status = graph_rpc_param.Get(magic_num);
            if (ret_status != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "Get failed");
                ret = ret_status;
                goto EXIT;
            }

            if (magic_num == graph_rpc_magic_number)
            {
                m_profiler->UpdateNodeOutputs(m_name, op_impl->GetOutputArrays());
                ret_status = m_profiler->Deserialize(m_name, graph_rpc_param);

                if (ret_status != Status::OK)
                {
                    AURA_ADD_ERROR_STRING(m_ctx, "Deserialize failed");
                    ret = ret_status;
                    goto EXIT;
                }
            }
        }
    }
    else
    {
        if (m_profiler->IsEnablePerf())
        {
            m_profiler->UpdateNodeOutputs(m_name, op_impl->GetOutputArrays());
            ret |= m_profiler->Deserialize(m_name, graph_rpc_param);

            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "Deserialize failed");
                goto EXIT;
            }
        }

        if (rpc_param.m_rpc_param.m_capacity > graph_rpc_param.m_rpc_param.m_capacity - graph_rpc_param.m_rpc_param.m_size)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "size not satisfied");
            ret = Status::ERROR;
            goto EXIT;
        }
        memcpy(rpc_param.m_rpc_param.m_origin, graph_rpc_param.m_rpc_param.m_data, rpc_param.m_rpc_param.m_capacity);
        rpc_param.ResetBuffer();
    }

EXIT:
    AURA_FREE(m_ctx, buffer.m_origin);
    AURA_RETURN(m_ctx, ret);
}
#endif // AURA_ENABLE_HEXAGON

MI_BOOL Graph::CheckValid()
{
    if (!m_is_valid)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid graph");
        return MI_FALSE;
    }

    for (auto &node : m_nodes)
    {
        if (node.second->GetType() == NodeType::ALGO)
        {
            Graph *graph = dynamic_cast<Algo*>(node.second->GetOp())->GetGraph();
            if (!graph->CheckValid())
            {
                return MI_FALSE;
            }
        }
    }

    return MI_TRUE;
}

Status Graph::UpdateNodes()
{
    Status ret = Status::OK;

    const std::vector<std::string> &print_props = m_config->m_props["print"];
    const std::vector<std::string> &dump_props = m_config->m_props["dump"];

    for (auto &node : m_nodes)
    {
        node.second->SetPrint(print_props);
        node.second->SetDump(dump_props);

        ret = m_profiler->AddNewNode(node.second);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "AddNewNode failed");
            return ret;
        }

        if (node.second->GetType() == NodeType::ALGO)
        {
            Graph *graph = dynamic_cast<Algo*>(node.second->GetOp())->GetGraph();
            if (MI_NULL == graph)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "null ptr");
                return Status::ERROR;
            }

            graph->m_config = m_config;
            graph->m_profiler = m_profiler;
            graph->m_timer = m_timer;
            graph->m_name = node.second->GetName();

            ret = graph->UpdateNodes();
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "UpdateNodes failed");
                return ret;
            }
        }
    }

    AURA_RETURN(m_ctx, ret);
}

Mat* Graph::CreateMat(const std::string &name, ElemType elem_type, const Sizes3 &sizes,
                      MI_S32 mem_type, const Sizes &strides)
{
    return CreateArrayImpl<Mat>(name, elem_type, sizes, mem_type, strides);
}

Mat* Graph::CreateMat(const std::string &name, ElemType elem_type, const Sizes3 &sizes,
                      const Buffer &buffer, const Sizes &strides)
{
    return CreateArrayImpl<Mat>(name, elem_type, sizes, buffer, strides);
}

Mat* Graph::CreateMat(const std::string &name, const Mat *src, const Rect &roi)
{
    if (!m_is_valid)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid graph");
        return MI_NULL;
    }

    if (m_timer && m_timer->IsTimedOut())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "timed out");
        return MI_NULL;
    }

    if (MI_NULL == src)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "null ptr");
        return MI_NULL;
    }

    if (name.find('.') != std::string::npos)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid name");
        return MI_NULL;
    }

    Time start, end;
    if (m_timer)
    {
        start = m_timer->Now();
    }
    Mat *mat = MI_NULL;
    AURA_VOID *buffer = AURA_ALLOC_PARAM(m_ctx, AURA_MEM_HEAP, sizeof(Mat), 0);
    if (buffer != MI_NULL)
    {
        mat = new(buffer) Mat(*src, roi);
    }
    if (MI_NULL == mat)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "null ptr");
        return MI_NULL;
    }
    if (m_timer)
    {
        end = m_timer->Now();
    }

    if (m_profiler)
    {
        std::string full_name = m_name.empty() ? name : m_name + "." + name;
        Status ret = m_profiler->AddCreateArrayProfiling(full_name, mat, start, end);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "AddCreateArrayProfiling failed");
            Delete<Mat>(m_ctx, &mat);
            return MI_NULL;
        }
    }

    return mat;
}

Mat* Graph::CloneMat(const std::string &name, const Mat *src, const Rect &roi, const Sizes &strides)
{
    if (!m_is_valid)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid graph");
        return MI_NULL;
    }

    if (m_timer && m_timer->IsTimedOut())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "timed out");
        return MI_NULL;
    }

    if (MI_NULL == src)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "null ptr");
        return MI_NULL;
    }

    if (name.find('.') != std::string::npos)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid name");
        return MI_NULL;
    }

    Time start, end;
    if (m_timer)
    {
        start = m_timer->Now();
    }
    Mat *mat = MI_NULL;
    AURA_VOID *buffer = AURA_ALLOC_PARAM(m_ctx, AURA_MEM_HEAP, sizeof(Mat), 0);
    if (buffer != MI_NULL)
    {
        mat = new(buffer) Mat();
    }
    if (MI_NULL == mat)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "null ptr");
        return MI_NULL;
    }
    *mat = src->Clone(roi, strides);
    if (m_timer)
    {
        end = m_timer->Now();
    }

    if (m_profiler)
    {
        std::string full_name = m_name.empty() ? name : m_name + "." + name;
        Status ret = m_profiler->AddCreateArrayProfiling(full_name, mat, start, end);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "AddCreateArrayProfiling failed");
            Delete<Mat>(m_ctx, &mat);
            return MI_NULL;
        }
    }

    return mat;
}

#if defined(AURA_ENABLE_OPENCL)
CLMem* Graph::CreateClMem(const std::string &name, const CLMemParam &cl_param, ElemType elem_type,
                          const Sizes3 &sizes, const Sizes &strides)
{
    return CreateArrayImpl<CLMem>(name, cl_param, elem_type, sizes, strides);
}

CLMem* Graph::CreateClMem(const std::string &name, const CLMemParam &cl_param, ElemType elem_type,
                          const Sizes3 &sizes, const Buffer &buffer, const Sizes &strides)
{
    return CreateArrayImpl<CLMem>(name, cl_param, elem_type, sizes, buffer, strides);
}

CLMem* Graph::CreateClMem(const std::string &name, const CLMemParam &cl_param, const Mat *mat)
{
    if (MI_NULL == mat)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "null ptr");
        return MI_NULL;
    }

    return CreateArrayImpl<CLMem>(name, cl_param, mat->GetElemType(), mat->GetSizes(), mat->GetBuffer(), mat->GetStrides());
}
#endif // AURA_ENABLE_OPENCL

Buffer Graph::CreateBuffer(const std::string &name, MI_S64 size, MI_S32 type, MI_S32 align)
{
    if (!m_is_valid)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid graph");
        return Buffer();
    }

    if (m_timer && m_timer->IsTimedOut())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "timed out");
        return Buffer();
    }

    if (name.find('.') != std::string::npos)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid name");
        return Buffer();
    }

    Time start, end;
    if (m_timer)
    {
        start = m_timer->Now();
    }
    Buffer buffer = m_ctx->GetMemPool()->GetBuffer(AURA_ALLOC_PARAM(m_ctx, type, size, align));
    if (m_timer)
    {
        end = m_timer->Now();
    }

    if (m_profiler)
    {
        std::string full_name = m_name.empty() ? name : m_name + "." + name;
        Status ret = m_profiler->AddCreateBufferProfiling(full_name, buffer, start, end);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "AddCreateBufferProfiling failed");
            AURA_FREE(m_ctx, buffer.m_origin);
            buffer.Clear();
        }
    }

    return buffer;
}

AURA_VOID Graph::DeleteBuffer(Buffer &buffer)
{
    if (!m_is_valid)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid graph");
        return;
    }

    Time start, end;
    if (m_timer)
    {
        start = m_timer->Now();
    }
    AURA_FREE(m_ctx, buffer.m_origin);
    if (m_timer)
    {
        end = m_timer->Now();
    }

    if (m_profiler)
    {
        m_profiler->AddDeleteBufferProfiling(buffer, start, end);
    }
    buffer.Clear();
}

Buffer Graph::AddExternalMem(const std::string &name, MI_S32 type, MI_S64 size, AURA_VOID *data, MI_S32 property)
{
    if (!m_is_valid)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid graph");
        return Buffer();
    }

    if (!m_profiler)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "null ptr");
        return Buffer();
    }

    if (m_timer->IsTimedOut())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "timed out");
        return Buffer();
    }

    if (MI_NULL == data)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "null ptr");
        return Buffer();
    }

    if (name.find('.') != std::string::npos)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid name");
        return Buffer();
    }

    Buffer buffer = Buffer(type, size, size, data, data, property);

    std::string full_name = m_name.empty() ? name : m_name + "." + name;
    Status ret = m_profiler->AddExternalMem(full_name, buffer);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "AddExternalMem failed");
        buffer.Clear();
    }

    return buffer;
}

Status Graph::AddExternalArray(const std::string &name, const Array *array)
{
    if (!m_is_valid)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid graph");
        return Status::ERROR;
    }

    if (!m_profiler)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "profiler is null");
        return Status::ERROR;
    }

    if (m_timer->IsTimedOut())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "timed out");
        return Status::ERROR;
    }

    if (MI_NULL == array)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "array is null");
        return Status::ERROR;
    }

    if (name.find('.') != std::string::npos)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid name");
        return Status::ERROR;
    }

    std::string array_name = m_name.empty() ? name : m_name + "." + name;
    Status ret = m_profiler->AddCreateArrayProfiling(array_name, array, m_timer->Now(), m_timer->Now(), MI_FALSE);

    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "AddCreateArrayProfiling failed");
        return Status::ERROR;
    }

    const Buffer buffer = array->GetBuffer();

    if (!buffer.IsValid())
    {
        return Status::ABORT;
    }

    std::string buffer_name = array_name + ":buffer";
    ret = m_profiler->AddExternalMem(buffer_name, buffer);

    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "AddExternalMem failed");
    }

    return Status::OK;
}

} // namespace aura