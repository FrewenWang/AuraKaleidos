#include "aura/algos/core/node.hpp"
#include "profiler_impl.hpp"
#if defined(AURA_ENABLE_OPENCL)
#  include "aura/runtime/cl_mem.h"
#endif // AURA_ENABLE_OPENCL
#if defined(AURA_BUILD_HOST)
#  include "aura/tools/json.h"
#endif // AURA_BUILD_HOST

#include <fstream>

namespace aura
{

Profiler::Impl::Impl(Context *ctx) : m_ctx(ctx), m_enable_perf(DT_FALSE)
{}

DT_BOOL Profiler::Impl::IsEnablePerf() const
{
    return m_enable_perf;
}

DT_VOID Profiler::Impl::Initialize(DT_BOOL enable_perf)
{
    m_enable_perf = enable_perf;
}

Status Profiler::Impl::AddNewNode(Node *node)
{
    if (!m_enable_perf)
    {
        return Status::OK;
    }

    if (DT_NULL == node)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "the node pointer is null");
        return Status::ERROR;
    }

    std::string node_name = node->GetName();
    if (m_node_profiling.find(node_name) == m_node_profiling.end())
    {
        NodeProfiling node_profiling;
        node_profiling.type = node->GetType();
        m_node_profiling[node_name] = node_profiling;
    }
    else
    {
        std::string error_msg = "the node name [" + node_name + "] is not found in the profiling";
        AURA_ADD_ERROR_STRING(m_ctx, error_msg.c_str());
        return Status::ERROR;
    }

    return Status::OK;
}

Status Profiler::Impl::AddNodeProfiling(Node *node, const Time &start, const Time &end, Status result, NodeExecType exec_type)
{
    if (!m_enable_perf)
    {
        return Status::OK;
    }

    if (DT_NULL == node)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "the node pointer is null");
        return Status::ERROR;
    }

    std::string node_name = node->GetName();

    if (m_node_profiling.find(node_name) == m_node_profiling.end())
    {
        std::string error_msg = "the node name [" + node_name + "] is not found in the profiling";
        AURA_ADD_ERROR_STRING(m_ctx, error_msg.c_str());
        return Status::ERROR;
    }

    NodeProfiling &node_profiling = m_node_profiling[node_name];
    if ((NodeExecType::INITIALIZE == exec_type) || (node_profiling.node_exec.empty()))
    {
        node_profiling.node_exec.push_back(NodeExec());
    }

    NodeExec &node_exec = node_profiling.node_exec.back();
    ExecInfo exec_info = {start, end, result, std::hash<std::thread::id>()(std::this_thread::get_id())};

    switch (exec_type)
    {
        case NodeExecType::INITIALIZE:
        {
            node_profiling.op_name = node->GetOp()->GetName();
            node_exec.op_target = node->GetOp()->GetOpTarget();
            node_exec.op_info = node->GetOp()->ToString();
            node_exec.init = exec_info;

            std::lock_guard<std::mutex> lock_guard(m_mutex);
            std::vector<const Array*> inputs = node->GetOp()->GetInputArrays();
            for (size_t i = 0; i < inputs.size(); i++)
            {
                if (inputs[i] != DT_NULL)
                {
                    if (m_array_map.find(const_cast<Array*>(inputs[i])) == m_array_map.end())
                    {
                        std::string array_name = node->GetName() + "_autogen_input_" + std::to_string(i);
                        DT_VOID *origin = inputs[i]->GetBuffer().m_origin;
                        if (m_buffer_map.find(origin) == m_buffer_map.end() && inputs[i]->GetBuffer().IsValid())
                        {
                            std::string buffer_name = array_name + ":buffer";
                            m_buffer_map[origin] = buffer_name;
                            m_external_mem.insert(origin);
                        }
                        m_array_map[const_cast<Array*>(inputs[i])] = array_name;
                        m_array_profiling[array_name].buffer_name = m_buffer_map[origin];
                    }

                    node_exec.inputs.push_back(m_array_map[const_cast<Array*>(inputs[i])]);
                }
            }

            std::vector<const Array*> outputs = node->GetOp()->GetOutputArrays();
            for (size_t i = 0; i < outputs.size(); i++)
            {
                if (outputs[i] != DT_NULL)
                {
                    if (m_array_map.find(outputs[i]) == m_array_map.end())
                    {
                        std::string array_name = node->GetName() + "_autogen_output_" + std::to_string(i);
                        DT_VOID *origin = outputs[i]->GetBuffer().m_origin;
                        if (m_buffer_map.find(origin) == m_buffer_map.end() && outputs[i]->GetBuffer().IsValid())
                        {
                            std::string buffer_name = array_name + ":buffer";
                            m_buffer_map[origin] = buffer_name;
                            m_external_mem.insert(origin);
                        }
                        m_array_map[outputs[i]] = array_name;
                        m_array_profiling[array_name].buffer_name = m_buffer_map[origin];
                    }

                    node_exec.outputs.push_back(m_array_map[outputs[i]]);
                }
            }

            break;
        }

        case NodeExecType::RUN:
        {
            node_exec.op_info = node->GetOp()->ToString();
            node_exec.run.push_back(exec_info);
            UpdateNodeOutputs(node_name, node->GetOp()->GetOutputArrays());
            break;
        }

        case NodeExecType::DEINITIALIZE:
        {
            node_exec.deinit = exec_info;
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "the exec_type is invalid");
            return Status::ERROR;
        }
    }

    return Status::OK;
}

Status Profiler::Impl::UpdateNodeOutputs(const std::string &node_name, const std::vector<const Array*> &outputs)
{
    if (!m_enable_perf)
    {
        return Status::OK;
    }

    if (!outputs.size())
    {
        return Status::OK;
    }

    if (m_node_profiling.find(node_name) == m_node_profiling.end())
    {
        std::string error_msg = "the node name [" + node_name + "] is not found in the node profiling";
        AURA_ADD_ERROR_STRING(m_ctx, error_msg.c_str());
        return Status::ERROR;
    }

    NodeProfiling &node_profiling = m_node_profiling[node_name];

    if (node_profiling.node_exec.empty())
    {
        return Status::OK;
    }

    NodeExec &node_exec = node_profiling.node_exec.back();

    if (node_exec.outputs.size() == outputs.size())
    {
        return Status::OK;
    }

    node_exec.outputs.clear();

    for (const Array *array : outputs)
    {
        if (array != DT_NULL && m_array_map.find(array) != m_array_map.end())
        {
            node_exec.outputs.push_back(m_array_map[array]);
        }
    }

    return Status::OK;
}

Status Profiler::Impl::AddCreateArrayProfiling(const std::string &name, const Array *array, const Time &start, const Time &end, DT_BOOL add_buffer)
{
    if (!m_enable_perf)
    {
        return Status::OK;
    }

    if (DT_NULL == array)
    {
        std::string error_msg = "the array pointer with the name [" + name + "] is null";
        AURA_ADD_ERROR_STRING(m_ctx, error_msg.c_str());
        return Status::ERROR;
    }

    std::string record_name = name;
    std::lock_guard<std::mutex> lock_guard(m_mutex);

    if (m_array_profiling.find(record_name) != m_array_profiling.end())
    {
        if (m_repeat_arrays.find(record_name) == m_repeat_arrays.end())
        {
            m_repeat_arrays[record_name] = 1;
        }

        record_name += "_" + std::to_string(m_repeat_arrays[record_name]++);
    }

    ExecInfo alloc = {start, end, Status::OK, std::hash<std::thread::id>()(std::this_thread::get_id())};
    ArrayProfiling array_profiling =
    {
        array->GetElemType(),
        array->GetArrayType(),
        array->GetSizes(),
        array->GetStrides(),
        array->GetTotalBytes(),
#if defined(AURA_ENABLE_OPENCL)
        CLMemParam(),
#endif // AURA_ENABLE_OPENCL
        std::string(),
        array->GetBuffer().GetOffset(),
        alloc,
        ExecInfo()
    };

    m_array_map[array] = record_name;
    switch (array->GetArrayType())
    {
        case ArrayType::MAT:
        {
            if (m_buffer_map.find(array->GetBuffer().m_origin) == m_buffer_map.end() &&
                array->GetBuffer().IsValid() && add_buffer)
            {
                Status ret = AddCreateBufferProfilingImpl(record_name + ":buffer", array->GetBuffer(), start, end);
                if (ret != Status::OK)
                {
                    AURA_ADD_ERROR_STRING(m_ctx, "AddCreateBufferProfilingImpl failed");
                    return Status::ERROR;
                }
            }
            break;
        }

#if defined(AURA_ENABLE_OPENCL)
        case ArrayType::CL_MEMORY:
        {
            const CLMem *clmem = dynamic_cast<const CLMem*>(array);
            if (DT_NULL == clmem)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "the clmem pointer is null");
                return Status::ERROR;
            }
            array_profiling.cl_param = clmem->GetCLMemParam();
            break;
        }
#endif // AURA_ENABLE_OPENCL

        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "the array type is invalid");
            return Status::ERROR;
        }
    }

    array_profiling.buffer_name = m_buffer_map[array->GetBuffer().m_origin];
    m_array_profiling[record_name] = array_profiling;

    return Status::OK;
}

Status Profiler::Impl::AddExternalMem(const std::string &name, const Buffer &buffer)
{
    if (!m_enable_perf)
    {
        return Status::OK;
    }

    std::lock_guard<std::mutex> lock_guard(m_mutex);
    return AddExternalMemImpl(name, buffer);
}

#if defined(AURA_BUILD_HOST)
static DT_U64 TimeToU64(const Time &time)
{
    std::stringstream os;
    os << time.sec << std::setfill('0') << std::setw(3) << time.ms << std::setw(3) << time.us;
    return std::stoull(os.str());
}

Status Profiler::Impl::Save(const std::string &prefix)
{
    if (!m_enable_perf)
    {
        return Status::OK;
    }

    Status ret = Status::ERROR;

    std::string fname = prefix + ".json";
    FILE *fp = fopen(fname.c_str(), "wb");
    if (DT_NULL == fp)
    {
        std::string info = "open " + fname + " failed";
        AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
        return ret;
    }

    aura_json::ordered_json j;
    for (auto &node_profiling : m_node_profiling)
    {
        NodeProfiling &np = node_profiling.second;
        aura_json::ordered_json j_node;
        j_node["name"] = node_profiling.first;
        j_node["op_name"] = np.op_name;
        j_node["type"] = NodeTypesToString(np.type);
        for (auto &node_exec : np.node_exec)
        {
            aura_json::ordered_json j_node_exec;
            j_node_exec["op_info"] = node_exec.op_info;
            j_node_exec["op_target"] = node_exec.op_target.ToString();
            j_node_exec["input_names"] = node_exec.inputs;
            j_node_exec["output_names"] = node_exec.outputs;
            j_node_exec["init"]["start"] = TimeToU64(node_exec.init.start);
            j_node_exec["init"]["end"] = TimeToU64(node_exec.init.end);
            j_node_exec["init"]["thread_id"] = node_exec.init.thread_id;
            j_node_exec["init"]["status"] = StatusToString(node_exec.init.status);
            for (auto &run : node_exec.run)
            {
                aura_json::ordered_json j_run;
                j_run["start"] = TimeToU64(run.start);
                j_run["end"] = TimeToU64(run.end);
                j_run["thread_id"] = run.thread_id;
                j_run["status"] = StatusToString(run.status);

                j_node_exec["run"].push_back(j_run);
            }
            j_node_exec["deinit"]["start"] = TimeToU64(node_exec.deinit.start);
            j_node_exec["deinit"]["end"] = TimeToU64(node_exec.deinit.end);
            j_node_exec["deinit"]["thread_id"] = node_exec.deinit.thread_id;
            j_node_exec["deinit"]["status"] = StatusToString(node_exec.deinit.status);

            j_node["execute_info"].push_back(j_node_exec);
        }

        j["nodes"].push_back(j_node);
    }

    for (auto &array_profiling : m_array_profiling)
    {
        ArrayProfiling &ap = array_profiling.second;
        aura_json::ordered_json j_array;
        j_array["name"] = array_profiling.first;
        j_array["array_type"] = ArrayTypesToString(ap.array_type);
        j_array["elem_type"] = ElemTypesToString(ap.elem_type);
        j_array["sizes"] = ap.sizes.ToString();
        j_array["strides"] = ap.strides.ToString();
        j_array["total_bytes"] = ap.total_bytes;
#if defined(AURA_ENABLE_OPENCL)
        j_array["cl_param"] = ap.cl_param.ToString();
#endif // AURA_ENABLE_OPENCL
        j_array["buffer_name"] = ap.buffer_name;
        j_array["offset"] = ap.offset;
        j_array["alloc"]["start"] = TimeToU64(ap.alloc.start);
        j_array["alloc"]["end"] = TimeToU64(ap.alloc.end);
        j_array["alloc"]["thread_id"] = ap.alloc.thread_id;
        j_array["alloc"]["status"] = StatusToString(ap.alloc.status);
        j_array["free"]["start"] = TimeToU64(ap.free.start);
        j_array["free"]["end"] = TimeToU64(ap.free.end);
        j_array["free"]["thread_id"] = ap.free.thread_id;
        j_array["free"]["status"] = StatusToString(ap.free.status);

        j["arrays"].push_back(j_array);
    }

    for (auto &buffer_profiling : m_buffer_profiling)
    {
        BufferProfiling &bp = buffer_profiling.second;
        aura_json::ordered_json j_buffer;
        j_buffer["name"] = buffer_profiling.first;
        j_buffer["type"] = MemTypeToString(bp.type);
        j_buffer["capacity"] = bp.capacity;
        j_buffer["property"] = bp.property;
        j_buffer["alloc"]["start"] = TimeToU64(bp.alloc.start);
        j_buffer["alloc"]["end"] = TimeToU64(bp.alloc.end);
        j_buffer["alloc"]["thread_id"] = bp.alloc.thread_id;
        j_buffer["alloc"]["status"] = StatusToString(bp.alloc.status);
        j_buffer["free"]["start"] = TimeToU64(bp.free.start);
        j_buffer["free"]["end"] = TimeToU64(bp.free.end);
        j_buffer["free"]["thread_id"] = bp.free.thread_id;
        j_buffer["free"]["status"] = StatusToString(bp.free.status);

        j["buffers"].push_back(j_buffer);
    }

    m_external_mem.clear();
    m_array_map.clear();
    m_buffer_map.clear();
    m_array_profiling.clear();
    m_buffer_profiling.clear();
    m_repeat_arrays.clear();
    m_repeat_buffers.clear();

    for (auto &node_profiling : m_node_profiling)
    {
        node_profiling.second.node_exec.clear();
    }

    std::string json_str = j.dump(4);

    size_t bytes = fwrite(json_str.c_str(), 1, json_str.size(), fp);
    if (bytes != json_str.size())
    {
        std::string info = "fwrite size(" + std::to_string(bytes) + "," + std::to_string(json_str.size()) + ") not match";
        AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
    }
    else
    {
        ret = Status::OK;
    }

    fclose(fp);

    return ret;
}
#endif // AURA_BUILD_HOST

Status Profiler::Impl::AddDeleteArrayProfiling(const Array *array, const Buffer &buffer, const Time &start, const Time &end)
{
    if (!m_enable_perf)
    {
        return Status::OK;
    }

    if (DT_NULL == array)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "the array pointer is null");
        return Status::ERROR;
    }

    std::lock_guard<std::mutex> lock_guard(m_mutex);
    if (m_array_map.find(array) == m_array_map.end())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "the array is not found in the array map");
        return Status::ERROR;
    }

    if (m_array_profiling.find(m_array_map[array]) == m_array_profiling.end())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "the array is not found in the array profiling");
        return Status::ERROR;
    }

    ArrayProfiling &array_profiling = m_array_profiling[m_array_map[array]];
    array_profiling.free = {start, end, Status::OK, std::hash<std::thread::id>()(std::this_thread::get_id())};

    if (m_external_mem.find(buffer.m_origin) == m_external_mem.end() && buffer.IsValid())
    {
        Buffer buffer_tmp = m_ctx->GetMemPool()->GetBuffer(buffer.m_origin);
        if (!buffer_tmp.IsValid())
        {
            Status ret = AddDeleteBufferProfilingImpl(buffer, start, end);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "AddDeleteBufferProfilingImpl failed");
                return Status::ERROR;
            }
        }
    }

    return Status::OK;
}

Status Profiler::Impl::AddCreateBufferProfilingImpl(const std::string &name, const Buffer &buffer, const Time &start, const Time &end)
{
    if (!buffer.IsValid())
    {
        std::string error_msg = "the buffer with the name [" + name + "] is invalid";
        AURA_ADD_ERROR_STRING(m_ctx, error_msg.c_str());
        return Status::ERROR;
    }

    std::string record_name = name;

    if (m_buffer_profiling.find(record_name) != m_buffer_profiling.end())
    {
        if (m_repeat_buffers.find(record_name) == m_repeat_buffers.end())
        {
            m_repeat_buffers[record_name] = 1;
        }

        record_name += "_" + std::to_string(m_repeat_buffers[record_name]++);
    }

    ExecInfo alloc = {start, end, Status::OK, std::hash<std::thread::id>()(std::this_thread::get_id())};
    BufferProfiling buffer_profiling = {buffer.m_type, buffer.m_capacity, buffer.m_property, alloc, ExecInfo()};

    m_buffer_map[buffer.m_origin] = record_name;
    m_buffer_profiling[record_name] = buffer_profiling;

    return Status::OK;
}

Status Profiler::Impl::AddCreateBufferProfiling(const std::string &name, const Buffer &buffer, const Time &start, const Time &end)
{
    if (!m_enable_perf)
    {
        return Status::OK;
    }

    std::lock_guard<std::mutex> lock_guard(m_mutex);
    return AddCreateBufferProfilingImpl(name, buffer, start, end);
}

Status Profiler::Impl::AddDeleteBufferProfilingImpl(const Buffer &buffer, const Time &start, const Time &end)
{
    if (!buffer.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "the buffer is invalid");
        return Status::ERROR;
    }

    if (m_buffer_map.find(buffer.m_origin) == m_buffer_map.end())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "the buffer origin pointer is not found in the buffer map");
        return Status::ERROR;
    }

    if (m_buffer_profiling.find(m_buffer_map[buffer.m_origin]) == m_buffer_profiling.end())
    {
        std::string error_msg = "the buffer name [" + m_buffer_map[buffer.m_origin] + "] is not found in the buffer profiling";
        AURA_ADD_ERROR_STRING(m_ctx, error_msg.c_str());
        return Status::ERROR;
    }

    BufferProfiling &buffer_profiling = m_buffer_profiling[m_buffer_map[buffer.m_origin]];
    buffer_profiling.free = {start, end, Status::OK, std::hash<std::thread::id>()(std::this_thread::get_id())};

    m_buffer_map.erase(buffer.m_origin);

    return Status::OK;
}

Status Profiler::Impl::AddDeleteBufferProfiling(const Buffer &buffer, const Time &start, const Time &end)
{
    if (!m_enable_perf)
    {
        return Status::OK;
    }

    std::lock_guard<std::mutex> lock_guard(m_mutex);
    return AddDeleteBufferProfilingImpl(buffer, start, end);
}

Status Profiler::Impl::AddExternalMemImpl(const std::string &name, const Buffer &buffer)
{
    if (!buffer.IsValid())
    {
        std::string error_msg = "the buffer with the name [" + name + "] is invalid";
        AURA_ADD_ERROR_STRING(m_ctx, error_msg.c_str());
        return Status::ERROR;
    }

    if (m_buffer_profiling.find(name) != m_buffer_profiling.end())
    {
        std::string error_msg = "the buffer name [" + name + "] is not found in the buffer profiling";
        AURA_ADD_ERROR_STRING(m_ctx, error_msg.c_str());
        return Status::ERROR;
    }

    BufferProfiling buffer_profiling = {buffer.m_type, buffer.m_capacity, buffer.m_property, ExecInfo(), ExecInfo()};

    m_buffer_map[buffer.m_origin] = name;
    m_buffer_profiling[name] = buffer_profiling;
    m_external_mem.insert(buffer.m_origin);

    return Status::OK;
}

#if defined(AURA_BUILD_HEXAGON) || defined(AURA_ENABLE_HEXAGON)
template <typename Tp, typename std::enable_if<std::is_same<Tp, ExecInfo>::value>::type* = DT_NULL>
Status Serialize(Context *ctx, HexagonRpcParam *rpc_param, const Tp &exec_info)
{
    Status ret = rpc_param->Set(exec_info.start, exec_info.end, exec_info.status, exec_info.thread_id);
    AURA_RETURN(ctx, ret);
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, ExecInfo>::value>::type* = DT_NULL>
Status Deserialize(Context *ctx, HexagonRpcParam *rpc_param, Tp &exec_info)
{
    Status ret = rpc_param->Get(exec_info.start, exec_info.end, exec_info.status, exec_info.thread_id);
    AURA_RETURN(ctx, ret);
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, OpTarget>::value>::type* = DT_NULL>
Status Serialize(Context *ctx, HexagonRpcParam *rpc_param, const Tp &target)
{
    Status ret = rpc_param->Set(target.m_type);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Set failed");
        return Status::ERROR;
    }

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            ret = rpc_param->Set(target.m_data.none.enable_mt);
            break;
        }

        case TargetType::OPENCL:
        {
            ret = rpc_param->Set(target.m_data.opencl.profiling);
            break;
        }

        case TargetType::HVX:
        {
            ret = rpc_param->Set(target.m_data.hvx.profiling);
            break;
        }

        default:
        {
            break;
        }
    }
    AURA_RETURN(ctx, ret);
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, OpTarget>::value>::type* = DT_NULL>
Status Deserialize(Context *ctx, HexagonRpcParam *rpc_param, Tp &target)
{
    Status ret = rpc_param->Get(target.m_type);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get failed");
        return Status::ERROR;
    }

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            ret = rpc_param->Get(target.m_data.none.enable_mt);
            break;
        }

        case TargetType::OPENCL:
        {
            ret = rpc_param->Get(target.m_data.opencl.profiling);
            break;
        }

        case TargetType::HVX:
        {
            ret = rpc_param->Get(target.m_data.hvx.profiling);
            break;
        }

        default:
        {
            break;
        }
    }
    AURA_RETURN(ctx, ret);
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, NodeExec>::value>::type* = DT_NULL>
Status Serialize(Context *ctx, HexagonRpcParam *rpc_param, const Tp &node_exec)
{
    Status ret = rpc_param->Set(node_exec.op_target, node_exec.op_info, node_exec.inputs, node_exec.outputs,
                                node_exec.init, node_exec.run, node_exec.deinit);
    AURA_RETURN(ctx, ret);
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, NodeExec>::value>::type* = DT_NULL>
Status Deserialize(Context *ctx, HexagonRpcParam *rpc_param, Tp &node_exec)
{
    Status ret = rpc_param->Get(node_exec.op_target, node_exec.op_info, node_exec.inputs, node_exec.outputs,
                                node_exec.init, node_exec.run, node_exec.deinit);
    AURA_RETURN(ctx, ret);
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, NodeProfiling>::value>::type* = DT_NULL>
Status Serialize(Context *ctx, HexagonRpcParam *rpc_param, const Tp &node_profiling)
{
    Status ret = rpc_param->Set(node_profiling.type, node_profiling.op_name, node_profiling.node_exec);
    AURA_RETURN(ctx, ret);
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, NodeProfiling>::value>::type* = DT_NULL>
Status Deserialize(Context *ctx, HexagonRpcParam *rpc_param, Tp &node_profiling)
{
    Status ret = rpc_param->Get(node_profiling.type, node_profiling.op_name, node_profiling.node_exec);
    AURA_RETURN(ctx, ret);
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, ArrayProfiling>::value>::type* = DT_NULL>
Status Serialize(Context *ctx, HexagonRpcParam *rpc_param, const Tp &array_profiling)
{
    Status ret = rpc_param->Set(array_profiling.elem_type, array_profiling.array_type, array_profiling.sizes,
                                array_profiling.strides, array_profiling.total_bytes, array_profiling.buffer_name,
                                array_profiling.offset, array_profiling.alloc, array_profiling.free);
    AURA_RETURN(ctx, ret);
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, ArrayProfiling>::value>::type* = DT_NULL>
Status Deserialize(Context *ctx, HexagonRpcParam *rpc_param, Tp &array_profiling)
{
    Status ret = rpc_param->Get(array_profiling.elem_type, array_profiling.array_type, array_profiling.sizes,
                                array_profiling.strides, array_profiling.total_bytes, array_profiling.buffer_name,
                                array_profiling.offset, array_profiling.alloc, array_profiling.free);
    AURA_RETURN(ctx, ret);
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, BufferProfiling>::value>::type* = DT_NULL>
Status Serialize(Context *ctx, HexagonRpcParam *rpc_param, const Tp &buffer_profiling)
{
    Status ret = rpc_param->Set(buffer_profiling.type, buffer_profiling.capacity, buffer_profiling.property,
                                buffer_profiling.alloc, buffer_profiling.free);
    AURA_RETURN(ctx, ret);
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, BufferProfiling>::value>::type* = DT_NULL>
Status Deserialize(Context *ctx, HexagonRpcParam *rpc_param, Tp &buffer_profiling)
{
    Status ret = rpc_param->Get(buffer_profiling.type, buffer_profiling.capacity, buffer_profiling.property,
                                buffer_profiling.alloc, buffer_profiling.free);
    AURA_RETURN(ctx, ret);
}
#endif // defined(AURA_BUILD_HEXAGON) || defined(AURA_ENABLE_HEXAGON)

#if defined(AURA_BUILD_HEXAGON)
Status Profiler::Impl::Serialize(HexagonRpcParam &rpc_param)
{
    if (!m_enable_perf)
    {
        return Status::OK;
    }

    Status ret = rpc_param.Set(m_node_profiling, m_array_profiling, m_buffer_profiling);
    if (ret != Status::OK)
    {
        rpc_param.ResetBuffer();
        rpc_param.Set((DT_S32)0);
        AURA_ADD_ERROR_STRING(m_ctx, "Set failed");
    }

    AURA_RETURN(m_ctx, ret);
}
#endif // AURA_BUILD_HEXAGON

#if defined(AURA_ENABLE_HEXAGON)
Status Profiler::Impl::Deserialize(const std::string &node_full_name, HexagonRpcParam &rpc_param)
{
    if (!m_enable_perf)
    {
        return Status::OK;
    }

    std::unordered_map<std::string, NodeProfiling> hexagon_node_profiling_map;
    std::unordered_map<std::string, ArrayProfiling> hexagon_array_profiling_map;
    std::unordered_map<std::string, BufferProfiling> hexagon_buffer_profiling_map;

    Status ret = rpc_param.Get(hexagon_node_profiling_map, hexagon_array_profiling_map, hexagon_buffer_profiling_map);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Get failed");
        return ret;
    }

    NodeProfiling &host_node_profiling = m_node_profiling[node_full_name];
    NodeExec &host_node_exec = host_node_profiling.node_exec.back();
    auto &host_inputs = host_node_exec.inputs;
    auto &host_outputs = host_node_exec.outputs;


    NodeProfiling &hexagon_node_profiling = hexagon_node_profiling_map[node_full_name];
    NodeExec &hexagon_node_exec = hexagon_node_profiling.node_exec.back();
    auto &hexagon_inputs = hexagon_node_exec.inputs;
    auto &hexagon_outputs = hexagon_node_exec.outputs;

    if ((host_inputs.size() != hexagon_inputs.size()) || (host_outputs.size() != hexagon_outputs.size()))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "the size between host inputs/outputs with hexagon inputs/outputs does not match");
        return Status::ERROR;
    }

    std::unordered_map<std::string, std::string> array_name_map;
    std::unordered_map<std::string, std::string> buffer_name_map;

    for (size_t i = 0; i < host_inputs.size(); i++)
    {
        array_name_map[hexagon_inputs[i]] = host_inputs[i];
        buffer_name_map[hexagon_array_profiling_map[hexagon_inputs[i]].buffer_name] = m_array_profiling[host_inputs[i]].buffer_name;
    }

    for (size_t i = 0; i < host_outputs.size(); i++)
    {
        array_name_map[hexagon_outputs[i]] = host_outputs[i];
        buffer_name_map[hexagon_array_profiling_map[hexagon_outputs[i]].buffer_name] = m_array_profiling[host_outputs[i]].buffer_name;
    }

    std::unordered_map<DT_S32, std::string> fds;
    for (auto &buffer_profiling : m_buffer_profiling)
    {
        DT_S32 fd = buffer_profiling.second.property;
        if (fd != 0)
        {
            if (fds.find(fd) != fds.end())
            {
                Time start_last = m_buffer_profiling[fds[fd]].alloc.start;
                Time start_this = buffer_profiling.second.alloc.start;
                if (start_this > start_last)
                {
                    fds[fd] = buffer_profiling.first;
                }
            }
            else
            {
                fds[fd] = buffer_profiling.first;
            }
        }
    }
    for (auto &buffer_profiling : hexagon_buffer_profiling_map)
    {
        if (fds.find(buffer_profiling.second.property) != fds.end() && buffer_name_map.find(buffer_profiling.first) == buffer_name_map.end())
        {
            buffer_name_map[buffer_profiling.first] = fds[buffer_profiling.second.property];
        }
    }

    for (auto &node_profiling : hexagon_node_profiling_map)
    {
        if (node_profiling.first != node_full_name)
        {
            for (auto &node_exec : node_profiling.second.node_exec)
            {
                for (auto &input : node_exec.inputs)
                {
                    if (array_name_map.find(input) != array_name_map.end())
                    {
                        input = array_name_map[input];
                        break;
                    }
                }
                for (auto &output : node_exec.outputs)
                {
                    if (array_name_map.find(output) != array_name_map.end())
                    {
                        output = array_name_map[output];
                        break;
                    }
                }
            }

            m_node_profiling[node_profiling.first] = node_profiling.second;
        }
    }

    for (auto &array_profiling : hexagon_array_profiling_map)
    {
        if (array_name_map.find(array_profiling.first) == array_name_map.end())
        {
            std::string &buffer_name = array_profiling.second.buffer_name;
            if (buffer_name_map.find(buffer_name) != buffer_name_map.end())
            {
                buffer_name = buffer_name_map[buffer_name];
            }
            m_array_profiling[array_profiling.first] = array_profiling.second;
        }
    }

    for (auto &buffer_profiling : hexagon_buffer_profiling_map)
    {
        if (buffer_name_map.find(buffer_profiling.first) == buffer_name_map.end())
        {
            m_buffer_profiling[buffer_profiling.first] = buffer_profiling.second;
        }
    }

    return Status::OK;
}
#endif // AURA_ENABLE_HEXAGON

Profiler::Profiler(Context *ctx) : m_impl(new Profiler::Impl(ctx))
{}

DT_BOOL Profiler::IsEnablePerf() const
{
    if (m_impl)
    {
        return m_impl->IsEnablePerf();
    }
    return DT_FALSE;
}

DT_VOID Profiler::Initialize(DT_BOOL enable_perf)
{
    if (m_impl)
    {
        return m_impl->Initialize(enable_perf);
    }
}

Status Profiler::AddNewNode(Node *node)
{
    if (m_impl)
    {
        return m_impl->AddNewNode(node);
    }
    return Status::ERROR;
}

Status Profiler::AddNodeProfiling(Node *node, const Time &start, const Time &end, Status result, NodeExecType exec_type)
{
    if (m_impl)
    {
        return m_impl->AddNodeProfiling(node, start, end, result, exec_type);
    }
    return Status::ERROR;
}

Status Profiler::UpdateNodeOutputs(const std::string &node_name, const std::vector<const Array*> &outputs)
{
    if (m_impl)
    {
        return m_impl->UpdateNodeOutputs(node_name, outputs);
    }

    return Status::ERROR;
}

Status Profiler::AddCreateArrayProfiling(const std::string &name, const Array *array, const Time &start, const Time &end, DT_BOOL add_buffer)
{
    if (m_impl)
    {
        return m_impl->AddCreateArrayProfiling(name, array, start, end, add_buffer);
    }
    return Status::ERROR;
}

Status Profiler::AddDeleteArrayProfiling(const Array *array, const Buffer &buffer, const Time &start, const Time &end)
{
    if (m_impl)
    {
        return m_impl->AddDeleteArrayProfiling(array, buffer, start, end);
    }
    return Status::ERROR;
}

Status Profiler::AddCreateBufferProfiling(const std::string &name, const Buffer &buffer, const Time &start, const Time &end)
{
    if (m_impl)
    {
        return m_impl->AddCreateBufferProfiling(name, buffer, start, end);
    }
    return Status::ERROR;
}

Status Profiler::AddDeleteBufferProfiling(const Buffer &buffer, const Time &start, const Time &end)
{
    if (m_impl)
    {
        return m_impl->AddDeleteBufferProfiling(buffer, start, end);
    }
    return Status::ERROR;
}

Status Profiler::AddExternalMem(const std::string &name, const Buffer &buffer)
{
    if (m_impl)
    {
        return m_impl->AddExternalMem(name, buffer);
    }
    return Status::ERROR;
}

#if defined(AURA_BUILD_HOST)
Status Profiler::Save(const std::string &prefix)
{
    if (m_impl)
    {
        return m_impl->Save(prefix);
    }
    return Status::ERROR;
}
#endif // AURA_BUILD_HOST

#if defined(AURA_BUILD_HEXAGON)
Status Profiler::Serialize(HexagonRpcParam &rpc_param)
{
    if (m_impl)
    {
        return m_impl->Serialize(rpc_param);
    }
    return Status::ERROR;
}
#elif defined(AURA_ENABLE_HEXAGON)
Status Profiler::Deserialize(const std::string &node_full_name, HexagonRpcParam &rpc_param)
{
    if (m_impl)
    {
        return m_impl->Deserialize(node_full_name, rpc_param);
    }
    return Status::ERROR;
}
#endif // AURA_ENABLE_HEXAGON

} // namespace aura