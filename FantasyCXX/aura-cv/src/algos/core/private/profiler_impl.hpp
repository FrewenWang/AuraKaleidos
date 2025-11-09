
#ifndef AURA_ALGOS_CORE_PROFILER_IMPL_HPP__
#define AURA_ALGOS_CORE_PROFILER_IMPL_HPP__

#include "aura/algos/core/profiler.hpp"
#include "aura/algos/core/graph.hpp"
#include "aura/ops/core.h"

#include <unordered_set>

namespace aura
{

struct ExecInfo
{
    Time start;
    Time end;
    Status status;
    DT_U64 thread_id;
};

struct NodeExec
{
    OpTarget op_target = OpTarget::Default();
    std::string op_info;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    ExecInfo init;
    std::vector<ExecInfo> run;
    ExecInfo deinit;
};

struct NodeProfiling
{
    NodeType type;
    std::string op_name;
    std::vector<NodeExec> node_exec;
};

struct ArrayProfiling
{
    ElemType elem_type;
    ArrayType array_type;
    Sizes3 sizes;
    Sizes strides;
    DT_S64 total_bytes;
#if defined(AURA_ENABLE_OPENCL)
    CLMemParam cl_param;
#endif // AURA_ENABLE_OPENCL
    std::string buffer_name;
    DT_S32 offset;
    ExecInfo alloc;
    ExecInfo free;
};

struct BufferProfiling
{
    DT_S32 type;
    DT_S64 capacity;
    DT_S32 property;
    ExecInfo alloc;
    ExecInfo free;
};

class Profiler::Impl
{
public:
    Impl(Context *ctx);

    DT_BOOL IsEnablePerf() const;
    DT_VOID Initialize(DT_BOOL enable_perf);

    Status AddNewNode(Node *node);
    Status AddNodeProfiling(Node *node, const Time &start, const Time &end, Status result, NodeExecType exec_type);
    Status UpdateNodeOutputs(const std::string &node_name, const std::vector<const Array*> &outputs);

    // array
    Status AddCreateArrayProfiling(const std::string &name, const Array *array, const Time &start, const Time &end, DT_BOOL add_buffer = DT_TRUE);
    Status AddDeleteArrayProfiling(const Array *array, const Buffer &buffer, const Time &start, const Time &end);

    // buffer
    Status AddCreateBufferProfiling(const std::string &name, const Buffer &buffer, const Time &start, const Time &end);
    Status AddDeleteBufferProfiling(const Buffer &buffer, const Time &start, const Time &end);
    Status AddExternalMem(const std::string &name, const Buffer &buffer);

    Status Save(const std::string &prefix);

#if defined(AURA_BUILD_HEXAGON)
    Status Serialize(HexagonRpcParam &rpc_param);
#elif defined(AURA_ENABLE_HEXAGON)
    Status Deserialize(const std::string &node_full_name, HexagonRpcParam &rpc_param);
#endif // AURA_ENABLE_HEXAGON

private:
    Status AddCreateBufferProfilingImpl(const std::string &name, const Buffer &buffer, const Time &start, const Time &end);
    Status AddDeleteBufferProfilingImpl(const Buffer &buffer, const Time &start, const Time &end);
    Status AddExternalMemImpl(const std::string &name, const Buffer &buffer);

    Context *m_ctx;
    DT_BOOL m_enable_perf;
    std::mutex m_mutex;
    std::unordered_set<DT_VOID*> m_external_mem;
    std::unordered_map<const Array*, std::string> m_array_map;
    std::unordered_map<DT_VOID*, std::string> m_buffer_map;
    std::unordered_map<std::string, NodeProfiling> m_node_profiling;
    std::unordered_map<std::string, ArrayProfiling> m_array_profiling;
    std::unordered_map<std::string, BufferProfiling> m_buffer_profiling;
    std::unordered_map<std::string, DT_U32> m_repeat_arrays;
    std::unordered_map<std::string, DT_U32> m_repeat_buffers;
};

} // namespace aura

#endif // AURA_ALGOS_CORE_PROFILER_IMPL_HPP__