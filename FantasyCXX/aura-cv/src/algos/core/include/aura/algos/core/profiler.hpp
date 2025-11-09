#ifndef AURA_ALGOS_CORE_PROFILER_HPP__
#define AURA_ALGOS_CORE_PROFILER_HPP__

#include "aura/runtime/array.h"
#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
#  include "aura/runtime/hexagon.h"
#endif // (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))

namespace aura
{

enum class NodeExecType
{
    INITIALIZE = 0,
    RUN,
    DEINITIALIZE,
};

class Node;
class AURA_EXPORTS Profiler
{
public:
    Profiler(Context *ctx);

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

#if defined(AURA_BUILD_HOST)
    Status Save(const std::string &prefix);
#endif // AURA_BUILD_HOST

#if defined(AURA_BUILD_HEXAGON)
    Status Serialize(HexagonRpcParam &rpc_param);
#elif defined(AURA_ENABLE_HEXAGON)
    Status Deserialize(const std::string &node_full_name, HexagonRpcParam &rpc_param);
#endif // AURA_ENABLE_HEXAGON

private:
    class Impl;
    std::shared_ptr<Impl> m_impl;
};

} // namespace aura

#endif // AURA_ALGOS_CORE_PROFILER_HPP__