
#ifndef AURA_ALGOS_CORE_HEXAGON_GRAPH_RPC_FUNC_REGISTER_HPP__
#define AURA_ALGOS_CORE_HEXAGON_GRAPH_RPC_FUNC_REGISTER_HPP__

#include "aura/algos/core/graph.hpp"

#define AURA_HEXAGON_GRAPH_RPC_FUNC_REGISTER(package, module, func)    \
    static aura::GraphRpcFuncRegister g_##func(package + std::string(".") + module, func)

namespace aura
{

using GraphRpcFunc = Status (*)(Graph*, const std::string&, HexagonRpcParam&);

class AURA_EXPORTS GraphRpcFuncRegister
{
public:
    GraphRpcFuncRegister(const std::string &name, GraphRpcFunc func);
};

} // namespace aura

#endif // AURA_ALGOS_CORE_HEXAGON_GRAPH_RPC_FUNC_REGISTER_HPP__