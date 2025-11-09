
#ifndef AURA_ALGOS_CORE_GRAPH_RPC_IMPL_HPP__
#define AURA_ALGOS_CORE_GRAPH_RPC_IMPL_HPP__

#include "aura/runtime/core.h"

#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))

static constexpr DT_S32 graph_rpc_magic_number = 0x12345678;

#define AURA_ALGOS_GRAPH_PACKAGE_NAME    "aura.algos.graph"
#define AURA_ALGOS_GRAPH_MODULE_NAME     "core"

#endif

#endif // AURA_ALGOS_CORE_GRAPH_RPC_IMPL_HPP__