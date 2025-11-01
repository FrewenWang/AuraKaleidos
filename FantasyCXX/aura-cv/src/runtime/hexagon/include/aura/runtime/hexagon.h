#ifndef AURA_RUNTIME_HEXAGON_H__
#define AURA_RUNTIME_HEXAGON_H__

#include "aura/config.h"

#include "aura/runtime/hexagon/comm.hpp"
#include "aura/runtime/hexagon/rpc_param.hpp"
#if defined(AURA_BUILD_HOST)
#  include "aura/runtime/hexagon/host/hexagon_engine.hpp"
#else // AURA_BUILD_HEXAGON
#  include "aura/runtime/hexagon/device/hexagon_runtime.hpp"
#endif

#endif // AURA_RUNTIME_HEXAGON_H__