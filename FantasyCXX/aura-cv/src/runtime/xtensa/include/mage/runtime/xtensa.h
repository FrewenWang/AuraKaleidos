#ifndef AURA_RUNTIME_XTENSA_H__
#define AURA_RUNTIME_XTENSA_H__

#include "aura/config.h"

#if defined(AURA_ENABLE_XTENSA)
#  include "aura/runtime/array/host/xtensa_mat.hpp"
#  include "aura/runtime/xtensa/host/rpc_param.hpp"
#else
#  include "aura/runtime/xtensa/device/xtensa_runtime.hpp"
#  include "aura/runtime/xtensa/device/xtensa_utils.hpp"
#  include "aura/runtime/xtensa/device/rpc_param.hpp"
#  include "aura/runtime/xtensa/device/xtensa_frame.hpp"
#  include "aura/runtime/xtensa/device/xtensa_tile.hpp"
#endif

#endif // AURA_RUNTIME_XTENSA_H__