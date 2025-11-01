#ifndef AURA_OPS_CORE_H__
#define AURA_OPS_CORE_H__

#include "aura/config.h"

#if defined(AURA_BUILD_XTENSA)
#  include "aura/ops/core/xtensa/core.hpp"
#else
#  include "aura/ops/core/comm.hpp"
#  if defined(AURA_BUILD_HEXAGON)
#    include "aura/ops/core/hexagon/core.hpp"
#  elif defined(AURA_BUILD_HOST)
#    if defined(AURA_ENABLE_OPENCL)
#      include "aura/ops/core/host/cl.hpp"
#    endif
#    if defined(AURA_ENABLE_NEON)
#      include "aura/ops/core/host/neon.hpp"
#    endif
#  endif
#endif

#endif // AURA_OPS_CORE_H__