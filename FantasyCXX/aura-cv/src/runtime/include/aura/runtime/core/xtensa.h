#ifndef AURA_RUNTIME_CORE_XTENSA_H__
#define AURA_RUNTIME_CORE_XTENSA_H__

#if defined(AURA_ENABLE_XTENSA) || defined(AURA_BUILD_XTENSA)
#  include "aura/runtime/core/xtensa/comm.hpp"
#  if defined(AURA_BUILD_XTENSA)
#    include "aura/runtime/core/xtensa/types.h"
#    include "aura/runtime/core/xtensa/isa.h"
#    include "aura/runtime/core/xtensa/maths.hpp"
#  endif
#endif

#endif // AURA_RUNTIME_CORE_XTENSA_H__