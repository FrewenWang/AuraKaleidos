#ifndef AURA_RUNTIME_CORE_H__
#define AURA_RUNTIME_CORE_H__

#include "aura/config.h"

#include "aura/runtime/core/defs.hpp"
#include "aura/runtime/core/types.h"
#include "aura/runtime/core/traits.hpp"
#include "aura/runtime/core/xtensa.h"
#if !defined(AURA_BUILD_XTENSA)
#  if defined(AURA_BUILD_HOST)
#    include "aura/runtime/core/limits.hpp"
#  endif // AURA_BUILD_HOST
#  include "aura/runtime/core/status.hpp"
#  include "aura/runtime/core/maths.hpp"
#  include "aura/runtime/core/saturate.hpp"
#  include "aura/runtime/core/log.hpp"
#  if defined(AURA_ENABLE_NEON)
#    include "aura/runtime/core/neon.h"
#  endif
#  include "aura/runtime/core/atomic.hpp"
#  include "aura/runtime/core/time.hpp"
#  include "aura/runtime/core/hexagon.h"
#endif

#endif // AURA_RUNTIME_CORE_H__