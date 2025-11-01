#ifndef AURA_RUNTIME_CORE_HEXAGON_H__
#define AURA_RUNTIME_CORE_HEXAGON_H__

#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
#  include "aura/runtime/core/hexagon/comm.hpp"
#  if defined(AURA_BUILD_HEXAGON)
#    include "aura/runtime/core/hexagon/device/core.hpp"
#    include "aura/runtime/core/hexagon/device/load.hpp"
#    include "aura/runtime/core/hexagon/device/splat.hpp"
#    include "aura/runtime/core/hexagon/device/traits.hpp"
#    include "aura/runtime/core/hexagon/device/add.hpp"
#    include "aura/runtime/core/hexagon/device/mul.hpp"
#    include "aura/runtime/core/hexagon/device/cmp.hpp"
#    include "aura/runtime/core/hexagon/device/cvt.hpp"
#    include "aura/runtime/core/hexagon/device/sub.hpp"
#    include "aura/runtime/core/hexagon/device/print.hpp"
#    include "aura/runtime/core/hexagon/device/lut.hpp"
#    include "aura/runtime/core/hexagon/device/div.hpp"
#    include "aura/runtime/core/hexagon/device/divn.hpp"
#    include "aura/runtime/core/hexagon/device/memcpy.hpp"
#    include "aura/runtime/core/hexagon/device/align.hpp"
#    include "aura/runtime/core/hexagon/device/minmax.hpp"
#  endif
#endif

#endif // AURA_RUNTIME_CORE_HEXAGON_H__