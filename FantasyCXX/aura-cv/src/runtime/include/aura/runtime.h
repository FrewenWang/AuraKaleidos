#ifndef AURA_RUNTIME_H__
#define AURA_RUNTIME_H__

#include "aura/config.h"
#include "aura/runtime/core.h"

#if defined(AURA_BUILD_XTENSA)
#  include "aura/runtime/xtensa.h"
#else
#  include "aura/runtime/context.h"
#  include "aura/runtime/logger.h"
#  include "aura/runtime/thread_object.h"
#  include "aura/runtime/thread_buffer.h"
#  include "aura/runtime/systrace.h"
#  include "aura/runtime/memory.h"
#  include "aura/runtime/array.h"
#  include "aura/runtime/mat.h"
#  include "aura/runtime/cl_mem.h"
#  include "aura/runtime/xtensa_mat.h"
#  include "aura/runtime/worker_pool.h"
#  if defined(AURA_ENABLE_NN)
#    include "aura/runtime/nn.h"
#  endif // AURA_ENABLE_NN
#  if defined(AURA_ENABLE_OPENCL)
#    include "aura/runtime/opencl.h"
#  endif // AURA_ENABLE_OPENCL
#  if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
#    include "aura/runtime/hexagon.h"
#  endif
#  if defined(AURA_ENABLE_XTENSA)
#    include "aura/runtime/xtensa.h"
#  endif
#endif

#endif // AURA_RUNTIME_H__