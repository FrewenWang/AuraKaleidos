#ifndef AURA_RUNTIME_WORKER_POOL_H__
#define AURA_RUNTIME_WORKER_POOL_H__

#include "aura/config.h"

#if defined(AURA_BUILD_HEXAGON)
#  include "aura/runtime/worker_pool/hexagon/worker_pool.hpp"
#elif defined(AURA_BUILD_HOST) // AURA_BUILD_HOST
#  include "aura/runtime/worker_pool/host/worker_pool.hpp"
#endif // AURA_BUILD_HEXAGON

#endif // AURA_RUNTIME_WORKER_POOL_H__