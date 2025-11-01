#ifndef AURA_RUNTIME_CONTEXT_H__
#define AURA_RUNTIME_CONTEXT_H__

#include "aura/config.h"

#if defined(AURA_BUILD_HOST)
#  include "aura/runtime/context/host/context.hpp"
#elif defined(AURA_BUILD_HEXAGON)
#  include "aura/runtime/context/hexagon/context.hpp"
#endif // AURA_BUILD_HOST

#endif // AURA_RUNTIME_CONTEXT_H__