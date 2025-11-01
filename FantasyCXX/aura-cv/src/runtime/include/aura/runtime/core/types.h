#ifndef AURA_RUNTIME_CORE_TYPES_H__
#define AURA_RUNTIME_CORE_TYPES_H__

#include "aura/config.h"

#include "aura/runtime/core/types/built-in.hpp"
#include "aura/runtime/core/types/range.hpp"
#include "aura/runtime/core/types/point.hpp"
#include "aura/runtime/core/types/sizes.hpp"
#include "aura/runtime/core/types/rect.hpp"
#include "aura/runtime/core/types/keypoint.hpp"
#include "aura/runtime/core/types/scalar.hpp"
#include "aura/runtime/core/types/sequence.hpp"
#if defined(AURA_BUILD_HOST)
#  include "aura/runtime/core/types/fp16.hpp"
#endif
#endif // AURA_RUNTIME_CORE_TYPES_H__