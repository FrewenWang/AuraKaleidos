#ifndef AURA_OPS_FILTER_COMM_HPP__
#define AURA_OPS_FILTER_COMM_HPP__

#include "aura/config.h"
/// 如果是构建HEXAGON  或者 XTENSA 的话，我们是可以启用这个宏定义的

#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON) || \
     defined(AURA_ENABLE_XTENSA) || defined(AURA_BUILD_XTENSA))
#  define AURA_OPS_FILTER_PACKAGE_NAME              "aura.ops.filter"
#endif

#endif // AURA_OPS_FILTER_COMM_HPP__