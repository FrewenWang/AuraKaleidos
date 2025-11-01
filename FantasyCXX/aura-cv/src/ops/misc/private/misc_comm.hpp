/** @brief      : misc common header for aura
 *  @file       : misc_comm.hpp
 *  @author     : lidong11@xiaomi.com
 *  @version    : 1.0.0
 *  @date       : Oct. 18, 2023
 *  @Copyright  : Copyright 2023 Xiaomi Mobile Software Co., Ltd. All Rights reserved.
 */

#ifndef AURA_OPS_MISC_COMM_HPP__
#define AURA_OPS_MISC_COMM_HPP__

#include "aura/config.h"

#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
#  define AURA_OPS_MISC_PACKAGE_NAME              "aura.ops.misc"
#endif

#endif // AURA_OPS_MISC_COMM_HPP__