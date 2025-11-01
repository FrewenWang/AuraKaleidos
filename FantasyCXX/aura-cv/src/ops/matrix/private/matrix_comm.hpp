/** @brief     : matrix_comm header for aura
*  @file       : matrix_comm.hpp
*  @author     : zhanghong16@xiaomi.com
*  @version    : 1.0.0
*  @date       : Oct. 30, 2023
*  @Copyright  : Copyright 2023 Xiaomi Mobile Software Co., Ltd. All Rights reserved.
*/

#ifndef AURA_OPS_MATRIX_COMM_HPP__
#define AURA_OPS_MATRIX_COMM_HPP__

#include "aura/config.h"

#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
#  define AURA_OPS_MATRIX_PACKAGE_NAME              "aura.ops.matrix"
#endif

#endif // AURA_OPS_MATRIX_COMM_HPP__