/** @brief      : qnn executor impl v2280 header for aura
 *  @file       : qnn_executor_impl_v2280.hpp
 *  @author     : haoxin3@xiaomi.com
 *  @version    : 1.0.0
 *  @date       : November. 14, 2024
 *  @Copyright  : Copyright 2024 Xiaomi Mobile Software Co., Ltd. All Rights reserved.
 */

#ifndef AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2280_HPP__
#define AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2280_HPP__

#include "qnn_executor_impl.hpp"
#include "aura/runtime/nn/nn_utils.hpp"

#include "qnn2.28.0/QnnInterface.h"
#include "qnn2.28.0/System/QnnSystemInterface.h"
#include "qnn2.28.0/HTP/QnnHtpDevice.h"
#include "qnn2.28.0/HTP/QnnHtpContext.h"

#include <dlfcn.h>
#include <fstream>
#include <unordered_set>

#define AURA_QNN_V2_MAGIC           (0x716E08E8)            // 'q'(0x71) 'n'(0x6E) 2280(0x08E8)
#define QnnExecutorImplV2           QnnExecutorImplV2280
#define QnnLibrary                  QnnLibraryV2280
#define QnnUtils                    QnnUtilsV2280
#define QnnTensorMap                QnnTensorMapV2280
#define QNN_EXECUTOR_IMPL_V2280

#include "qnn_executor_impl_2x.inl"

#endif // AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2280_HPP__