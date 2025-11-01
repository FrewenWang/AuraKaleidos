#ifndef AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2251_HPP__
#define AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2251_HPP__

#include "qnn_executor_impl.hpp"
#include "aura/runtime/nn/nn_utils.hpp"

#include "qnn2.25.1/QnnInterface.h"
#include "qnn2.25.1/System/QnnSystemInterface.h"
#include "qnn2.25.1/HTP/QnnHtpDevice.h"
#include "qnn2.25.1/HTP/QnnHtpContext.h"

#include <dlfcn.h>
#include <fstream>
#include <unordered_set>

#define AURA_QNN_V2_MAGIC           (0x716E08CB)            // 'q'(0x71) 'n'(0x6E) 2251(0x08CB)
#define QnnExecutorImplV2           QnnExecutorImplV2251
#define QnnLibrary                  QnnLibraryV2251
#define QnnUtils                    QnnUtilsV2251
#define QnnTensorMap                QnnTensorMapV2251
#define QNN_EXECUTOR_IMPL_V2251

#include "qnn_executor_impl_2x.inl"

#endif // AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2251_HPP__