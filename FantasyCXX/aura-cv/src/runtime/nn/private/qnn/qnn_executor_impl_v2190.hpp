#ifndef AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2190_HPP__
#define AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2190_HPP__

#include "qnn_executor_impl.hpp"
#include "aura/runtime/nn/nn_utils.hpp"

#include "qnn2.19.0/QnnInterface.h"
#include "qnn2.19.0/System/QnnSystemInterface.h"
#include "qnn2.19.0/HTP/QnnHtpDevice.h"
#include "qnn2.19.0/HTP/QnnHtpContext.h"

#include <dlfcn.h>
#include <fstream>
#include <unordered_set>

#define AURA_QNN_V2_MAGIC           (0x716E088E)            // 'q'(0x71) 'n'(0x6E) 2190(0x088E)
#define QnnExecutorImplV2           QnnExecutorImplV2190
#define QnnLibrary                  QnnLibraryV2190
#define QnnUtils                    QnnUtilsV2190
#define QnnTensorMap                QnnTensorMapV2190
#define QNN_EXECUTOR_IMPL_V2190

#include "qnn_executor_impl_2x.inl"

#endif // AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2190_HPP__