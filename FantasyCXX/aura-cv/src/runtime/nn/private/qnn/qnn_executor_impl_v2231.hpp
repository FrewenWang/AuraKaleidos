#ifndef AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2231_HPP__
#define AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2231_HPP__

#include "qnn_executor_impl.hpp"
#include "aura/runtime/nn/nn_utils.hpp"

#include "qnn2.23.1/QnnInterface.h"
#include "qnn2.23.1/System/QnnSystemInterface.h"
#include "qnn2.23.1/HTP/QnnHtpDevice.h"
#include "qnn2.23.1/HTP/QnnHtpContext.h"

#include <dlfcn.h>
#include <fstream>
#include <unordered_set>

#define AURA_QNN_V2_MAGIC           (0x716E08B7)            // 'q'(0x71) 'n'(0x6E) 2231(0x08B7)
#define QnnExecutorImplV2           QnnExecutorImplV2231
#define QnnLibrary                  QnnLibraryV2231
#define QnnUtils                    QnnUtilsV2231
#define QnnTensorMap                QnnTensorMapV2231
#define QNN_EXECUTOR_IMPL_V2231

#include "qnn_executor_impl_2x.inl"

#endif // AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2231_HPP__