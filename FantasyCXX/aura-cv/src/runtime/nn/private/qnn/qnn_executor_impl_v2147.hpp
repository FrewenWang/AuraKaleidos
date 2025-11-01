#ifndef AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2147_HPP__
#define AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2147_HPP__

#include "qnn_executor_impl.hpp"
#include "aura/runtime/nn/nn_utils.hpp"

#include "qnn2.14.7/QnnInterface.h"
#include "qnn2.14.7/System/QnnSystemInterface.h"
#include "qnn2.14.7/HTP/QnnHtpDevice.h"

#include <dlfcn.h>
#include <fstream>
#include <unordered_set>

#define AURA_QNN_V2_MAGIC           (0x716E0863)            // 'q'(0x71) 'n'(0x6E) 2147(0x0863)
#define QnnExecutorImplV2           QnnExecutorImplV2147
#define QnnLibrary                  QnnLibraryV2147
#define QnnUtils                    QnnUtilsV2147
#define QnnTensorMap                QnnTensorMapV2147
#define QNN_EXECUTOR_IMPL_V2147

#include "qnn_executor_impl_2x.inl"

#endif // AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2147_HPP__