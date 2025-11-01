#ifndef AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2133_HPP__
#define AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2133_HPP__

#include "qnn_executor_impl.hpp"
#include "aura/runtime/nn/nn_utils.hpp"

#include "qnn2.13.3/QnnInterface.h"
#include "qnn2.13.3/System/QnnSystemInterface.h"
#include "qnn2.13.3/HTP/QnnHtpDevice.h"

#include <dlfcn.h>
#include <fstream>
#include <unordered_set>

#define AURA_QNN_V2_MAGIC           (0x716E0855)            // 'q'(0x71) 'n'(0x6E) 2133(0x0855)
#define QnnExecutorImplV2           QnnExecutorImplV2133
#define QnnLibrary                  QnnLibraryV2133
#define QnnUtils                    QnnUtilsV2133
#define QnnTensorMap                QnnTensorMapV2133
#define QNN_EXECUTOR_IMPL_V2133

#include "qnn_executor_impl_2x.inl"

#endif // AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2133_HPP__