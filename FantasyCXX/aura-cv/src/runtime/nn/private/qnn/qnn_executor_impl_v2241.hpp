
#ifndef AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2241_HPP__
#define AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2241_HPP__

#include "qnn_executor_impl.hpp"
#include "aura/runtime/nn/nn_utils.hpp"

#include "qnn2.24.15/QnnInterface.h"
#include "qnn2.24.15/System/QnnSystemInterface.h"
#include "qnn2.24.15/HTP/QnnHtpDevice.h"
#include "qnn2.24.15/HTP/QnnHtpContext.h"

#include <dlfcn.h>
#include <fstream>
#include <unordered_set>

#define AURA_QNN_V2_MAGIC           (0x716E08C1)            // 'q'(0x71) 'n'(0x6E) 2241(0x08C1)
#define QnnExecutorImplV2           QnnExecutorImplV2241
#define QnnLibrary                  QnnLibraryV2241
#define QnnUtils                    QnnUtilsV2241
#define QnnTensorMap                QnnTensorMapV2241
#define QNN_EXECUTOR_IMPL_V2241

#include "qnn_executor_impl_2x.inl"

#endif // AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2241_HPP__