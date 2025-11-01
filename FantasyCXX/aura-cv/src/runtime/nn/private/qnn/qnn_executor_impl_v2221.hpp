#ifndef AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2221_HPP__
#define AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2221_HPP__

#include "qnn_executor_impl.hpp"
#include "aura/runtime/nn/nn_utils.hpp"

#include "qnn2.22.1/QnnInterface.h"
#include "qnn2.22.1/System/QnnSystemInterface.h"
#include "qnn2.22.1/HTP/QnnHtpDevice.h"
#include "qnn2.22.1/HTP/QnnHtpContext.h"

#include <dlfcn.h>
#include <fstream>
#include <unordered_set>

#define AURA_QNN_V2_MAGIC           (0x716E08AD)            // 'q'(0x71) 'n'(0x6E) 2221(0x08AD)
#define QnnExecutorImplV2           QnnExecutorImplV2221
#define QnnLibrary                  QnnLibraryV2221
#define QnnUtils                    QnnUtilsV2221
#define QnnTensorMap                QnnTensorMapV2221
#define QNN_EXECUTOR_IMPL_V2221

#include "qnn_executor_impl_2x.inl"

#endif // AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2221_HPP__