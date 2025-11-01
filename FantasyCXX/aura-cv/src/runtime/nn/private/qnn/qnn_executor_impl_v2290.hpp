
#ifndef AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2290_HPP__
#define AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2290_HPP__

#include "qnn_executor_impl.hpp"
#include "aura/runtime/nn/nn_utils.hpp"

#include "qnn2.29.0/QnnInterface.h"
#include "qnn2.29.0/System/QnnSystemInterface.h"
#include "qnn2.29.0/HTP/QnnHtpDevice.h"
#include "qnn2.29.0/HTP/QnnHtpContext.h"

#include <dlfcn.h>
#include <fstream>
#include <unordered_set>

#define AURA_QNN_V2_MAGIC           (0x716E08F2)            // 'q'(0x71) 'n'(0x6E) 2290(0x08F2)
#define QnnExecutorImplV2           QnnExecutorImplV2290
#define QnnLibrary                  QnnLibraryV2290
#define QnnUtils                    QnnUtilsV2290
#define QnnTensorMap                QnnTensorMapV2290
#define QNN_EXECUTOR_IMPL_V2290

#include "qnn_executor_impl_2x.inl"

#endif // AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2290_HPP__