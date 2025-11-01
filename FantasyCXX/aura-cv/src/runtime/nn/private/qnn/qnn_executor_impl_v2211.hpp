#ifndef AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2211_HPP__
#define AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2211_HPP__

#include "qnn_executor_impl.hpp"
#include "aura/runtime/nn/nn_utils.hpp"

#include "qnn2.21.1/QnnInterface.h"
#include "qnn2.21.1/System/QnnSystemInterface.h"
#include "qnn2.21.1/HTP/QnnHtpDevice.h"
#include "qnn2.21.1/HTP/QnnHtpContext.h"

#include <dlfcn.h>
#include <fstream>
#include <unordered_set>

#define AURA_QNN_V2_MAGIC           (0x716E08A3)            // 'q'(0x71) 'n'(0x6E) 2211(0x08A3)
#define QnnExecutorImplV2           QnnExecutorImplV2211
#define QnnLibrary                  QnnLibraryV2211
#define QnnUtils                    QnnUtilsV2211
#define QnnTensorMap                QnnTensorMapV2211
#define QNN_EXECUTOR_IMPL_V2211

#include "qnn_executor_impl_2x.inl"

#endif // AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2211_HPP__