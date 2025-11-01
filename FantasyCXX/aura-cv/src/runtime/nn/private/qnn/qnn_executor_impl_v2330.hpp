
#ifndef AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2330_HPP__
#define AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2330_HPP__

#include "qnn_executor_impl.hpp"
#include "aura/runtime/nn/nn_utils.hpp"

#include "qnn2.33.0/QnnInterface.h"
#include "qnn2.33.0/System/QnnSystemInterface.h"
#include "qnn2.33.0/HTP/QnnHtpDevice.h"
#include "qnn2.33.0/HTP/QnnHtpContext.h"

#include <dlfcn.h>
#include <fstream>
#include <unordered_set>

#define AURA_QNN_V2_MAGIC           (0x716E091A)            // 'q'(0x71) 'n'(0x6E) 2330(0x091A)
#define QnnExecutorImplV2           QnnExecutorImplV2330
#define QnnLibrary                  QnnLibraryV2330
#define QnnUtils                    QnnUtilsV2330
#define QnnTensorMap                QnnTensorMapV2330
#define QNN_EXECUTOR_IMPL_V2330

#include "qnn_executor_impl_2x.inl"

#endif // AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2330_HPP__