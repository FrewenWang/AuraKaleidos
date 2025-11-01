#ifndef AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2271_HPP__
#define AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2271_HPP__

#include "qnn_executor_impl.hpp"
#include "aura/runtime/nn/nn_utils.hpp"

#include "qnn2.27.1/QnnInterface.h"
#include "qnn2.27.1/System/QnnSystemInterface.h"
#include "qnn2.27.1/HTP/QnnHtpDevice.h"
#include "qnn2.27.1/HTP/QnnHtpContext.h"

#include <dlfcn.h>
#include <fstream>
#include <unordered_set>

#define AURA_QNN_V2_MAGIC           (0x716E08DF)            // 'q'(0x71) 'n'(0x6E) 2271(0x08DF)
#define QnnExecutorImplV2           QnnExecutorImplV2271
#define QnnLibrary                  QnnLibraryV2271
#define QnnUtils                    QnnUtilsV2271
#define QnnTensorMap                QnnTensorMapV2271
#define QNN_EXECUTOR_IMPL_V2271

#include "qnn_executor_impl_2x.inl"

#endif // AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2271_HPP__