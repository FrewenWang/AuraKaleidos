
#ifndef AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2261_HPP__
#define AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2261_HPP__

#include "qnn_executor_impl.hpp"
#include "aura/runtime/nn/nn_utils.hpp"

#include "qnn2.26.1/QnnInterface.h"
#include "qnn2.26.1/System/QnnSystemInterface.h"
#include "qnn2.26.1/HTP/QnnHtpDevice.h"
#include "qnn2.26.1/HTP/QnnHtpContext.h"

#include <dlfcn.h>
#include <fstream>
#include <unordered_set>

#define AURA_QNN_V2_MAGIC           (0x716E08D5)            // 'q'(0x71) 'n'(0x6E) 2261(0x08D5)
#define QnnExecutorImplV2           QnnExecutorImplV2261
#define QnnLibrary                  QnnLibraryV2261
#define QnnUtils                    QnnUtilsV2261
#define QnnTensorMap                QnnTensorMapV2261
#define QNN_EXECUTOR_IMPL_V2261

#include "qnn_executor_impl_2x.inl"

#endif // AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2261_HPP__