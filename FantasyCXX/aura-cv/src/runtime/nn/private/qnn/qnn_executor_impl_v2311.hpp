

#ifndef AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2311_HPP__
#define AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2311_HPP__

#include "qnn_executor_impl.hpp"
#include "aura/runtime/nn/nn_utils.hpp"

#include "qnn2.31.1/QnnInterface.h"
#include "qnn2.31.1/System/QnnSystemInterface.h"
#include "qnn2.31.1/HTP/QnnHtpDevice.h"
#include "qnn2.31.1/HTP/QnnHtpContext.h"

#include <dlfcn.h>
#include <fstream>
#include <unordered_set>

#define AURA_QNN_V2_MAGIC           (0x716E0907)            // 'q'(0x71) 'n'(0x6E) 2311(0x0907)
#define QnnExecutorImplV2           QnnExecutorImplV2311
#define QnnLibrary                  QnnLibraryV2311
#define QnnUtils                    QnnUtilsV2311
#define QnnTensorMap                QnnTensorMapV2311
#define QNN_EXECUTOR_IMPL_V2311

#include "qnn_executor_impl_2x.inl"

#endif // AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2311_HPP__