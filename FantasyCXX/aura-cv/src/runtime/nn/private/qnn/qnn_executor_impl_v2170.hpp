#ifndef AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2170_HPP__
#define AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2170_HPP__

#include "qnn_executor_impl.hpp"
#include "aura/runtime/nn/nn_utils.hpp"

#include "qnn2.17.0/QnnInterface.h"
#include "qnn2.17.0/System/QnnSystemInterface.h"
#include "qnn2.17.0/HTP/QnnHtpDevice.h"
#include "qnn2.17.0/HTP/QnnHtpContext.h"

#include <dlfcn.h>
#include <fstream>
#include <unordered_set>

#define AURA_QNN_V2_MAGIC           (0x716E087A)            // 'q'(0x71) 'n'(0x6E) 2170(0x087A)
#define QnnExecutorImplV2           QnnExecutorImplV2170
#define QnnLibrary                  QnnLibraryV2170
#define QnnUtils                    QnnUtilsV2170
#define QnnTensorMap                QnnTensorMapV2170
#define QNN_EXECUTOR_IMPL_V2170

#include "qnn_executor_impl_2x.inl"

#endif // AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2170_HPP__