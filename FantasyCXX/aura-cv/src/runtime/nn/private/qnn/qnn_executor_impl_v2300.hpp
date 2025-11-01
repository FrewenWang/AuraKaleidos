

#ifndef AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2300_HPP__
#define AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2300_HPP__

#include "qnn_executor_impl.hpp"
#include "aura/runtime/nn/nn_utils.hpp"

#include "qnn2.30.0/QnnInterface.h"
#include "qnn2.30.0/System/QnnSystemInterface.h"
#include "qnn2.30.0/HTP/QnnHtpDevice.h"
#include "qnn2.30.0/HTP/QnnHtpContext.h"

#include <dlfcn.h>
#include <fstream>
#include <unordered_set>

#define AURA_QNN_V2_MAGIC           (0x716E08FC)            // 'q'(0x71) 'n'(0x6E) 2300(0x08FC)
#define QnnExecutorImplV2           QnnExecutorImplV2300
#define QnnLibrary                  QnnLibraryV2300
#define QnnUtils                    QnnUtilsV2300
#define QnnTensorMap                QnnTensorMapV2300
#define QNN_EXECUTOR_IMPL_V2300

#include "qnn_executor_impl_2x.inl"

#endif // AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2300_HPP__