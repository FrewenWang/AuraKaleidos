#ifndef AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2201_HPP__
#define AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2201_HPP__

#include "qnn_executor_impl.hpp"
#include "aura/runtime/nn/nn_utils.hpp"

#include "qnn2.20.1/QnnInterface.h"
#include "qnn2.20.1/System/QnnSystemInterface.h"
#include "qnn2.20.1/HTP/QnnHtpDevice.h"
#include "qnn2.20.1/HTP/QnnHtpContext.h"

#include <dlfcn.h>
#include <fstream>
#include <unordered_set>

#define AURA_QNN_V2_MAGIC           (0x716E0899)            // 'q'(0x71) 'n'(0x6E) 2201(0x0899)
#define QnnExecutorImplV2           QnnExecutorImplV2201
#define QnnLibrary                  QnnLibraryV2201
#define QnnUtils                    QnnUtilsV2201
#define QnnTensorMap                QnnTensorMapV2201
#define QNN_EXECUTOR_IMPL_V2201

#include "qnn_executor_impl_2x.inl"

#endif // AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_V2201_HPP__