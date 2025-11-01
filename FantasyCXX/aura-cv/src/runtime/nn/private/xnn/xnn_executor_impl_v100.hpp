
#ifndef AURA_RUNTIME_NN_XNN_EXECUTOR_IMPL_V100_HPP__
#define AURA_RUNTIME_NN_XNN_EXECUTOR_IMPL_V100_HPP__

#include "xnn_executor_impl.hpp"
#include "aura/runtime/nn/nn_utils.hpp"

#include "xnn0.5.1/c_api/XnnCommon.h"
#include "xnn0.5.1/c_api/XnnConfig.h"
#include "xnn0.5.1/c_api/XnnPredictor.h"
#include "xnn0.5.1/c_api/XnnTensor.h"

#include <dlfcn.h>
#include <fstream>

#define AURA_NN_XNN_Vx_MAGIC     (0x786E6E64)                  // 'x'(0x78) 'n'(0x6E) 'n'(0x6E) 100(0x64)
#define XnnExecutorImplVx        XnnExecutorImplV100
#define XnnLibrary               XnnLibraryV100
#define XnnUtils                 XnnUtilsV100
#define XnnIOBufferMap           XnnIOBufferMapV100
#define XNN_EXECUTOR_IMPL_V100

#include "xnn_executor_impl_x.inl"

#endif // AURA_RUNTIME_NN_XNN_EXECUTOR_IMPL_V100_HPP__