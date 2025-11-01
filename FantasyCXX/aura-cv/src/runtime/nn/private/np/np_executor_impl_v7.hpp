#ifndef AURA_RUNTIME_NN_NP_EXECUTOR_IMPL_V7_HPP__
#define AURA_RUNTIME_NN_NP_EXECUTOR_IMPL_V7_HPP__

#include "np_executor_impl.hpp"
#include "aura/runtime/nn/nn_utils.hpp"

#include "np7/RuntimeV2.h"

#include <dlfcn.h>
#include <fstream>

#define AURA_NN_NP_Vx_MAGIC      (0x6E707)                  // 'n'(0x6E) 'p'(0x70) 7(0x7)
#define LIBNEURON_RUNTIME        "libneuron_runtime.7.so"
#define NpExecutorImplVx         NpExecutorImplV7
#define NpLibrary                NpLibraryV7
#define NpUtils                  NpUtilsV7
#define NpIOBufferMap            NpIOBufferMapV7
#define NP_EXECUTOR_IMPL_V7

#include "np_executor_impl_x.inl"

#endif // AURA_RUNTIME_NN_NP_EXECUTOR_IMPL_V7_HPP__