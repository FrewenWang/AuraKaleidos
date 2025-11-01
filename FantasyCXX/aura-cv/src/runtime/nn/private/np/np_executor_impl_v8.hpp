#ifndef AURA_RUNTIME_NN_NP_EXECUTOR_IMPL_V8_HPP__
#define AURA_RUNTIME_NN_NP_EXECUTOR_IMPL_V8_HPP__

#include "np_executor_impl.hpp"
#include "aura/runtime/nn/nn_utils.hpp"

#include "np8/RuntimeV2.h"

#include <dlfcn.h>
#include <fstream>

#define AURA_NN_NP_Vx_MAGIC      (0x6E708)                  // 'n'(0x6E) 'p'(0x70) 8(0x8)
#define LIBNEURON_RUNTIME        "libneuron_runtime.8.so"
#define NpExecutorImplVx         NpExecutorImplV8
#define NpLibrary                NpLibraryV8
#define NpUtils                  NpUtilsV8
#define NpIOBufferMap            NpIOBufferMapV8
#define NP_EXECUTOR_IMPL_V8

#include "np_executor_impl_x.inl"

#endif // AURA_RUNTIME_NN_NP_EXECUTOR_IMPL_V8_HPP__