/** @brief      : mnn executor impl v271 header for aura
 *  @file       : mnn_executor_impl_v271.hpp
 *  @author     : xuhaojie3@xiaomi.com
 *  @version    : 1.0.0
 *  @date       : Sep. 13, 2023
 *  @Copyright  : Copyright 2023 Xiaomi Mobile Software Co., Ltd. All Rights reserved.
 */

#ifndef AURA_RUNTIME_NN_MNN_EXECUTOR_IMPL_V271_HPP__
#define AURA_RUNTIME_NN_MNN_EXECUTOR_IMPL_V271_HPP__

#include "mnn_executor_impl.hpp"
#include "aura/runtime/nn/nn_utils.hpp"

#include "mnn2.7.1/include/mnn_wrapper_types.h"

#include <dlfcn.h>
#include <fstream>

#define AURA_MNN_Vx_MAGIC           (0x6D6E6E010F)            // 'm'(0x6D) 'n'(0x6E) 'n'(0x6E) 271(0x010F)
#define LIBMNN_WRAPPER              "libmnn_wrapper.so"
#define MnnExecutorImplVx           MnnExecutorImplV271
#define MnnLibrary                  MnnLibraryV271
#define MnnUtils                    MnnUtilsV271
#define MnnTensorMap                MnnTensorMapV271
#define MNN_EXECUTOR_IMPL_V271

#include "mnn_executor_impl_x.inl"

#endif // AURA_RUNTIME_NN_MNN_EXECUTOR_IMPL_V271_HPP__