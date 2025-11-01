/** @brief      : snpe executor impl v2241 header for aura
 *  @file       : snpe_executor_impl_v2241.hpp
 *  @author     : haoxin3@xiaomi.com
 *  @version    : 1.0.0
 *  @date       : November. 22, 2024
 *  @Copyright  : Copyright 2024 Xiaomi Mobile Software Co., Ltd. All Rights reserved.
 */

#ifndef AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_V2241_HPP__
#define AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_V2241_HPP__

#include "snpe_executor_impl.hpp"
#include "aura/runtime/nn/nn_utils.hpp"

#include "snpe2.24.15/SNPE/SNPE.h"
#include "snpe2.24.15/SNPE/SNPEUtil.h"
#include "snpe2.24.15/DlContainer/DlContainer.h"
#include "snpe2.24.15/DlSystem/RuntimeList.h"
#include "snpe2.24.15/SNPE/SNPEBuilder.h"

#include <dlfcn.h>
#include <fstream>

#define AURA_NN_SNPE_V2_MAGIC         (0x736E08C1)              // 's'(0x73) 'n'(0x6E) 2241(0x08C1)
#define SnpeExecutorImplV2            SnpeExecutorImplV2241
#define SnpeLibrary                   SnpeLibraryV2241
#define SnpeUtils                     SnpeUtilsV2241
#define SnpeUserBufferMap             SnpeUserBufferMapV2241
#define SNPE_EXECUTOR_IMPL_V2241

#include "snpe_executor_impl_2x.inl"

#endif // AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_V2241_HPP__