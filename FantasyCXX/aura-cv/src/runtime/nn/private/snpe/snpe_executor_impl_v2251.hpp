
#ifndef AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_V2251_HPP__
#define AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_V2251_HPP__

#include "snpe_executor_impl.hpp"
#include "aura/runtime/nn/nn_utils.hpp"

#include "snpe2.25.1/SNPE/SNPE.h"
#include "snpe2.25.1/SNPE/SNPEUtil.h"
#include "snpe2.25.1/DlContainer/DlContainer.h"
#include "snpe2.25.1/DlSystem/RuntimeList.h"
#include "snpe2.25.1/SNPE/SNPEBuilder.h"

#include <dlfcn.h>
#include <fstream>

#define AURA_NN_SNPE_V2_MAGIC         (0x736E08CB)              // 's'(0x73) 'n'(0x6E) 2251(0x08CB)
#define SnpeExecutorImplV2            SnpeExecutorImplV2251
#define SnpeLibrary                   SnpeLibraryV2251
#define SnpeUtils                     SnpeUtilsV2251
#define SnpeUserBufferMap             SnpeUserBufferMapV2251
#define SNPE_EXECUTOR_IMPL_V2251

#include "snpe_executor_impl_2x.inl"

#endif // AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_V2251_HPP__