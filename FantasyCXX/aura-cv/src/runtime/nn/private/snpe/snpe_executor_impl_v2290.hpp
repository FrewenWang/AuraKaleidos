#ifndef AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_V2290_HPP__
#define AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_V2290_HPP__

#include "snpe_executor_impl.hpp"
#include "aura/runtime/nn/nn_utils.hpp"

#include "snpe2.29.0/SNPE/SNPE.h"
#include "snpe2.29.0/SNPE/SNPEUtil.h"
#include "snpe2.29.0/DlContainer/DlContainer.h"
#include "snpe2.29.0/DlSystem/RuntimeList.h"
#include "snpe2.29.0/SNPE/SNPEBuilder.h"

#include <dlfcn.h>
#include <fstream>

#define AURA_NN_SNPE_V2_MAGIC         (0x736E08F2)              // 's'(0x73) 'n'(0x6E) 2290(0x08F2)
#define SnpeExecutorImplV2            SnpeExecutorImplV2290
#define SnpeLibrary                   SnpeLibraryV2290
#define SnpeUtils                     SnpeUtilsV2290
#define SnpeUserBufferMap             SnpeUserBufferMapV2290
#define SNPE_EXECUTOR_IMPL_V2290

#include "snpe_executor_impl_2x.inl"

#endif // AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_V2290_HPP__