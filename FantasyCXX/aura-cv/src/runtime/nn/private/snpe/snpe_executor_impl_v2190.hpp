#ifndef AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_V2190_HPP__
#define AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_V2190_HPP__

#include "snpe_executor_impl.hpp"
#include "aura/runtime/nn/nn_utils.hpp"

#include "snpe2.19.0/SNPE/SNPE.h"
#include "snpe2.19.0/SNPE/SNPEUtil.h"
#include "snpe2.19.0/DlContainer/DlContainer.h"
#include "snpe2.19.0/DlSystem/RuntimeList.h"
#include "snpe2.19.0/SNPE/SNPEBuilder.h"

#include <dlfcn.h>
#include <fstream>

#define AURA_NN_SNPE_V2_MAGIC         (0x736E088E)              // 's'(0x73) 'n'(0x6E) 2190(0x088E)
#define SnpeExecutorImplV2            SnpeExecutorImplV2190
#define SnpeLibrary                   SnpeLibraryV2190
#define SnpeUtils                     SnpeUtilsV2190
#define SnpeUserBufferMap             SnpeUserBufferMapV2190
#define SNPE_EXECUTOR_IMPL_V2190

#include "snpe_executor_impl_2x.inl"

#endif // AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_V2190_HPP__