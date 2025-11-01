#ifndef AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_V2133_HPP__
#define AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_V2133_HPP__

#include "snpe_executor_impl.hpp"
#include "aura/runtime/nn/nn_utils.hpp"

#include "snpe2.13.3/SNPE/SNPE.h"
#include "snpe2.13.3/SNPE/SNPEUtil.h"
#include "snpe2.13.3/DlContainer/DlContainer.h"
#include "snpe2.13.3/DlSystem/RuntimeList.h"
#include "snpe2.13.3/SNPE/SNPEBuilder.h"

#include <dlfcn.h>
#include <fstream>

#define AURA_NN_SNPE_V2_MAGIC         (0x736E0855)              // 's'(0x73) 'n'(0x6E) 2133(0x0855)
#define SnpeExecutorImplV2            SnpeExecutorImplV2133
#define SnpeLibrary                   SnpeLibraryV2133
#define SnpeUtils                     SnpeUtilsV2133
#define SnpeUserBufferMap             SnpeUserBufferMapV2133
#define SNPE_EXECUTOR_IMPL_V2133

#include "snpe_executor_impl_2x.inl"

#endif // AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_V2133_HPP__