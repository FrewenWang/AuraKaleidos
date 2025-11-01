#ifndef AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_V2280_HPP__
#define AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_V2280_HPP__

#include "snpe_executor_impl.hpp"
#include "aura/runtime/nn/nn_utils.hpp"

#include "snpe2.28.0/SNPE/SNPE.h"
#include "snpe2.28.0/SNPE/SNPEUtil.h"
#include "snpe2.28.0/DlContainer/DlContainer.h"
#include "snpe2.28.0/DlSystem/RuntimeList.h"
#include "snpe2.28.0/SNPE/SNPEBuilder.h"

#include <dlfcn.h>
#include <fstream>

#define AURA_NN_SNPE_V2_MAGIC         (0x736E08E8)              // 's'(0x73) 'n'(0x6E) 2280(0x08E8)
#define SnpeExecutorImplV2            SnpeExecutorImplV2280
#define SnpeLibrary                   SnpeLibraryV2280
#define SnpeUtils                     SnpeUtilsV2280
#define SnpeUserBufferMap             SnpeUserBufferMapV2280
#define SNPE_EXECUTOR_IMPL_V2280

#include "snpe_executor_impl_2x.inl"

#endif // AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_V2280_HPP__