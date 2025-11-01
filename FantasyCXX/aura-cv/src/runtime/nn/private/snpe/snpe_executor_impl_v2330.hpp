#ifndef AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_V2330_HPP__
#define AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_V2330_HPP__

#include "snpe_executor_impl.hpp"
#include "aura/runtime/nn/nn_utils.hpp"

#include "snpe2.33.0/SNPE/SNPE.h"
#include "snpe2.33.0/SNPE/SNPEUtil.h"
#include "snpe2.33.0/DlContainer/DlContainer.h"
#include "snpe2.33.0/DlSystem/RuntimeList.h"
#include "snpe2.33.0/SNPE/SNPEBuilder.h"

#include <dlfcn.h>
#include <fstream>

#define AURA_NN_SNPE_V2_MAGIC         (0x736E091A)              // 's'(0x73) 'n'(0x6E) 2330(0x091A)
#define SnpeExecutorImplV2            SnpeExecutorImplV2330
#define SnpeLibrary                   SnpeLibraryV2330
#define SnpeUtils                     SnpeUtilsV2330
#define SnpeUserBufferMap             SnpeUserBufferMapV2330
#define SNPE_EXECUTOR_IMPL_V2330

#include "snpe_executor_impl_2x.inl"

#endif // AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_V2330_HPP__