#ifndef AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_V2211_HPP__
#define AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_V2211_HPP__

#include "snpe_executor_impl.hpp"
#include "aura/runtime/nn/nn_utils.hpp"

#include "snpe2.21.1/SNPE/SNPE.h"
#include "snpe2.21.1/SNPE/SNPEUtil.h"
#include "snpe2.21.1/DlContainer/DlContainer.h"
#include "snpe2.21.1/DlSystem/RuntimeList.h"
#include "snpe2.21.1/SNPE/SNPEBuilder.h"

#include <dlfcn.h>
#include <fstream>

#define AURA_NN_SNPE_V2_MAGIC         (0x736E08A3)              // 's'(0x73) 'n'(0x6E) 2211(0x08A3)
#define SnpeExecutorImplV2            SnpeExecutorImplV2211
#define SnpeLibrary                   SnpeLibraryV2211
#define SnpeUtils                     SnpeUtilsV2211
#define SnpeUserBufferMap             SnpeUserBufferMapV2211
#define SNPE_EXECUTOR_IMPL_V2211

#include "snpe_executor_impl_2x.inl"

#endif // AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_V2211_HPP__