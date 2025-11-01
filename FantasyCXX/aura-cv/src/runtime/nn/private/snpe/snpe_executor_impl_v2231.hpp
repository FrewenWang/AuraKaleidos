
#ifndef AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_V2231_HPP__
#define AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_V2231_HPP__

#include "snpe_executor_impl.hpp"
#include "aura/runtime/nn/nn_utils.hpp"

#include "snpe2.23.1/SNPE/SNPE.h"
#include "snpe2.23.1/SNPE/SNPEUtil.h"
#include "snpe2.23.1/DlContainer/DlContainer.h"
#include "snpe2.23.1/DlSystem/RuntimeList.h"
#include "snpe2.23.1/SNPE/SNPEBuilder.h"

#include <dlfcn.h>
#include <fstream>

#define AURA_NN_SNPE_V2_MAGIC         (0x736E08B7)              // 's'(0x73) 'n'(0x6E) 2231(0x08B7)
#define SnpeExecutorImplV2            SnpeExecutorImplV2231
#define SnpeLibrary                   SnpeLibraryV2231
#define SnpeUtils                     SnpeUtilsV2231
#define SnpeUserBufferMap             SnpeUserBufferMapV2231
#define SNPE_EXECUTOR_IMPL_V2231

#include "snpe_executor_impl_2x.inl"

#endif // AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_V2231_HPP__