#ifndef AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_V2271_HPP__
#define AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_V2271_HPP__

#include "snpe_executor_impl.hpp"
#include "aura/runtime/nn/nn_utils.hpp"

#include "snpe2.27.1/SNPE/SNPE.h"
#include "snpe2.27.1/SNPE/SNPEUtil.h"
#include "snpe2.27.1/DlContainer/DlContainer.h"
#include "snpe2.27.1/DlSystem/RuntimeList.h"
#include "snpe2.27.1/SNPE/SNPEBuilder.h"

#include <dlfcn.h>
#include <fstream>

#define AURA_NN_SNPE_V2_MAGIC         (0x736E08DF)              // 's'(0x73) 'n'(0x6E) 2271(0x08DF)
#define SnpeExecutorImplV2            SnpeExecutorImplV2271
#define SnpeLibrary                   SnpeLibraryV2271
#define SnpeUtils                     SnpeUtilsV2271
#define SnpeUserBufferMap             SnpeUserBufferMapV2271
#define SNPE_EXECUTOR_IMPL_V2271

#include "snpe_executor_impl_2x.inl"

#endif // AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_V2271_HPP__