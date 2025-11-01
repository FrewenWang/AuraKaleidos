#ifndef AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_V2261_HPP__
#define AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_V2261_HPP__

#include "snpe_executor_impl.hpp"
#include "aura/runtime/nn/nn_utils.hpp"

#include "snpe2.26.1/SNPE/SNPE.h"
#include "snpe2.26.1/SNPE/SNPEUtil.h"
#include "snpe2.26.1/DlContainer/DlContainer.h"
#include "snpe2.26.1/DlSystem/RuntimeList.h"
#include "snpe2.26.1/SNPE/SNPEBuilder.h"

#include <dlfcn.h>
#include <fstream>

#define AURA_NN_SNPE_V2_MAGIC         (0x736E08D5)              // 's'(0x73) 'n'(0x6E) 2261(0x08D5)
#define SnpeExecutorImplV2            SnpeExecutorImplV2261
#define SnpeLibrary                   SnpeLibraryV2261
#define SnpeUtils                     SnpeUtilsV2261
#define SnpeUserBufferMap             SnpeUserBufferMapV2261
#define SNPE_EXECUTOR_IMPL_V2261

#include "snpe_executor_impl_2x.inl"

#endif // AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_V2261_HPP__