
#ifndef AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_V2221_HPP__
#define AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_V2221_HPP__

#include "snpe_executor_impl.hpp"
#include "aura/runtime/nn/nn_utils.hpp"

#include "snpe2.22.1/SNPE/SNPE.h"
#include "snpe2.22.1/SNPE/SNPEUtil.h"
#include "snpe2.22.1/DlContainer/DlContainer.h"
#include "snpe2.22.1/DlSystem/RuntimeList.h"
#include "snpe2.22.1/SNPE/SNPEBuilder.h"

#include <dlfcn.h>
#include <fstream>

#define AURA_NN_SNPE_V2_MAGIC         (0x736E08AD)              // 's'(0x73) 'n'(0x6E) 2221(0x08AD)
#define SnpeExecutorImplV2            SnpeExecutorImplV2221
#define SnpeLibrary                   SnpeLibraryV2221
#define SnpeUtils                     SnpeUtilsV2221
#define SnpeUserBufferMap             SnpeUserBufferMapV2221
#define SNPE_EXECUTOR_IMPL_V2221

#include "snpe_executor_impl_2x.inl"

#endif // AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_V2221_HPP__