#ifndef AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_V2300_HPP__
#define AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_V2300_HPP__

#include "snpe_executor_impl.hpp"
#include "aura/runtime/nn/nn_utils.hpp"

#include "snpe2.30.0/SNPE/SNPE.h"
#include "snpe2.30.0/SNPE/SNPEUtil.h"
#include "snpe2.30.0/DlContainer/DlContainer.h"
#include "snpe2.30.0/DlSystem/RuntimeList.h"
#include "snpe2.30.0/SNPE/SNPEBuilder.h"

#include <dlfcn.h>
#include <fstream>

#define AURA_NN_SNPE_V2_MAGIC         (0x736E08FC)              // 's'(0x73) 'n'(0x6E) 2300(0x08FC)
#define SnpeExecutorImplV2            SnpeExecutorImplV2300
#define SnpeLibrary                   SnpeLibraryV2300
#define SnpeUtils                     SnpeUtilsV2300
#define SnpeUserBufferMap             SnpeUserBufferMapV2300
#define SNPE_EXECUTOR_IMPL_V2300

#include "snpe_executor_impl_2x.inl"

#endif // AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_V2300_HPP__