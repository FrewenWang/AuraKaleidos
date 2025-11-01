#ifndef AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_V2311_HPP__
#define AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_V2311_HPP__

#include "snpe_executor_impl.hpp"
#include "aura/runtime/nn/nn_utils.hpp"

#include "snpe2.31.1/SNPE/SNPE.h"
#include "snpe2.31.1/SNPE/SNPEUtil.h"
#include "snpe2.31.1/DlContainer/DlContainer.h"
#include "snpe2.31.1/DlSystem/RuntimeList.h"
#include "snpe2.31.1/SNPE/SNPEBuilder.h"

#include <dlfcn.h>
#include <fstream>

#define AURA_NN_SNPE_V2_MAGIC         (0x736E0907)              // 's'(0x73) 'n'(0x6E) 2311(0x0907)
#define SnpeExecutorImplV2            SnpeExecutorImplV2311
#define SnpeLibrary                   SnpeLibraryV2311
#define SnpeUtils                     SnpeUtilsV2311
#define SnpeUserBufferMap             SnpeUserBufferMapV2311
#define SNPE_EXECUTOR_IMPL_V2311

#include "snpe_executor_impl_2x.inl"

#endif // AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_V2311_HPP__