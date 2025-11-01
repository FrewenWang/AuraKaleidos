#ifndef AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_V2201_HPP__
#define AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_V2201_HPP__

#include "snpe_executor_impl.hpp"
#include "aura/runtime/nn/nn_utils.hpp"

#include "snpe2.20.1/SNPE/SNPE.h"
#include "snpe2.20.1/SNPE/SNPEUtil.h"
#include "snpe2.20.1/DlContainer/DlContainer.h"
#include "snpe2.20.1/DlSystem/RuntimeList.h"
#include "snpe2.20.1/SNPE/SNPEBuilder.h"

#include <dlfcn.h>
#include <fstream>

#define AURA_NN_SNPE_V2_MAGIC         (0x736E0899)              // 's'(0x73) 'n'(0x6E) 2201(0x0899)
#define SnpeExecutorImplV2            SnpeExecutorImplV2201
#define SnpeLibrary                   SnpeLibraryV2201
#define SnpeUtils                     SnpeUtilsV2201
#define SnpeUserBufferMap             SnpeUserBufferMapV2201
#define SNPE_EXECUTOR_IMPL_V2201

#include "snpe_executor_impl_2x.inl"

#endif // AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_V2201_HPP__