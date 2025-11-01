#ifndef VISION_INTENAL_BLOB
#define VISION_INTENAL_BLOB

#include "build_config.h"
#include "vision/util/VaAllocator.h"

namespace aura::vision {
namespace model_resource {

#if !USE_EXTERNAL_MODEL
// 用于存储模型数据到 SDK 内部，结构为：tag + len + model_data
// tag: _IOVVASDK_MODEL_RESOURCE，用于定位模型数据的起始地址
// len: 模型长度，预留8字节
// model_data: 模型数据
static constexpr int INTERNAL_MODEL_OFFSET = 32;
static char _g_internal_model_blob[MODEL_LEN + INTERNAL_MODEL_OFFSET] = {"_IOVVASDK_MODEL_RESOURCE"};
#endif

} // namespace model_resource
} // namespace vision

#endif // VISION_INTENAL_BLOB