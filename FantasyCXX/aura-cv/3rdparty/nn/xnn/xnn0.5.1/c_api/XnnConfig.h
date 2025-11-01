// Copyright (c) 2023 Xiaomi Technology Co. All Rights Reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef XNN_CONFIG_H
#define XNN_CONFIG_H

#include "c_api/XnnCommon.h"

#ifdef __cplusplus
extern "C" {
#endif

#define XNN_MODEL_PATH_MAX_LEN 256
#define NETWORK_IO_MAX_DIM 6
#define NETWORK_IO_MAX_NAME_LENGTH 64

/**
 * @brief A struct which defines Configure
 */
typedef struct XNN_MobileConfig_t {
    XnnPowerMode         mPowerMode{XNN_POWER_BALANCE};
    XnnTargetType        mTarget{XNN_NPU};
    XnnPriority          mPriority{XNN_PRIORITY_HIGH};
    XnnPrecisionType     mPrecision{kUInt8};
    char                 mModelPath[XNN_MODEL_PATH_MAX_LEN];
    char*                mModelBuf{nullptr};
    uint32_t             mModelSize{0};
    XnnLogLevel          mLogLevel{XNN_CLOSE};
    XnnProfileLevel      mProfileLevel{XNN_PROFILE_LEVEL_DISABLE};
    bool                 mDDrVoteClose{false};
    XnnNetworkCreateType mNetCreateType{XNN_CREATE_NETWORK_FROM_MEMORY};
    XnnNetworkShareMode  mNetShareMode{XNN_SHARED_NONE};
    char*                mModelDmaBufName{nullptr}; //NOT more than 32 bytes
} XNN_MobileConfig;

/**
 * @brief A struct which defines Task Param
 */
typedef struct XNN_TaskPara_t {
    XnnPowerMode     mPowerMode{XNN_POWER_BALANCE};
    XnnTargetType    mTarget{XNN_NPU};
    XnnPriority      mPriority{XNN_PRIORITY_HIGH};
    XnnProfileLevel  mProfileLevel{XNN_PROFILE_LEVEL_DISABLE};
} XNN_TaskPara;

typedef struct XnnIOInfo_t {
    uint32_t          dim_count;
    uint32_t          dim_size[NETWORK_IO_MAX_DIM];
    XnnPrecisionType  data_format;
    XnnQuantFormat    quan_format;
    int32_t           fixed_pos;
    float             scale;
    int32_t           zero_point;
    char              name[NETWORK_IO_MAX_NAME_LENGTH];
    int32_t           size;
} XnnIOInfo;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // XNN_CONFIG_H