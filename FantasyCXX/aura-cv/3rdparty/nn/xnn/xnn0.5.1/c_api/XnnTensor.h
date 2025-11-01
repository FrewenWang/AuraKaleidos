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

#ifndef XNN_XNNTENSOR_H
#define XNN_XNNTENSOR_H

#include "c_api/XnnCommon.h"
#include "c_api/XnnConfig.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    XNN_TENSOR_ERROR_MIN_ERROR             = XNN_MIN_ERROR_TENSOR,
    XNN_TENSOR_ERROR_INVALID_HANDLE        = XNN_MIN_ERROR_TENSOR + 0,
    XNN_TENSOR_ERROR_INVALID_CONFIG        = XNN_MIN_ERROR_TENSOR + 1,
    XNN_TENSOR_ERROR_CREATE_FAIL           = XNN_MIN_ERROR_TENSOR + 2,
    XNN_TENSOR_ERROR_EXTERNAL_MEMORY_NULL  = XNN_MIN_ERROR_TENSOR + 3,
    XNN_TENSOR_ERROR_BUFFER_OWN_DATA_ERROR = XNN_MIN_ERROR_TENSOR + 4,
    XNN_TENSOR_ERROR_SHAPE_SIZE            = XNN_MIN_ERROR_TENSOR + 5,
    XNN_TENSOR_ERROR_CACHE_SYNC            = XNN_MIN_ERROR_TENSOR + 6,
    XNN_TENSOR_NO_ERROR                    = XNN_SUCCESS,

    XNN_TENSOR_MAX_ERROR                   = XNN_MAX_ERROR_TENSOR,
}XNNTensor_Error_t;

XNN_API
XNNTensor_Error_t XNNTensor_create(XNN_TensorHandle_t* tensorHandle,
                                   XnnTargetType target);

XNN_API
XNNTensor_Error_t XNNTensor_setShape(XNN_TensorHandle_t tensorHandle, int64_t* shape, int shapeNum);

XNN_API
XNN_BufferHandle_t XNNTensor_mutableData(XNN_TensorHandle_t tensorHandle, XnnPrecisionType precisionType);

XNN_API
XNN_BufferHandle_t XNNTensor_mutableDataWithSize(XNN_TensorHandle_t tensorHandle, size_t size);

XNN_API
void* XNNTensor_getAddr(XNN_TensorHandle_t tensorHandle);

XNN_API
XnnPrecisionType XNNTensor_getPrecision(XNN_TensorHandle_t tensorHandle);

XNN_API
XNNTensor_Error_t XNNTensor_getShape(XNN_TensorHandle_t tensorHandle, int64_t* shape, int* shapeSize);

XNN_API
XNNTensor_Error_t XNNTensor_shareExternalMemory(XNN_TensorHandle_t tensorHandle, XnnTargetType target, uint32_t size, uint8_t *addr,
                                                int32_t mem_handle);

XNN_API
XNNTensor_Error_t XNNTensor_shareExternalMemoryWithOffset(XNN_TensorHandle_t tensorHandle, XnnTargetType target, uint32_t size, uint8_t *addr,
                                                          int32_t mem_handle, uint32_t offset, uint32_t offsetSize);

XNN_API
XNNTensor_Error_t XNNTensor_cacheStart(XNN_TensorHandle_t tensorHandle);

XNN_API
XNNTensor_Error_t XNNTensor_cacheEnd(XNN_TensorHandle_t tensorHandle);

XNN_API
XNNTensor_Error_t XNNTensor_free(XNN_TensorHandle_t tensorHandle);

XNN_API
XNNTensor_Error_t XNNTensor_unmapShareMemory(XNN_TensorHandle_t tensorHandle, int32_t mem_handle);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // XNN_XNNTENSOR_H