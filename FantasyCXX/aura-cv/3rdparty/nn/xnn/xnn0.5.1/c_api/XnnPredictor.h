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

#ifndef XNN_PREDICTOR_H
#define XNN_PREDICTOR_H

#include "c_api/XnnCommon.h"
#include "c_api/XnnConfig.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    XNN_PREDICTOR_ERROR_MIN_ERROR        = XNN_MIN_ERROR_PREDICTOR,
    XNN_PREDICTOR_ERROR_INVALID_HANDLE   = XNN_MIN_ERROR_PREDICTOR + 0,
    XNN_PREDICTOR_ERROR_INVALID_CONFIG   = XNN_MIN_ERROR_PREDICTOR + 1,
    XNN_PREDICTOR_ERROR_PARA_SET_ILLEGAL = XNN_MIN_ERROR_PREDICTOR + 2,
    XNN_PREDICTOR_ERROR_ASYNC_RUN_FAILED = XNN_MIN_ERROR_PREDICTOR + 3,
    XNN_PREDICTOR_NO_ERROR               = XNN_SUCCESS,

    XNN_PREDICTOR_MAX_ERROR              = XNN_MAX_ERROR_PREDICTOR,
}XNNPredictor_Error_t;

XNN_API void XNNPredictor_callBack(XNNPredictor_Error_t error);

XNN_API
XNNPredictor_Error_t XNNPredictor_create(XNN_PredictorHandle_t* predictorHandle,
                                      XNN_MobileConfig* config);

XNN_API
XNN_TensorHandle_t XNNPredictor_getInput(XNN_PredictorHandle_t predictorHandle, int i);

XNN_API
XNN_TensorHandle_t XNNPredictor_getOutput(XNN_PredictorHandle_t predictorHandle, int i);

XNN_API
uint32_t XNNPredictor_getInputCount(XNN_PredictorHandle_t predictorHandle);

XNN_API
uint32_t XNNPredictor_getOutputCount(XNN_PredictorHandle_t predictorHandle);

XNN_API
XNNPredictor_Error_t XNNPredictor_getInputInfo(XNN_PredictorHandle_t predictorHandle, XnnIOInfo* inputInfo);

XNN_API
XNNPredictor_Error_t XNNPredictor_getOutputInfo(XNN_PredictorHandle_t predictorHandle, XnnIOInfo* outputInfo);

XNN_API
XNNPredictor_Error_t XNNPredictor_run(XNN_PredictorHandle_t predictorHandle);

XNN_API
XNNPredictor_Error_t XNNPredictor_asyncRunWithCallBack(XNN_PredictorHandle_t predictorHandle, void(*callback)(XNNPredictor_Error_t));

XNN_API
XNNPredictor_Error_t XNNPredictor_asyncRun(XNN_PredictorHandle_t predictorHandle);

XNN_API
XNNPredictor_Error_t XNNPredictor_synchronize(XNN_PredictorHandle_t predictorHandle);

XNN_API
XNNPredictor_Error_t XNNPredictor_free(XNN_PredictorHandle_t predictorHandle);

XNN_API
XNNPredictor_Error_t XNNSharedPredictor_create(XNN_PredictorHandle_t* sharedPredictorHandle,
                                               XNN_PredictorHandle_t basePredictorHandle,
                                               XNN_MobileConfig* config);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // XNN_PREDICTOR_H