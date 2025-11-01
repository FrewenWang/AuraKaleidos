list(APPEND lib_srcs
     ${SRC_ROOT}/inference/AbsPredictor.cpp
     ${SRC_ROOT}/inference/InferenceRegistry.cpp
     ${SRC_ROOT}/inference/CustomizePredictor.cpp)

if (BUILD_NCNN)
    list(APPEND lib_srcs ${SRC_ROOT}/inference/NcnnPredictor.cpp)
endif ()

if (USE_PADDLE_LITE)
    list(APPEND lib_srcs ${SRC_ROOT}/inference/PaddleLitePredictor.cpp)
endif ()

if (USE_TF_LITE)
    list(APPEND lib_srcs ${SRC_ROOT}/inference/TfLitePredictor.cpp)
endif ()

if (USE_SNPE)
    list(APPEND lib_srcs ${SRC_ROOT}/inference/SnpePredictor.cpp)
endif ()

if (USE_QNN)
    list(APPEND lib_srcs ${SRC_ROOT}/inference/QnnPredictor.cpp)
endif ()

if (USE_ONNX)
    list(APPEND lib_srcs ${SRC_ROOT}/inference/OnnxPredictor.cpp)
endif ()