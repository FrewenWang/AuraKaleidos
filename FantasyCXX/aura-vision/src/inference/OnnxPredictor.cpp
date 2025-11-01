#include <array>
#include "OnnxPredictor.h"
#include "util/InferenceConverter.hpp"
#include "util/DebugUtil.h"

namespace aura::vision {

static const char *TAG = "OnnxPredictor(5000)";

OnnxPredictor::OnnxPredictor() : session(nullptr), memoryInfo(nullptr) {

}

int OnnxPredictor::init(ModelInfo &model) {
    mPerfTag = "[Predictor]: [onnx] " + std::to_string(model.id);
    modelInfo = model;
    if (modelInfo.blobs.size() != 1) {
        VLOGE(TAG, "Predictor model mem error, one blob needed! error model info:%s", model.version.c_str());
        V_RET(Error::MODEL_INIT_ERR);
    }
    VLOGI(TAG, "Predictor init model: %s", modelInfo.port_desc.ability().c_str());

    auto *modelData = static_cast<const uint8_t *>(model.blobs[0].data);
    auto modelSize = static_cast<const size_t>(model.blobs[0].len);

    // onnxruntime setup onnxruntime的初始化
    env = Ort::Env(ORT_LOGGING_LEVEL_VERBOSE, "OnnxPredictor");

    // 使用1个线程执行op,若想提升速度，增加线程数
    session_options.SetIntraOpNumThreads(16);

    // 如果模型配置的是使用GPU,则初始化GPU相关的配置。注意此处逻辑需要安装CUDA和CUDNN。
    // 注意在系统中安装CUDA和CUDNN。
    if (modelInfo.device == GPU) {
        OrtCUDAProviderOptions provider_options; // C接口
        provider_options.device_id = 0;
        provider_options.arena_extend_strategy = 0;
        provider_options.gpu_mem_limit = 2 * 1024 * 1024 * 1024;
        provider_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
        provider_options.do_copy_in_default_stream = 1;
        session_options.AppendExecutionProvider_CUDA(provider_options);
    } else if (modelInfo.device == CPU) {
        // 如果使用CPU则不进行任何处理
    }

    // 设置图优化的级别
    // ORT_DISABLE_ALL = 0,
    // ORT_ENABLE_BASIC = 1,  # 启用基础优化
    // ORT_ENABLE_EXTENDED = 2,
    // ORT_ENABLE_ALL = 99   # 启用所有可能的优化
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.SetLogSeverityLevel(3);
    // 创建通过ONNX模型路径进行实例化的Session
    // Ort::Session session(env, "model_path", session_options);
    // 通过ONNX模型的原始数据和模型大小
    session = Ort::Session(env, modelData, modelSize, session_options);

    // create MemoryInfo
    memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    inputCount = model.port_desc.input_size();
    outputCount = model.port_desc.output_size();
    // ONNX 模型推理的时候，需要知道输入网络层和输出网络层的名字。通过Ort::Session进行获取
    for (auto i = 0; i < inputCount; ++i) {
        Ort::AllocatedStringPtr inputNodeName = session.GetInputNameAllocated(i, inputAllocator);
        inputNodeNames.push_back((new std::string(inputNodeName.get()))->c_str());

        Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(i);
        auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
        inputDimsList.push_back(inputDims);
    }

    for (auto i = 0; i < outputCount; ++i) {
        Ort::AllocatedStringPtr outputNodeName = session.GetOutputNameAllocated(i, outputAllocator);
        outputNodeNames.push_back((new std::string(outputNodeName.get()))->c_str());

        Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(i);
        auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> outputDims = outputTensorInfo.GetShape();
        outputDimsList.push_back(outputDims);
    }

    inited = true;
    V_RET(Error::OK);
}

int OnnxPredictor::deinit() {
    memoryInfo.release();
    session.release();
    V_RET(Error::OK);
}

int OnnxPredictor::doPredict(TensorArray &inputs, TensorArray &outputs, PerfUtil *perf) {
    outputs.clear();
    VLOGD(TAG, "start predict with model: %s", modelInfo.port_desc.ability().c_str());
    V_CHECK_COND(!inited, Error::MODEL_UN_INITED, "Error: predictor not initialized!");
    V_CHECK_COND(inputs.empty(), Error::MODEL_UN_INITED, "Error: predictor inputs are empty!");
    // 我们现在所有的模型都是单输入。所以直接取得inputs[0]
    auto srcTensor = inputs[0];
    // 工程中默认的VTensor的layout是NHWC。
    if (srcTensor.dLayout != NCHW) {
        srcTensor = srcTensor.changeLayout(NCHW);
    }
    
    DBG_PRINT_ARRAY((float *) srcTensor.data, 20, "OnnxPredictor_doPredict_prepare");
    
    auto inputShape = inputDimsList.at(0);
    auto inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, (float *) srcTensor.data, srcTensor.size(),
                                                       inputShape.data(), inputShape.size());

    // 进行模型推理
    auto outputTensors =
            session.Run(Ort::RunOptions{nullptr}, inputNodeNames.data(), &inputTensor, 1,
                        outputNodeNames.data(),
                        outputCount);

    // 模型输出数据的处理
    for (int i = 0; i < outputCount; ++i) {
        float *floatArr = outputTensors[i].GetTensorMutableData<float>();

        auto outputShape = outputDimsList.at(i);
        // 计算返回结果的w h c
        int w = 1, h = 1, c = 1, n = 1;
        if (2 == outputShape.size()) {
            h = outputShape[0];
            w = outputShape[1];
        } else if (3 == outputShape.size()) {
            c = outputShape[0];
            h = outputShape[1];
            w = outputShape[2];
        } else if (4 == outputShape.size()) {
            n = outputShape[0];
            c = outputShape[1];
            h = outputShape[2];
            w = outputShape[3];
        }
        VTensor t(w, h, c, FP32, NCHW);
        memcpy(reinterpret_cast<char *>(t.data), reinterpret_cast<char *>(floatArr), n * c * h * w * sizeof(float));
        DBG_PRINT_ARRAY((char *) srcTensor.data, 20, "OnnxPredictor_doPredict_post_" + std::to_string(i));
        outputs.push_back(t);
    }

    V_RET(Error::OK);
}

DLayout OnnxPredictor::getSupportedLayout() {
    return NCHW;
}

bool OnnxPredictor::valid() {
    return inited;
}

std::vector<ModelInput> OnnxPredictor::get_input_desc() const {
    std::vector<ModelInput> inputs;
    for (const auto &input: modelInfo.port_desc.input()) {
        inputs.emplace_back(input);
    }
    return inputs;
}


} // namespace vision