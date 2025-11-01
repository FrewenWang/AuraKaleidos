#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <onnxruntime_cxx_api.h>
// #include <tensorrt_provider_factory.h>
// #include <tensorrt_provider_options.h>
#include "Helpers.cpp"


using namespace cv;
using namespace std;


int main(int, char **) {

    Ort::Env env;
    Ort::Session session(nullptr);
    Ort::SessionOptions sessionOptions{nullptr};

    constexpr int64_t numChannles = 1;
    constexpr int64_t width = 320;
    constexpr int64_t height = 224;
    constexpr int64_t numClasses = 1000;
    constexpr int64_t numInputElements = numChannles * height * width;

    const string imgFile = "/home/frewen/03.ProgramSpace/20.AIStudy/01.WorkSpace/NyxAILearning/AliceInference/baidu_face_detection/cxx/assets/xia30_001.jpg";
    const string labelFile = "/home/frewen/03.ProgramSpace/20.AIStudy/03.Source/onnxruntime_resnet/assets/imagenet_classes.txt";
    auto modelPath = "/home/frewen/03.ProgramSpace/20.AIStudy/03.Source/onnxruntime_resnet/assets/resnet50v2.onnx";
    auto faceModel = "/home/frewen/03.ProgramSpace/20.AIStudy/03.Source/onnxruntime_resnet/assets/FaceDetection20230315V4Main.onnx";
    const bool isGPU = false;
    const bool isTRT = false;

    sessionOptions = Ort::SessionOptions();

    // cuda
    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
    OrtCUDAProviderOptions cudaOption;

    // tensorrt
    auto TrtAvailable = std::find(availableProviders.begin(), availableProviders.end(), "TenosrrtExecutionProvider");
    OrtTensorRTProviderOptions trtOptions;
    trtOptions.device_id = 0;
    trtOptions.trt_fp16_enable = 1;
    trtOptions.trt_int8_enable = 0;
    trtOptions.trt_engine_cache_enable = 1;
    trtOptions.trt_engine_cache_path = "/home/yp/workDir/cmake_demo/cache";


    if (isGPU && (cudaAvailable == availableProviders.end())) {
        std::cout << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
        std::cout << "Inference device: CPU" << std::endl;
    } else if (isGPU && (cudaAvailable != availableProviders.end())) {
        std::cout << "Inference device: GPU" << std::endl;
        sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
    } else if (isTRT) {
        std::cout << "Inference device: Trt" << std::endl;
        sessionOptions.AppendExecutionProvider_TensorRT(trtOptions);
    } else {
        std::cout << "Inference device: CPU" << std::endl;
    }


    //load labels
    vector<string> labels = loadLabels(labelFile);
    if (labels.empty()) {
        cout << "Failed to load labels: " << labelFile << endl;
        return 1;
    }

    //load image
    const vector<float> imageVec = loadFaceImage(imgFile);
    if (imageVec.empty()) {
        cout << "Invalid image format. Must be 224*224 RGB image. " << endl;
        return 1;
    }

    // create session
    cout << "create session. " << endl;
    session = Ort::Session(env, faceModel, sessionOptions);

    // get the number of input
    size_t num_input_nodes = session.GetInputCount();
    std::vector<const char *> input_node_names(num_input_nodes);
    std::vector<int64_t> input_node_dims;

    std::cout << "Number of inputs = " << num_input_nodes << std::endl;

    // define shape
    const array<int64_t, 4> inputShape = {1, numChannles, height, width};
    const array<int64_t, 2> outputShape = {1, numClasses};

    // define array
    array<float, numInputElements> input;
    array<float, numClasses> results;
    // copy image data to input array
    copy(imageVec.begin(), imageVec.end(), input.begin());
    // define Tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    auto inputTensor = Ort::Value::CreateTensor<float>(memory_info, input.data(), input.size(), inputShape.data(),
                                                       inputShape.size());
    auto outputTensor = Ort::Value::CreateTensor<float>(memory_info, results.data(), results.size(), outputShape.data(),
                                                        outputShape.size());



    // define names
    Ort::AllocatorWithDefaultOptions ort_alloc;

    std::vector<const char *> inputNames;
    std::vector<const char *> outputNames;
    auto inputNodeName = session.GetInputNameAllocated(0, ort_alloc);
    inputNames.push_back(inputNodeName.get());

    int outputCount = session.GetOutputCount();
    for (int i = 0; i < outputCount; ++i) {
        // auto outputNodeName = session.GetOutputNameAllocated(i, ort_alloc);
        // outputNames.push_back(outputNodeName.get());
        if (i == 0) {
            outputNames.push_back("conv2d_69.tmp_1");
        } else if (i == 1) {
            outputNames.push_back("conv2d_74.tmp_1");
        } else if (i == 2) {
            outputNames.push_back("conv2d_78.tmp_1");
        }
    }

    std::cout << "Input name: " << inputNames[0] << std::endl;
    std::cout << "Output name0: " << outputNames[0] << std::endl;
    std::cout << "Output name1: " << outputNames[1] << std::endl;
    std::cout << "Output name2: " << outputNames[2] << std::endl;

    // run inference
    cout << "run inference. " << endl;
    try {
        for (size_t i = 0; i < 10; i++) {
            auto start = std::chrono::system_clock::now();
            // 获取输出的tensor结果
            auto outputTensors = session.Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor, 1,
                                             outputNames.data(), 3);
            auto end = std::chrono::system_clock::now();
            std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms"
                      << std::endl;
            // 进行推理后处理解析
            for (int i = 0; i < outputCount; ++i) {

            }
        }
    }
    catch (Ort::Exception &e) {
        cout << e.what() << endl;
        return 1;
    }

    // sort results
    vector<pair<size_t, float>> indexValuePairs;
    for (size_t i = 0; i < results.size(); ++i) {
        indexValuePairs.emplace_back(i, results[i]);
    }
    sort(indexValuePairs.begin(), indexValuePairs.end(),
         [](const auto &lhs, const auto &rhs) { return lhs.second > rhs.second; });


    // show Top5
    for (size_t i = 0; i < 5; ++i) {
        const auto &result = indexValuePairs[i];
        cout << i + 1 << ": " << labels[result.first] << " " << result.second << endl;
    }
}



// rm -r build && mkdir build && cd build && cmake .. && make && ./demo
