#include "Helpers.h"

#include <onnxruntime_cxx_api.h>

#include <array>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: AliceBaiduFaceDetection <model.onnx> <image>\n";
        return 2;
    }

    try {
        const std::filesystem::path model_path(argv[1]);
        Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "alice-inference");
        Ort::SessionOptions options;
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        Ort::Session session(environment, model_path.c_str(), options);

        if (session.GetInputCount() != 1) {
            throw std::runtime_error("Only models with one input tensor are supported");
        }

        auto input_shape = session.GetInputTypeInfo(0)
                               .GetTensorTypeAndShapeInfo()
                               .GetShape();
        if (input_shape.size() != 4 || input_shape[0] > 1) {
            throw std::runtime_error("Expected a single NCHW image input");
        }
        input_shape[0] = 1;
        const auto input = alice::inference::load_image_nchw(
            argv[2], input_shape[1], input_shape[2], input_shape[3]);

        const auto memory_info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(input.data()),
            input.size(),
            input_shape.data(),
            input_shape.size());

        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name = session.GetInputNameAllocated(0, allocator);
        const std::array<const char*, 1> input_names{input_name.get()};

        std::vector<Ort::AllocatedStringPtr> output_name_storage;
        std::vector<const char*> output_names;
        output_name_storage.reserve(session.GetOutputCount());
        output_names.reserve(session.GetOutputCount());
        for (std::size_t index = 0; index < session.GetOutputCount(); ++index) {
            output_name_storage.emplace_back(session.GetOutputNameAllocated(index, allocator));
            output_names.push_back(output_name_storage.back().get());
        }

        auto outputs = session.Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            &input_tensor,
            input_names.size(),
            output_names.data(),
            output_names.size());

        std::cout << "Inference completed with " << outputs.size() << " output tensor(s).\n";
        for (std::size_t index = 0; index < outputs.size(); ++index) {
            const auto shape = outputs[index].GetTensorTypeAndShapeInfo().GetShape();
            std::cout << "  " << output_names[index] << ": [";
            for (std::size_t dimension = 0; dimension < shape.size(); ++dimension) {
                if (dimension != 0) {
                    std::cout << ", ";
                }
                std::cout << shape[dimension];
            }
            std::cout << "]\n";
        }
        return 0;
    } catch (const Ort::Exception& error) {
        std::cerr << "ONNX Runtime error: " << error.what() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
    }
    return 1;
}
