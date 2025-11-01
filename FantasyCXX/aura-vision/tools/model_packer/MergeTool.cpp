#include "MergeTool.h"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <utility>

#include "plainbuffer/text_format.h"
#include "util.h"

namespace vision {
namespace tools {

static const char* TAG = "MergeModelTool: ";
const std::unordered_map<std::string, ModelId> MergeModelTool::_model_id_map {
    {"source1_camera_cover",        VISION_TYPE_SOURCE1_CAMERA_COVER},
    {"face_rect",                   VISION_TYPE_FACE_RECT},
    {"face_landmark",               VISION_TYPE_FACE_LANDMARK},
    {"face_feature",                VISION_TYPE_FACE_FEATURE},
    {"face_no_interact_living_rgb", VISION_TYPE_FACE_NO_INTERACT_LIVING_RGB},
    {"face_no_interact_living_ir",  VISION_TYPE_FACE_NO_INTERACT_LIVING_IR},
    {"face_2dto3d",                 VISION_TYPE_FACE_2DTO3D},
    {"face_attribute_rgb",          VISION_TYPE_FACE_ATTRIBUTE_RGB},
    {"face_attribute_ir",           VISION_TYPE_FACE_ATTRIBUTE_IR},
    {"face_call",                   VISION_TYPE_FACE_CALL},
    {"face_dangerous_driving",      VISION_TYPE_FACE_DANGEROUS_DRIVING},
    {"face_drink",                  VISION_TYPE_FACE_DRINK},
    {"face_emotion",                VISION_TYPE_FACE_EMOTION},
    {"face_eye_center",             VISION_TYPE_FACE_EYE_CENTER},
    {"face_mouth_landmark",         VISION_TYPE_FACE_MOUTH_LANDMARK},
    {"face_eye_gaze",               VISION_TYPE_FACE_EYE_GAZE},
    {"face_quality",                VISION_TYPE_FACE_QUALITY},
    {"gesture_rect",                VISION_TYPE_GESTURE_RECT},
    {"gesture_landmark",            VISION_TYPE_GESTURE_LANDMARK},
    {"person_body",                 VISION_TYPE_PERSON_BODY},
    {"person_landmark",             VISION_TYPE_PERSON_LANDMARK},
    {"biology_category",            VISION_TYPE_BIOLOGY_CATEGORY},
    {"face_reconstruct",            VISION_TYPE_FACE_RECONSTRUCT},
    {"head_shoulder",               VISION_TYPE_HEAD_SHOULDER},
    {"body_landmark",               VISION_TYPE_BODY_LANDMARK}
};

const std::unordered_map<std::string, DType> MergeModelTool::_dtype_map {
    {"fp32", FP32},
    {"fp16", FP16},
    {"int8", INT8}
};

const std::unordered_map<std::string, InferType> MergeModelTool::_infer_type_map{
        {"ncnn",        NCNN},
        {"tnn",         TNN},
        {"paddle-lite", PADDLE_LITE},
        {"tf_lite",     TF_LITE},
        {"snpe",        SNPE},
        {"customize",   CUSTOMIZE},
        {"qnn",         QNN},
        {"onnx",        ONNX},
};

const std::unordered_map<std::string, Device> MergeModelTool::_device_map{
        {"cpu", CPU},
        {"gpu", GPU},
        {"dsp", DSP},
        {"htp", HTP}
};

// 每种推理库对应的 [模型文件后缀 + 是否需要加密]
// 全加解密耗时较长，一般只对文本类型的信息做加密处理
const std::tuple<std::string, bool> ncnn_param{".param", true};
const std::tuple<std::string, bool> ncnn_bin{".bin", false};
const std::tuple<std::string, bool> tnn_proto{".tnnproto", true};
const std::tuple<std::string, bool> tnn_model{".tnnmodel", false};
const std::tuple<std::string, bool> paddle_model{".nb", false};
const std::tuple<std::string, bool> snpe_model{".dlc", false};
const std::tuple<std::string, bool> tflite_model{".tflite", false};
const std::tuple<std::string, bool> customize_model{".txt", false};
const std::tuple<std::string, bool> qnn_model{".bin", false};
const std::tuple<std::string, bool> onnx_model{".onnx", false};

const std::unordered_map<std::string, std::vector<std::tuple<std::string, bool>>> MergeModelTool::_model_extensions{
        {"ncnn",        {ncnn_param, ncnn_bin}},
        {"tnn",         {tnn_proto,  tnn_model}},
        {"paddle-lite", {paddle_model}},
        {"tf_lite",     {tflite_model}},
        {"snpe",        {snpe_model}},
        {"customize",   {customize_model}},
        {"qnn",         {qnn_model}},
        {"onnx",        {onnx_model}}
};

MergeModelTool::MergeModelTool(std::string cfg_path, std::string src_path,
                               std::string dest_path, std::string model_arch)
        : _cfg_path(std::move(cfg_path)),
          _src_path(std::move(src_path)),
          _dest_path(std::move(dest_path)),
          _model_arch(std::move(model_arch)) {
    _models.clear();
    _packer = std::make_unique<ModelPacker>();
    remove_end_slash(_src_path);
    remove_end_slash(_dest_path);
}

MergeModelTool::~MergeModelTool() {
    _models.clear();
}

void MergeModelTool::remove_end_slash(std::string& str) {
    if (!str.empty() && str[str.size() - 1] == '/') {
        str = str.substr(0, str.size() - 1);
    }
}

void MergeModelTool::merge_model(ModelEncryptType encrypt) {
    if (!parse_config()) {
        return;
    }

    if (_config.model_size() <= 0) {
        std::cerr << TAG << "No model FOUND in the config" << std::endl;
        return;
    }

    _packer->set_encrpt_type(encrypt);

    // 准备模型合并信息
    for (int i = 0; i < _config.model_size(); ++i) {
        const auto& model = _config.model(i);

        if (!model.enable()) {
            continue;
        }

        auto model_id = str_to_id(model.ability());
        if (model_id == VISION_TYPE_UNKNOWN) {
            std::cerr << TAG << "Ignore unknown type: " << model.ability() << std::endl;
            continue;
        }
        // TODO 需要优化关于模型的拷贝、下载、合并的逻辑。综合考虑model_arch的设计
        auto base_path = _src_path + "/" + model.ability() + "/" + model.version() + "/" + model.infer_type();
        if (model.infer_type() == "paddle-lite") {
            base_path = _src_path + "/" + model.ability() + "/" + model.version() + "/" + model.infer_type() + "/" +
                        _model_arch;
        }
        ModelPackInfo mp;
        mp.id = model_id;
        mp.dtype = str_to_dtype(model.dtype());
        mp.infer_type = str_to_infer_type(model.infer_type());
        mp.device = str_to_device(model.device());
        mp.version = model.version();
        mp.name = model.ability();
        mp.enable_omp = model.enable_omp();
        mp.version_code = model.versionCode();
        mp.extends = model.extends();
        mp.modelArch = _model_arch;    //设置模型支持的系统arch架构

        // 特定推理框架的模型文件格式
        if (_model_extensions.find(model.infer_type()) == _model_extensions.end()) {
            std::cerr << TAG << "Model infer_type: " << model.infer_type() << " does not registered file extensions, ignore" << std::endl;
            continue;
        }

        bool file_valid = true;
        for (const auto& ext : _model_extensions.at(model.infer_type())) {
            std::string file;
            if (mp.dtype == INT8) {
                file = base_path + "/" + model.version() + "_int8" + std::get<0>(ext);
            } else {
                file = base_path + "/" + model.version() + std::get<0>(ext);
            }

            mp.files.emplace_back(file, std::get<1>(ext));
            if (!Util::exists_file(file)) {
                std::cerr << "[" << TAG << "] model file: " << file.c_str() << " CANNOT be found! ignore this model"
                          << std::endl;
                file_valid = false;
                break;
            }
        }

        if (file_valid) {
            _models.emplace_back(mp);
        }
    }

    // 执行模型合并
    auto merged_name = _dest_path + "/vision_model.bin";
    auto ret = _packer->pack(_models, const_cast<char*>(merged_name.c_str()));
    if (ret != 0) {
        std::cerr << TAG << "Merge models FAILED! errCode=" << ret << std::endl;
    } else {
        std::cout << TAG << "Merge models SUCCESS! total models: " << _models.size() << std::endl;
        std::cout << TAG << "Merged model info:" << std::endl;

        auto ver_info_path = _dest_path + "/versions";
        std::ofstream ofs(ver_info_path.c_str());
        auto opened = ofs.is_open();
        if (opened) {
            ofs << "product: " << _config.product() << std::endl;
            std::time_t cur_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            ofs << "model_number: " << _models.size() << std::endl;
            ofs << "generated_date: " << std::ctime(&cur_time) << std::endl;
            ofs << "versions:" << std::endl;
        }
        for (const auto &m: _models) {
            std::cout << "ModelName: " << std::setw(30) << std::left << m.name << "\t"
                      << "Version: " << std::setw(40) << m.version << "\t"
                      << "Dtype: " << std::setw(5) << dtype_to_str(m.dtype) << "\t"
                      << "InferType: " << std::setw(10) << infer_type_to_str(m.infer_type) << "\t"
                      << "Device: " << std::setw(8) << device_to_str(m.device) << "\t"
                      << "Arch: " << std::setw(8) << _model_arch << "\t"
                      << "VersionCode: " << std::setw(8) << m.version_code << "\t"
                      << "Extends: " << std::setw(8) << m.extends << std::endl;
            if (opened) {
                ofs << "ModelName: " << std::setw(30) << std::left << m.name << "\t"
                    << "Version: " << std::setw(40) << m.version << "\t"
                    << "Dtype: " << std::setw(5) << dtype_to_str(m.dtype) << "\t"
                    << "InferType: " << std::setw(10) << infer_type_to_str(m.infer_type) << "\t"
                    << "Device: " << std::setw(8) << device_to_str(m.device) << "\t"
                    << "Arch: " << std::setw(8) << _model_arch << "\t"
                    << "VersionCode: " << std::setw(8) << m.version_code << "\t"
                    << "Extends: " << std::setw(8) << m.extends << std::endl;
            }
        }
        ofs.flush();
        ofs.close();
    }
}

ModelId MergeModelTool::str_to_id(const std::string& str) const {
    auto l_str = str_tolower(str);
    if (_model_id_map.find(l_str) == _model_id_map.end()) {
        return VISION_TYPE_UNKNOWN;
    }

    return _model_id_map.at(l_str);
}

DType MergeModelTool::str_to_dtype(const std::string& str) const {
    auto l_str = str_tolower(str);
    if (_dtype_map.find(l_str) == _dtype_map.end()) {
        return DTYPE_UNKNOWN;
    }

    return _dtype_map.at(l_str);
}

std::string MergeModelTool::dtype_to_str(DType dtype) const {
    for (const auto& p : _dtype_map) {
        if (p.second == dtype) {
            return p.first;
        }
    }
    return "unknown";
}

InferType MergeModelTool::str_to_infer_type(const std::string& str) const {
    auto l_str = str_tolower(str);
    if (_infer_type_map.find(l_str) == _infer_type_map.end()) {
        return INFER_TYPE_UNKNOWN;
    }

    return _infer_type_map.at(l_str);
}

std::string MergeModelTool::infer_type_to_str(InferType type) const {
    for (const auto &p: _infer_type_map) {
        if (p.second == type) {
            return p.first;
        }
    }
    return "unknown";
}

Device MergeModelTool::str_to_device(const std::string& str) const {
    auto l_str = str_tolower(str);
    if (_device_map.find(l_str) == _device_map.end()) {
        return CPU;
    }

    return _device_map.at(l_str);
}

std::string MergeModelTool::device_to_str(Device device) const {
    for (const auto& p : _device_map) {
        if (p.second == device) {
            return p.first;
        }
    }
    return "unknown";
}

std::string MergeModelTool::str_tolower(std::string s) const {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
    return s;
}

bool MergeModelTool::parse_config() {
    std::string cfg_content;
    if (!Util::read_file(_cfg_path, cfg_content)) {
        std::cerr << TAG << "Read model config file FAILED!" << std::endl;
        return false;
    }
//    std::cout << TAG << "config content: " << std::endl;
//    std::cout << TAG << cfg_content << std::endl;

    auto ret = plainbuffer::TextFormat::ParseFromString(cfg_content, &_config);
    if (!ret) {
        std::cerr << TAG << "Parse model config FAILED!" << std::endl;
        return false;
    }

    std::cout << TAG << "product: " << _config.product() << std::endl;
    std::cout << TAG << "model count: " << _config.model_size() << std::endl;
    return true;
}

} // namespace tools
} // namespace vision