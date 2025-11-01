//
// Created by wangyan67 on 2020/5/27.
//

#ifndef VISION_MERGE_MODEL_H
#define VISION_MERGE_MODEL_H

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "proto/model_config.plain.h"
#include "model_packer.h"

namespace vision {
namespace tools {

class MergeModelTool {
public:
    MergeModelTool(std::string cfg_path, std::string src_path, std::string dest_path, std::string model_arch);
    ~MergeModelTool();
    void merge_model(ModelEncryptType encrypt = TP_DES);

private:
    bool parse_config();
    ModelId str_to_id(const std::string& str) const;
    DType str_to_dtype(const std::string& str) const;
    std::string dtype_to_str(DType dtype) const;
    InferType str_to_infer_type(const std::string& str) const;
    /**
     * 打印模型引擎类型
     * @param type
     * @return
     */
    std::string infer_type_to_str(InferType type) const;
    Device str_to_device(const std::string& str) const;
    std::string device_to_str(Device device) const;
    std::string str_tolower(std::string s) const;
    void remove_end_slash(std::string& str);

    std::string _cfg_path;
    std::string _src_path;
    std::string _dest_path;
    std::string _model_arch;

    std::vector<ModelPackInfo> _models;
    std::unique_ptr<ModelPacker> _packer;
    ModelConfig _config;
    static const std::unordered_map<std::string, ModelId> _model_id_map;
    static const std::unordered_map<std::string, DType> _dtype_map;
    static const std::unordered_map<std::string, InferType> _infer_type_map;
    static const std::unordered_map<std::string, Device> _device_map;
    static const std::unordered_map<std::string, std::vector<std::tuple<std::string, bool>>> _model_extensions;
};

} // namespace tools
} // namespace vision

#endif //VISION_MERGE_MODEL_H
