#pragma once

#include <algorithm>
#include <unordered_map>

#include "model_ir/model_info.h"
#include "vision/config/runtime_config/RtConfig.h"
#include "vision/util/log.h"

namespace aura::vision {
/**
 * 模型推理器配置的映射转换器.
 * 根据perception-vision-ability/config/*.prototxt文件中的模型配置拉转换成对应的
 */
class InferenceConverter {
public:
    template<typename Tid>
    static Tid getId(const std::string &tag);

    template<typename Tid>
    static std::string get_tag(const Tid &id);

private:
    static std::string str_tolower(std::string s);

    static const std::unordered_map<std::string, ModelId> modelIdMap;
    static const std::unordered_map<std::string, DType> dTypeMap;
    static const std::unordered_map<std::string, InferType> inferTypeMap;
};

inline std::string InferenceConverter::str_tolower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
    return s;
}

template<typename Tid>
inline std::string InferenceConverter::get_tag(const Tid &id) {
    VLOGE("InferenceConverter", "Unsupported typename for TagIdConverter");
    return std::string();
}

template<>
inline ModelId InferenceConverter::getId<ModelId>(const std::string &tag) {
    auto l_str = str_tolower(tag);
    if (modelIdMap.find(l_str) == modelIdMap.end()) {
        return VISION_TYPE_UNKNOWN;
    }
    return modelIdMap.at(l_str);
}

template<>
inline std::string InferenceConverter::get_tag<ModelId>(const ModelId &id) {
    for (const auto &p: modelIdMap) {
        if (p.second == id) {
            return p.first;
        }
    }
    return std::string();
}

template<>
inline DType InferenceConverter::getId<DType>(const std::string &tag) {
    auto l_str = str_tolower(tag);
    if (dTypeMap.find(l_str) == dTypeMap.end()) {
        return DTYPE_UNKNOWN;
    }
    return dTypeMap.at(l_str);
}

template<>
inline std::string InferenceConverter::get_tag<DType>(const DType &id) {
    for (const auto &p: dTypeMap) {
        if (p.second == id) {
            return p.first;
        }
    }
    return std::string();
}

template<>
inline InferType InferenceConverter::getId<InferType>(const std::string &tag) {
    auto l_str = str_tolower(tag);
    if (inferTypeMap.find(l_str) == inferTypeMap.end()) {
        return INFER_TYPE_UNKNOWN;
    }
    return inferTypeMap.at(l_str);
}

template<>
inline std::string InferenceConverter::get_tag<InferType>(const InferType &id) {
    for (const auto &p: inferTypeMap) {
        if (p.second == id) {
            return p.first;
        }
    }
    return std::string();
}

} // namespace aura::vision

