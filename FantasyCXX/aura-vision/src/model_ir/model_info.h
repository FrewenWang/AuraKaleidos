#ifndef VISION_MODEL_INFO_H
#define VISION_MODEL_INFO_H

#include "vision/core/common/VConstants.h"
#include "vision/core/common/VTensor.h"
#include "proto/model_port.plain.h"
#ifdef USE_QNN
#include "QnnWrapperUtils.hpp"
#endif

namespace aura::vision {

/// Inference framework type
enum InferType {
    NCNN = 1000,
    PADDLE_LITE = 1001,
    TNN = 1002,
    TF_LITE = 1003,
    SNPE = 2000,
    CUSTOMIZE = 3000,
    QNN = 4000,
    ONNX = 5000,
    INFER_TYPE_UNKNOWN
};

/**
 * 定义进行推力加速的硬件信息
 */
enum Device {
    CPU = 0,
    GPU = 1,
    DSP = 2,
    HTP = 3
};

struct MemBlob {
    int len;
    void* data;
    void* mutable_data() { return const_cast<void*>(data); }
    MemBlob(int len_, void* data_) : len(len_), data(data_) {}
};

struct ModelInfo {
    ModelId id;
    DType dtype;
    InferType infer_type;
    Device device;
    bool enable_omp;
    std::string version;
    std::vector<MemBlob> blobs;
    int versionCode = 0;
    std::string extends;

#ifdef USE_QNN
    // 如下字段目前专门位QnnPredictor使用
    bool hasLoaded = false;    // 标记此模型是否加载完毕
    qnn_wrapper_api::GraphInfo_t **graphsInfo;  // 生成图信息
    uint32_t graphsCount;                       // graphsCount
#endif

    PortDesc port_desc; // parsed from runtime config

    std::string to_string(bool verbose = false) const;
    void release();
};

class ModelMagic {
public:
    static std::string magic();
private:
    static std::string str_to_hex(const std::string& data);
    static const std::string _str_magic;
};

} // namespace vision

#endif //VISION_MODEL_INFO_H
