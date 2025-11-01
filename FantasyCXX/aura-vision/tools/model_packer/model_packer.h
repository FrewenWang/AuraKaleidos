#ifndef VISION_MODEL_PACKER_H
#define VISION_MODEL_PACKER_H

#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "proto/model_config.plain.h"
#include "model_ir/model_info.h"
#include "model_ir/encryption.h"

namespace vision {
namespace tools {

struct ModelPackInfo {
    ModelId id;
    DType dtype;
    InferType infer_type;
    Device device;
    std::string version;
    std::string name;
    bool enable_omp;
    int version_code;
    std::string extends;
    /** 模型支持的系统架构 */
    std::string modelArch;
    std::vector<std::tuple<std::string, bool>> files; // tuple: [filename, need to encrypt]
};

class ModelPacker {
public:
    ModelPacker();
    int pack(const std::vector<ModelPackInfo>& pack_infos, const std::string& out_file);
    void set_encrpt_type(ModelEncryptType encrypt) { _encrypt_type = encrypt; }

private:
    std::unique_ptr<Encryption> _encryptor;
    ModelEncryptType _encrypt_type;
};

} // namespace tools
} // namespace vision

#endif //VISION_MODEL_PACKER_H
