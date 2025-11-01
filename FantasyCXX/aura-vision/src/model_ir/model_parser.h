#ifndef VISION_MODEL_PARSER_H
#define VISION_MODEL_PARSER_H

#include <vector>
#include "model_info.h"
#include "proto/model_def.plain.h"
#include "encryption.h"

namespace aura::vision{

class ModelParser {
public:
    ModelParser();
    int parse(const void* data, int len, std::vector<ModelInfo>& models);

private:
    void model_desc_to_info(const ModelDesc& desc, ModelInfo& info);
    std::unique_ptr<Encryption> _encryptor;
};

} // namespace vision

#endif //VISION_MODEL_PARSER_H
