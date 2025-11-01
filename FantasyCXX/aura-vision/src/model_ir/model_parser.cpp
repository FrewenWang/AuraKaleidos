#include "model_parser.h"
#include "vision/core/common/VMacro.h"
#include "vision/util/log.h"
#include "vision/util/VaAllocator.h"

namespace aura::vision {

static const char* TAG = "ModelParser";

ModelParser::ModelParser() {
//    _encryptor = std::make_unique<Encryption>();
    _encryptor = std::unique_ptr<Encryption>(new Encryption());
    _encryptor->init(TP_DES, nullptr, nullptr);
}

int ModelParser::parse(const void* data, int len, std::vector<ModelInfo>& models) {
    models.clear();
    if (!data || len <= 0) {
        VLOGE(TAG, "Invalid memory or length for model parser");
        return -1;
    }

    auto model_data = std::make_shared<ModelData>();
    if (!model_data->ParseFromArray(data, len)) {
        VLOGE(TAG, "Parse model data FAILED!");
        return -1;
    }

    // validate magic
    if (model_data->model_flag().magic() != ModelMagic::magic()) {
        VLOGE(TAG, "Model magic ERROR!");
        return -2;
    }

    // decrypt and parse header
    auto* header_bytes = model_data->model_header_ptr();
    auto hader_len = model_data->model_header_len();
    auto* decrypted_header_bytes = VaAllocator::allocate(hader_len);
    auto encrypt = model_data->model_flag().encrypt_type();
    _encryptor->decrypt(encrypt,
                        (unsigned char*)header_bytes,
                        (unsigned char*)decrypted_header_bytes,
                        hader_len);

    auto header = std::make_shared<ModelHeader>();
    header->ParseFromArray(decrypted_header_bytes, hader_len);
    VaAllocator::deallocate(decrypted_header_bytes);

    auto model_cnt = model_data->model_flag().model_cnt();
    if (model_cnt != static_cast<int>(header->model_descs().size())) {
        VLOGE(TAG, "Model cnt ERROR! (%d, %lu)", model_cnt, header->model_descs().size());
        return -3;
    }

    const auto* blob = model_data->model_blob_ptr();
    auto blob_len = model_data->model_blob_len();
    VLOGI(TAG, "Model blob len=%d", V_TO_INT(blob_len));

    // iterate all the models and parse them into modelInfos
    for (const auto& desc : header->model_descs()) {
        ModelInfo info;
        model_desc_to_info(desc, info);

        if (desc.mem_blocks_size() <= 0) {
            VLOGE(TAG, "No mem_block found in the model: %d", desc.model_id());
            continue;
        }

        for (const auto& mem_block : desc.mem_blocks()) {
            auto mem_len = mem_block.len();
            auto offset = mem_block.offset();
            auto encrypted = mem_block.encrypted();
            auto* mem_addr = blob + offset;
            void* decrypted_mem = VaAllocator::allocate(mem_len);
            if (encrypted) {
                // decrypt
                _encryptor->decrypt(encrypt,
                                    (unsigned char*)mem_addr,
                                    (unsigned char*)decrypted_mem,
                                    mem_len);
            } else {
                memcpy(decrypted_mem, mem_addr, mem_len);
            }
            info.blobs.emplace_back(mem_len, decrypted_mem);
        }

        models.emplace_back(info);
    }

    return 0;
}

void ModelParser::model_desc_to_info(const ModelDesc& desc, ModelInfo& info) {
    info.infer_type = static_cast<InferType>(desc.infer_type());
    info.id = static_cast<ModelId>(desc.model_id());
    info.dtype = static_cast<DType>(desc.dtype());
    info.device = static_cast<Device>(desc.device());
    info.version = desc.version();
    info.enable_omp = desc.enable_omp();
    info.versionCode = desc.versionCode();
    info.extends = desc.extends();
}

} // namespace vision