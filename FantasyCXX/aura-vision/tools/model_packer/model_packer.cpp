#include "model_packer.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include "proto/model_def.plain.h"
#include "util/file_util.h"
#include "vision/util/VaAllocator.h"

namespace vision {
namespace tools {

ModelPacker::ModelPacker() : _encrypt_type(ModelEncryptType::TP_DES) {
    _encryptor = std::make_unique<Encryption>();
    _encryptor->init(_encrypt_type, nullptr, nullptr);
}

int ModelPacker::pack(const std::vector<ModelPackInfo>& pack_infos, const std::string& out_file) {
    int offset = 0;
    int valid_model_cnt = 0;
    auto model_data = std::make_shared<ModelData>();
    auto flag = model_data->mutable_model_flag();
    flag->set_magic(ModelMagic::magic());
    flag->set_encrypt_type(static_cast<int>(_encrypt_type));

    auto header = std::make_shared<ModelHeader>();
    std::vector<MemBlob> model_buffers;

    // read src model files
    for (auto& pi : pack_infos) {
        std::vector<MemBlob> tmp_buffers;
        std::vector<bool> tmp_buffers_encrypted;
        for (const auto& f : pi.files) {
            // read model content
            const auto& filename = std::get<0>(f);
            auto file_len = FileUtil::get_file_len(filename);
            if (file_len <= 0) {
                continue;
            }

            auto* buffer = VaAllocator::allocate(file_len);
            if (!FileUtil::read_file(filename, buffer, file_len)) {
                std::cerr << "Open model file: " << filename << " FAILED!" << std::endl;
                continue;
            }

            // encrypt
            auto need_encrypt = std::get<1>(f);
            if (need_encrypt) {
                _encryptor->encrypt(_encrypt_type,
                                    reinterpret_cast<unsigned char*>(buffer),
                                    reinterpret_cast<unsigned char*>(buffer),
                                    file_len);
            }

            tmp_buffers.emplace_back(file_len, buffer);
            tmp_buffers_encrypted.emplace_back(need_encrypt);
        }

//        if (tmp_buffers.size() < pi.files.size()) {
//            std::cerr << "Not all model file is available, ignore model: " << pi.name << std::endl;
//            continue;
//        }

        auto desc = header->add_model_descs();
        desc->set_model_id(static_cast<int>(pi.id));
        desc->set_infer_type(static_cast<int>(pi.infer_type));
        desc->set_device(static_cast<int>(pi.device));
        desc->set_dtype(static_cast<int>(pi.dtype));
        desc->set_version(pi.version);
        desc->set_enable_omp(pi.enable_omp);
        desc->setVersionCode(pi.version_code);
        desc->setExtends(pi.extends);

        for (int i = 0; i < tmp_buffers.size(); ++i) {
            auto block = desc->add_mem_blocks();
            block->set_len(tmp_buffers[i].len);
            block->set_offset(offset);
            block->set_encrypted(tmp_buffers_encrypted[i]);
            offset += tmp_buffers[i].len;
        }

        valid_model_cnt++;
        model_buffers.insert(model_buffers.end(), tmp_buffers.begin(), tmp_buffers.end());
    }

    flag->set_model_cnt(valid_model_cnt);

    // make header bytes
    auto header_len = header->ByteSizeLong();
    auto* header_buf = VaAllocator::allocate(header_len);
    header->SerializeToArray(header_buf, header_len);
    // encrypt header
    _encryptor->encrypt(_encrypt_type,
                        reinterpret_cast<unsigned char*>(header_buf),
                        reinterpret_cast<unsigned char*>(header_buf),
                        header_len);
    const auto* header_buf_c = (const char*)header_buf;
    model_data->set_model_header(header_buf_c, header_len);
    std::cout << "Header size: " << header_len << std::endl;

    // make blob bytes
    auto total_size = offset;
    auto* model_buffers_data = VaAllocator::allocate(total_size);
    char* dest_ptr = (char*)model_buffers_data;
    for (auto& buf : model_buffers) {
        memcpy(dest_ptr, buf.data, buf.len);
        dest_ptr += buf.len;
    }
    const auto* model_buf_c = (const char*)model_buffers_data;
    model_data->set_model_blob(model_buf_c, total_size);
    std::cout << "Blob size: " << total_size << std::endl;

    // serialize
    auto raw_model_len = model_data->ByteSizeLong();
    auto* raw_model_data = (char*)malloc(raw_model_len);
    std::cout << "Whole model len=" << raw_model_len << std::endl;
    model_data->SerializeToArray(raw_model_data, raw_model_len);

    // persist
    std::ofstream ofs(out_file);
    if (!ofs.is_open()) {
        std::cerr << "Write model file FAILED!" << std::endl;
    }
    ofs.write(static_cast<char*>(raw_model_data), raw_model_len);
    ofs.close();

    // release buffers
    VaAllocator::deallocate(model_buffers_data);
    for (auto& buf : model_buffers) {
        VaAllocator::deallocate(buf.data);
    }

    return 0;
}

} // namespace tools
} // namespace vision