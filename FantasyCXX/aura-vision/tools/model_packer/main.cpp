#include <iostream>
#include <memory>

#include "downloader.h"
#include "MergeTool.h"

int main(int argc, char** argv) {
    // check arguments
    if (argc < 4) {
        std::cout << "[MergeModelTool] Usage: merge_model_tool [config_path] [src_models_dir] [dest_model_dir] "
                     "[server_ip] [server_user] [model_hub_root] [encrypt_type]" << std::endl
                  << "encrypt_type = 0: None, 1: DES, 2: AES" << std::endl;
        return 0;
    }

    std::cout << "[MergeModelTool] ============ Merge Models ===========" << std::endl;

    auto cfg_path = argv[1];
    auto src_dir = argv[2];
    auto dest_dir = argv[3];

    std::string server_ip, server_user, model_hub_root, model_arch;
    if (argc > 4) {
         server_ip = argv[4];
    }
    if (argc > 5) {
        server_user = argv[5];
    }
    if (argc > 6) {
        model_hub_root = argv[6];
    }
    vision::ModelEncryptType enc = vision::TP_DES;
    if (argc > 7) {
        enc = static_cast<vision::ModelEncryptType>(atoi(argv[7]));
    }

    if (argc > 8) {
        model_arch = argv[8];
    }

    std::cout << "[MergeModelTool] arguments: " << std::endl
              << "cfg_path: " << cfg_path << std::endl
              << "src_dir: " << src_dir << std::endl
              << "dest_dir: " << dest_dir << std::endl
              << "server_ip: " << server_ip << std::endl
              << "server_user: " << server_user << std::endl
              << "model_hub_root: " << model_hub_root << std::endl
              << "encrypt_type: " << enc << std::endl
              << "model_arch: " << model_arch << std::endl;

    // search_or_download
    if (!vision::tools::Downloader::search_or_pull(cfg_path, src_dir, server_user,
                                                   server_ip, model_hub_root, model_arch)) {
        std::cout << "[MergeModelTool] terminated!" << std::endl;
        return -1;
    }
    // merge models
    auto packer = std::make_unique<vision::tools::MergeModelTool>(cfg_path, src_dir, dest_dir, model_arch);
    packer->merge_model(enc);
    std::cout << std::endl;
}