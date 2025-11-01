#include "downloader.h"

#include <array>
#include <iostream>

#include "plainbuffer/text_format.h"
#include "util.h"

namespace vision {
namespace tools {

static const char* TAG = "ModelDownloader";

bool Downloader::search_or_pull(const std::string &cfg_file,
                                const std::string &cache_path,
                                const std::string &server_user,
                                const std::string &server_ip,
                                const std::string &model_hub_root,
                                const std::string &model_arch) {
    if (cfg_file.empty()) {
        std::cout << TAG << " [Error]: model config file name is empty!" << std::endl;
        return false;
    }
    // 读取配置文件内容，写入字符串
    std::string cfg_str;
    if (!Util::read_file(cfg_file, cfg_str)) {
        std::cout << TAG << " [Error]: cannot open model config file!" << std::endl;
        return false;
    }

    // ParseFromString，
    auto cfg = std::make_shared<ModelConfig>();
    if (!plainbuffer::TextFormat::ParseFromString(cfg_str, cfg.get())) {
        std::cout << TAG << " [Error]: parse config file error!" << std::endl;
        return false;
    }

    if (!pull_from_hub(cfg, cache_path, server_user, server_ip, model_hub_root, model_arch)) {
        std::cout << TAG << " [Error]: download model error!" << std::endl;
        return false;
    }

    return true;
}

bool Downloader::pull_from_hub(std::shared_ptr<ModelConfig> &cfg,
                               const std::string &cache_path,
                               const std::string &server_user,
                               const std::string &server_ip,
                               const std::string &model_hub_root,
                               const std::string &model_arch) {
    if (!cfg) {
        std::cout << TAG << " [Error]: model config is null!" << std::endl;
        return false;
    }
    std::cout << std::endl;
    for (const auto& m : cfg->model()) {
        std::cout << "model ability: " << m.ability() << std::endl
                  << "model version: " << m.version() << std::endl
                  << "model infer_type: " << m.infer_type() << std::endl
                  << "model dtype: " << m.dtype() << std::endl
                  << "model device: " << m.device() << std::endl;
        bool found = false;
        auto local_path = cache_path + "/" + m.ability() + "/" + m.version() + "/" + m.infer_type() + "/" + m.version();
        std::cout << local_path << std::endl;

        if (m.infer_type() == "ncnn") {
            if (m.dtype() == "fp32") {
                found = (Util::exists_file(local_path + ".bin") && Util::exists_file(local_path + ".param"));
            } else if (m.dtype() == "int8") {
                found = (Util::exists_file(local_path + "_int8.bin") && Util::exists_file(local_path + "_int8.param"));
            }
        } else if (m.infer_type() == "snpe") {
            if (m.dtype() == "fp32" and m.device() == "cpu") {
                found = Util::exists_file(local_path + ".dlc");
            } else if (m.dtype() == "int8" and m.device() == "dsp") {
                found = Util::exists_file(local_path + "_int8.dlc");
            }
        } else if (m.infer_type() == "paddle-lite") {
            local_path = cache_path + "/" + m.ability() + "/" + m.version() + "/" + m.infer_type() + "/"
                         + model_arch + "/" + m.version();
            std::cout << "paddle_lite path:" << local_path << std::endl;
            if (m.dtype() == "fp32") {
                found = Util::exists_file(local_path + ".nb");
            } else if (m.dtype() == "int8") {
                found = Util::exists_file(local_path + "_int8.nb");
            }
        } else if (m.infer_type() == "customize") {
            found = Util::exists_file(local_path + ".txt");
        } else if (m.infer_type() == "tf_lite") {
            found = Util::exists_file(local_path + ".tflite");
        } else if(m.infer_type() == "qnn"){
            found = Util::exists_file(local_path + ".bin");
            if(!found){ // support dlopen model so
                found = Util::exists_file(local_path + ".so");
            }
        }

        // todo: other model formats...

        if (!found) {
            std::cout << "model NOT FOUND, begin to pull from model hub..." << std::endl;
            if (pull_by_scp(m, cache_path, server_user, server_ip, model_hub_root, model_arch)) {
                std::cout << "model pulling DONE!" << std::endl;
            } else {
                std::cout << "model pulling ERROR! Please manually convert and copy the models into the cache directory"
                          << std::endl;
                return false;
            }
        } else {
            std::cout << "FOUND model in the cache" << std::endl;
        }
        std::cout << std::endl;
    }
    return true;
}

bool Downloader::pull_by_scp(const Model &m,
                             const std::string &local_cache,
                             const std::string &server_user,
                             const std::string &server_ip,
                             const std::string &model_hub_root,
                             const std::string &model_arch) {
    if (local_cache.empty() or server_user.empty() or server_ip.empty() or model_hub_root.empty()) {
        std::cout << TAG << "[Error]: params invalid" << std::endl;
        return false;
    }
    auto local_dir = local_cache + "/" + m.ability() + "/" + m.version();
    auto cmd = std::string("scp -pr ") + server_user + "@" + server_ip + ":" + model_hub_root + "/" + m.version() + "/" + m.infer_type();
    cmd += " " + local_dir + "/";
    if (!Util::exists_dir(local_dir)) {
        exec_cmd(std::string("mkdir -p ") + local_dir);
    }
    exec_cmd(cmd);
    return true;
}

std::string Downloader::exec_cmd(const std::string& cmd) {
    std::array<char, 128> buffer;
    std::string result;

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe)
    {
        return "";
    }

    while (fgets(buffer.data(), 128, pipe) != NULL) {
        result += buffer.data();
    }
    pclose(pipe);

    return result;
}

} // namesapce tools
} // namespace vision