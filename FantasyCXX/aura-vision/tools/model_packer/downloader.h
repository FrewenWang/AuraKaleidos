#ifndef VISION_TOOLS_DOWNLOADER_H
#define VISION_TOOLS_DOWNLOADER_H

#include <memory>
#include <string>

#include "proto/model_config.plain.h"

namespace vision {
namespace tools {

class Downloader {
public:
    static bool search_or_pull(const std::string &cfg_file,
                               const std::string &cache_path,
                               const std::string &server_user,
                               const std::string &server_ip,
                               const std::string &model_hub_root,
                               const std::string &model_arch);

private:
    static bool pull_from_hub(std::shared_ptr<ModelConfig> &cfg,
                              const std::string &cache_path,
                              const std::string &server_user,
                              const std::string &server_ip,
                              const std::string &model_hub_root,
                              const std::string &model_arch);

    static bool pull_by_scp(const Model &m,
                            const std::string &local_cache,
                            const std::string &server_user,
                            const std::string &server_ip,
                            const std::string &model_hub_root,
                            const std::string &model_arch);

    static std::string exec_cmd(const std::string& cmd);
};

} // namesapce tools
} // namespace vision

#endif //VISION_TOOLS_DOWNLOADER_H
