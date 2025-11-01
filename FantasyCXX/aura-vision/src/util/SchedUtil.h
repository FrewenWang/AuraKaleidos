
#pragma once

#include "vision/config/runtime_config/RtConfig.h"
#include <memory>
#include <string>

namespace aura::vision {
class SchedUtil {
public:
    /**
     * 运行指令
     * @param cmd
     * @return
     */
    static std::string exec_cmd(const std::string &cmd);

    /**
     * 设置调度线程亲和性
     * @param ptr
     */
    static void set_affinity(std::shared_ptr<vision::RtConfig> ptr);
};
} // namespace aura::vision

