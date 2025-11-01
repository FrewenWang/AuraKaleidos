//
// Created by wangyan67 on 2019-10-31.
//

#include "SchedUtil.h"
#include "vision/config/runtime_config/RtConfig.h"
#include "vision/core/common/VConstants.h"
#include <array>
#include <sched.h>

using namespace aura::vision;

std::string SchedUtil::exec_cmd(const std::string &cmd) {
    std::array<char, 128> buffer;
    std::string result;

#ifdef __ANDROID__
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe)
    {
        return "";
    }

    while (fgets(buffer.data(), 128, pipe) != NULL) {
        result += buffer.data();
    }
    pclose(pipe);
#endif
    return result;
}

void SchedUtil::set_affinity(std::shared_ptr<vision::RtConfig> rtConfig) {

    if (rtConfig->threadAffinityPolicy != TP_BIG_CORE) {
        return;
    }

    // currently, only android devices have big.LITTLE core architecture
#ifdef __ANDROID__
    std::string cmd = "grep -c processor /proc/cpuinfo";
    std::string s_cpu_cnt = exec_cmd(cmd);
    int cpu_cnt = static_cast<int>(strtol(s_cpu_cnt.c_str(), nullptr, 0));

    cpu_set_t cpu_mask;
    CPU_ZERO(&cpu_mask);

    std::string cmd_base = "cat /sys/devices/system/cpu/cpu";

    std::vector<long> freqs;

    // find big cores
    for (int i = 0; i < cpu_cnt; ++i) {
        cmd = cmd_base + std::to_string(i) + "/cpufreq/cpuinfo_max_freq";
        std::string s_freq = exec_cmd(cmd);
        freqs.emplace_back(strtol(s_freq.c_str(), nullptr, 0));
    }

    auto max_ele = std::max_element(freqs.begin(), freqs.end());
    long max_freq = *max_ele;

    // bind thread to big cores
    for (int i = 0; i < cpu_cnt; ++i) {
        if (freqs[i] == max_freq) {
            CPU_SET(i, &cpu_mask);
        }
    }
    sched_setaffinity(0, sizeof(cpu_mask), &cpu_mask);
#endif
}