#include "mem_util.h"

#include <array>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <fstream>
#include <string>
#include <cstdlib>
#include <sstream>

#include "log_util.h"

#define VMRSS_LINE 18 // VMRSS所在行, 注:根据不同的系统,位置可能有所区别

namespace nd {

//获取进程占用内存
unsigned int MemUtil::get_proc_mem(unsigned int pid) {
    char file_name[64] = {0};
    FILE *fd;
    char line_buff[512] = {0};
    sprintf(file_name, "/proc/%d/status", pid);

    fd = fopen(file_name, "r");
    if (nullptr == fd) {
        return 0;
    }

    char name[64];
    int vmrss;
    for (int i = 0; i < VMRSS_LINE - 1; i++) {
        fgets(line_buff, sizeof(line_buff), fd);
    }

    fgets(line_buff, sizeof(line_buff), fd);
    sscanf(line_buff, "%s %d", name, &vmrss);
    fclose(fd);

    return vmrss;
}

/// 耗时较长，不使用
std::vector<float> MemUtil::get_proc_cpu_mem_via_top() {
    std::string cmd = "top -n 1 -d 1 -q -p " + std::to_string(getpid());

    WLog::i(cmd);
    auto res = exec_top_cmd(cmd);
//    WLog::i("cmd_res= " + res);

    std::vector<std::string> tokens;
    split(res, tokens);
    std::vector<float> info(2);
    if (tokens.size() < 9) {
        return info;
    }

    info[0] = std::stof(tokens[5]);
    info[1] = std::stof(tokens[8]);
    return info;
}

std::string MemUtil::exec_top_cmd(const std::string& cmd) {
    std::array<char, 256> buffer;
    std::string result;

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe)
    {
        WLog::e("execute cmd error: " + cmd);
        return "";
    }

    while (fgets(buffer.data(), 256, pipe) != NULL) {
        result += buffer.data();
    }
    pclose(pipe);

    // 从 pid 开始截断
    auto start_pos = result.find(std::to_string(getpid()));
    result = result.substr(start_pos);
    return result;
}

void MemUtil::split(const std::string& s, std::vector<std::string>& tokens, const std::string& delimiters, bool keep_delimiter) {
    auto last_pos = s.find_first_not_of(delimiters, 0);
    auto pos = s.find_first_of(delimiters, last_pos);
    while (std::string::npos != pos || std::string::npos != last_pos) {
        if (!keep_delimiter) {
            tokens.push_back(s.substr(last_pos, pos - last_pos));
        } else {
            tokens.push_back(s.substr(last_pos, pos - last_pos + 1));
        }
        last_pos = s.find_first_not_of(delimiters, pos);
        pos = s.find_first_of(delimiters, last_pos);
    }
}

}