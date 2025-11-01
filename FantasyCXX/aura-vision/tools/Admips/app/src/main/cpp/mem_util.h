#ifndef ADMIPS_MEM_UTIL_H
#define ADMIPS_MEM_UTIL_H

#include <string>
#include <vector>

namespace nd {

class MemUtil {
public:
    static unsigned int get_proc_mem(unsigned int pid);
    static std::vector<float> get_proc_cpu_mem_via_top();

private:
    static std::string exec_top_cmd(const std::string& cmd);
    static void split(const std::string& s, std::vector<std::string>& tokens,
            const std::string& delimiters = " ", bool keep_delimiter = false);
};

}

#endif //ADMIPS_MEM_UTIL_H
