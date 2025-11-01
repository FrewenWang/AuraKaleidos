#pragma once

#include <sys/types.h>

namespace nd {

struct TotalCpuOccupy {
    unsigned long user;
    unsigned long nice;
    unsigned long system;
    unsigned long idle;
};

struct ProcCpuOccupy {
    unsigned int pid;
    unsigned long utime;  //user time
    unsigned long stime;  //kernel time
    unsigned long cutime; //all user time
    unsigned long cstime; //all dead time
};

class DmipsUtil {
public:
    static unsigned long get_total_cpu_occupy();
    static unsigned long get_proc_cpu_occupy(pid_t pid);
    static const char* substr(const char *origin_buffer, char delimiter, int offset_num);
    static float get_cpu_oocupancy(unsigned long total_start, unsigned long total_end,
            unsigned long proc_start, unsigned long proc_end);
};

} // namespace nd