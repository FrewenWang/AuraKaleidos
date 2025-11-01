#include "dmips_util.h"

#include <stdio.h>
#include <string.h>
#include <unistd.h>

namespace nd {

unsigned long DmipsUtil::get_total_cpu_occupy() {
    FILE *fd;
    TotalCpuOccupy t;

    fd = fopen("/proc/stat", "r");
    if (nullptr == fd) {
        return 0;
    }

    char name[4] = {0};
    unsigned long a6 = 0;
    unsigned long a7 = 0;
    unsigned long a8 = 0;
    fscanf(fd, "%s %ld %ld %ld %ld %ld %ld %ld", name, &t.user, &t.nice, &t.system, &t.idle, &a6, &a7, &a8);
    fclose(fd);

    return (t.user + t.nice + t.system + t.idle + a6 + a7 + a8);
}

unsigned long DmipsUtil::get_proc_cpu_occupy(pid_t pid) {
//    pid = static_cast<unsigned int>(getpid());

    char file_name[64] = {0};
    ProcCpuOccupy t;
    FILE *fd;
    char line_buff[1024] = {0};
    sprintf(file_name, "/proc/%d/stat", pid);

    fd = fopen(file_name, "r");
    if (nullptr == fd) {
        return 0;
    }

    fgets(line_buff, sizeof(line_buff), fd);

    sscanf(line_buff, "%u", &t.pid);
    const char *q = substr(line_buff, ' ', 14);
    sscanf(q, "%ld %ld %ld %ld", &t.utime, &t.stime, &t.cutime, &t.cstime);
    fclose(fd);

    return (t.utime + t.stime + t.cutime + t.cstime);
}

const char *DmipsUtil::substr(const char *origin_buffer, char delimiter, int offset_num) {
    const char *p = origin_buffer;
    int len = static_cast<int>(strlen(origin_buffer));
    int count = 0;
    for (int i = 0; i < len; i++) {
        if (delimiter == *p) {
            count++;
            if (count == offset_num - 1) {
                p++;
                break;
            }
        }
        p++;
    }
    return p;
}

float DmipsUtil::get_cpu_oocupancy(unsigned long total_start, unsigned long total_end,
        unsigned long proc_start, unsigned long proc_end) {
    auto total_dif = total_end - total_start;
    auto proc_dif = proc_end - proc_start;
    float occu = total_dif ? static_cast<float>(proc_dif) / total_dif * 100 : 0;
    return occu;
}

}