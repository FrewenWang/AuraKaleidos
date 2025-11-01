#ifndef UTILS_H__
#define UTILS_H__

typedef struct
{
    unsigned char *addr;
    int size;
    int fd;
} Meminfo;

int UtilsInitIon(long long *ion_dev_fd);
int UtilsUnitIon(long long ion_dev_fd);

int AllocBuffer(long long ion_dev_fd, int mem_size, Meminfo *mem_buffer);

int DeleteBuffer(Meminfo *mem_buffer);

#endif // UTILS_H__