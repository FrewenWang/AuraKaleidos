//
// Created by Frewen.Wang on 25-4-24.
//
# pragma once

#include <unistd.h>

class KernelUtils {
public:
    /**
     * 读取内核文件（.cl） 文件
     * @param filename kernel内核文件的路径地址
     * @return
     */
    static char *readKernelSource(const char *filename);


    static char **readKernelSourceList(const char *file_name[], size_t length);
};
