//
// Created by Frewen.Wang on 25-4-24.
//

#include "KernelUtils.h"

#include <cstdio>
#include <cstdlib>

char *KernelUtils::readKernelSource(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Failed to open kernel file");
        exit(EXIT_FAILURE);
    }
    fseek(fp, 0, SEEK_END);
    // 确定文件的大小
    const size_t size = ftell(fp);
    rewind(fp);
    // 分配内存空间，读取文件内容，获取kernel程序
    const auto src = static_cast<char *>(malloc(size + 1));
    fread(src, 1, size, fp);
    // 添加结束符号
    src[size] = '\0';
    // 关键文件流
    fclose(fp);
    return src;
}

char **KernelUtils::readKernelSourceList(const char *file_name[], size_t length) {
    FILE *fp;
    size_t program_size[length];
    char *src[length];
    //读取两个文件源代码
    for (int i = 0; i < length; i++) {
        fp = fopen(file_name[i], "r");
        fseek(fp, 0, SEEK_END);
        program_size[i] = ftell(fp);
        rewind(fp);
        src[i] = static_cast<char *>(malloc(program_size[i] + 1));
        fread(src[i], sizeof(char), program_size[i], fp);
        fclose(fp);
    }
    return src;
}
