//
// Created by frewen on 1/29/23.
//

#include "aura/aura_utils/utils/PrintUtil.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include "aura/aura_utils/utils/AuraLog.h"


namespace aura::utils {

static const char *TAG = "PrintUtil";

void PrintUtil::print_info(const char *format, ...) {
    va_list args;
    va_start(args, format);
    printf("[debug info] ");
    vprintf(format, args);
    va_end(args);
}

void PrintUtil::printArray(const int *data, int len, const std::string &tag) {
    if (!data) {
        return;
    }

    std::stringstream ss;
    for (int i = 0; i < len; ++i) {
        ss << data[i] << " ";
    }
    ALOGD(TAG, "[%s]:%s", tag.c_str(), ss.str().c_str());
}

void PrintUtil::printArray(const float *data, int len, const std::string &tag) {
    if (!data) {
        return;
    }

    std::stringstream ss;
    for (int i = 0; i < len; ++i) {
        ss << data[i] << " ";
    }
    ALOGD(TAG, "[%s]:%s", tag.c_str(), ss.str().c_str());
}

void PrintUtil::printArray(const char *data, int len, const std::string &tag) {
    if (!data || len <= 0) {
        return;
    }

    print_info("%s: ", tag.c_str());
    std::stringstream ss;
    for (int i = 0; i < len; ++i) {
        ss << data[i] << " ";
        std::cout << "Hello:" << data[i] << " ";
    }
    ss << "\n";
    print_info(ss.str().c_str());
}

void PrintUtil::printArray(const unsigned char *data, int len, const std::string &tag) {
    if (!data) {
        return;
    }

    std::stringstream ss;
    for (int i = 0; i < len; ++i) {
        ss << (int) data[i] << " ";
    }
    ALOGD(TAG, "[%s]:%s", tag.c_str(), ss.str().c_str());
}

void PrintUtil::printArray(const double *data, int len, const std::string &tag) {
    if (!data) {
        return;
    }

    std::stringstream ss;
    for (int i = 0; i < len; ++i) {
        ss << data[i] << " ";
    }
    ALOGD(TAG, "[%s]:%s", tag.c_str(), ss.str().c_str());
}

}