//
// Created by frewen on 1/29/23.
//

#include "aura/utils/print_util.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include "aura/utils/logger.h"


namespace aura::utils
{

static const char *TAG = "PrintUtil";

void PrintUtil::Print_info(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    printf("[debug info] ");
    vprintf(format, args);
    va_end(args);
}

void PrintUtil::PrintArray(const int *data, int len, const std::string &tag)
{
    if (!data)
    {
        return;
    }

    std::stringstream ss;
    for (int i = 0; i < len; ++i)
    {
        ss << data[i] << " ";
    }
    // AURA_PRINTD(TAG, "[%s]:%s", tag.c_str(), ss.str().c_str());
}

void PrintUtil::PrintArray(const float *data, int len, const std::string &tag)
{
    if (!data)
    {
        return;
    }

    std::stringstream ss;
    for (int i = 0; i < len; ++i)
    {
        ss << data[i] << " ";
    }
    // AURA_PRINTD(TAG, "[%s]:%s", tag.c_str(), ss.str().c_str());
}

void PrintUtil::PrintArray(const char *data, int len, const std::string &tag)
{
    if (!data || len <= 0)
    {
        return;
    }

    Print_info("%s: ", tag.c_str());
    std::stringstream ss;
    for (int i = 0; i < len; ++i)
    {
        ss << data[i] << " ";
        std::cout << "Hello:" << data[i] << " ";
    }
    ss << "\n";
    Print_info(ss.str().c_str());
}

void PrintUtil::PrintArray(const unsigned char *data, int len, const std::string &tag)
{
    if (!data)
    {
        return;
    }

    std::stringstream ss;
    for (int i = 0; i < len; ++i)
    {
        ss << (int)data[i] << " ";
    }
    // AURA_PRINTD(TAG, "[%s]:%s", tag.c_str(), ss.str().c_str());
}

void PrintUtil::PrintArray(const double *data, int len, const std::string &tag)
{
    if (!data)
    {
        return;
    }

    std::stringstream ss;
    for (int i = 0; i < len; ++i)
    {
        ss << data[i] << " ";
    }
    // AURA_PRINTD(TAG, "[%s]:%s", tag.c_str(), ss.str().c_str());
}

void PrintUtil::PrintVector(const float *data, int len, const std::string &tag)
{

}

}
