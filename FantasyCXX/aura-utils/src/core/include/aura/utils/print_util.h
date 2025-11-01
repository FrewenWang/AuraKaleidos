//
// Created by frewen on 1/29/23.
//
#pragma once

#include <iostream>

namespace aura::utils {

class PrintUtil {
public:
    static void Print_info(const char *format, ...);

    static void PrintArray(const float *data, int len, const std::string &tag = "");

    static void PrintArray(const char *data, int len, const std::string &tag = "");

    static void PrintArray(const unsigned char *data, int len, const std::string &tag = "");

    static void PrintArray(const int *data, int len, const std::string &tag = "");

    static void PrintArray(const double *data, int len, const std::string &tag = "");

    static void PrintVector(const float *data, int len, const std::string &tag = "");
};

}
