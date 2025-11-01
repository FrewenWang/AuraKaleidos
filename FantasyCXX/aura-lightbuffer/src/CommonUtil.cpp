//
// Created by Frewen.Wong on 2022/4/23.
//
#include <algorithm>
#include "lightbuffer/CommonUtil.h"

namespace aura::light_buffer {

std::string CommonUtil::toUpper(const std::string &s) {
    std::string out = s;
    std::transform(s.begin(), s.end(), out.begin(), ::toupper);
    return out;
}

std::string CommonUtil::toLower(const std::string &s) {
    std::string out = s;
    std::transform(s.begin(), s.end(), out.begin(), ::tolower);
    return out;
}

}
