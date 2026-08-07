//
// LightBuffer text-format helper: convert a string token into a typed value.
//
#pragma once

#include <cstdint>
#include <string>

namespace plainbuffer {

class Util {
public:
    template<typename T>
    static T parse_from_str(const std::string &s);

private:
    Util() = default;
};

// --- specializations -------------------------------------------------------

template<>
inline int Util::parse_from_str<int>(const std::string &s) {
    return std::stoi(s);
}

template<>
inline short Util::parse_from_str<short>(const std::string &s) {
    return static_cast<short>(std::stoi(s));
}

template<>
inline char Util::parse_from_str<char>(const std::string &s) {
    return static_cast<char>(std::stoi(s));
}

template<>
inline int64_t Util::parse_from_str<int64_t>(const std::string &s) {
    return std::stoll(s);
}

template<>
inline float Util::parse_from_str<float>(const std::string &s) {
    return std::stof(s);
}

template<>
inline double Util::parse_from_str<double>(const std::string &s) {
    return std::stod(s);
}

template<>
inline bool Util::parse_from_str<bool>(const std::string &s) {
    if (s == "true" || s == "1" || s == "True" || s == "TRUE") {
        return true;
    }
    return false;
}

} // namespace plainbuffer
