//
// FileUtil - 轻量文件写出工具
// 兼容 LightBuffer 代码生成器（CodeGeneratorCpp.cpp）对
// aura/auralib/utils/FileUtil.h 的引用：FileUtil::save_file(path, content)。
//
#ifndef AURA_AURALIB_UTILS_FILEUTIL_H
#define AURA_AURALIB_UTILS_FILEUTIL_H

#include <string>
#include <fstream>

namespace aura {
namespace auralib {
namespace utils {

// 将 content 以二进制方式写入 path；成功返回 true，失败返回 false。
inline bool save_file(const std::string& path, const std::string& content) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        return false;
    }
    ofs << content;
    return true;
}

}  // namespace utils
}  // namespace auralib
}  // namespace aura

#endif  // AURA_AURALIB_UTILS_FILEUTIL_H
