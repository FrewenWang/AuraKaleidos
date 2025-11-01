#ifndef VISION_TOOLS_UTIL_H
#define VISION_TOOLS_UTIL_H

#include <cstdio>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace vision {
namespace tools {

class Util {
public:
    static inline bool read_file(const std::string& file_name, std::string& content) {
        std::ifstream ifs(file_name);
        if (!ifs.is_open()) {
            return false;
        }
        std::string str((std::istreambuf_iterator<char>(ifs)),(std::istreambuf_iterator<char>()));
        ifs.close();
        content = std::move(str);
        return true;
    }

    static inline bool exists_file(const std::string& file_path) {
        if (FILE *file = fopen(file_path.c_str(), "r")) {
            fclose(file);
            return true;
        } else {
            return false;
        }
    }

    static inline bool exists_dir(const std::string& dir_path) {
        struct stat info{};

        if (stat(dir_path.c_str(), &info) != 0) {
            return false;
        }
        return (info.st_mode & S_IFDIR) != 0;
    }
};

} // namespace tools
} // namespace vision

#endif // VISION_TOOLS_UTIL_H