#include "file_util.h"

#include <fstream>

#include "vision/util/log.h"

namespace aura::vision {

using namespace std;

static const char *TAG = "FileUtil";

void FileUtil::read_file(char *file_path, std::string &content) {
    ifstream is(file_path, ios::in | ios::ate);
    if (is) {
        size_t size = is.tellg();
        std::string str(size, '\0'); // construct string to stream size
        is.seekg(0);
        if (is.read(&str[0], size)) {
            content = str;
        }
    }
    is.close();
}

bool FileUtil::read_file(const std::string &file_path, std::vector<char> &buffer) {
    buffer.clear();
    std::ifstream ifs(file_path, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) {
        VLOGE(TAG, "open model file failed! filePath[%s]", file_path.c_str());
        return false;
    }

    auto len = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    buffer.reserve(len);
    buffer.insert(buffer.begin(), std::istream_iterator<char>(ifs), std::istream_iterator<char>());
    ifs.close();

    return true;
}

bool FileUtil::read_file(const std::string &file_path, void *buffer, int len) {
    std::ifstream ifs(file_path, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) {
        VLOGE(TAG, "open model file failed! filePath[%s]", file_path.c_str());
        return false;
    }

    int file_len = ifs.tellg();
    len = file_len > len ? len : file_len;
    ifs.seekg(0, std::ios::beg);
    ifs.read((char *) buffer, len);
    ifs.close();

    return true;
}

int FileUtil::get_file_len(const std::string &file_path) {
    std::ifstream ifs(file_path, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) {
        VLOGE(TAG, "open model file failed! filePath[%s]", file_path.c_str());
        return 0;
    }
    auto len = ifs.tellg();
    ifs.close();
    return len;
}

int FileUtil::writeToFileAppend(const string &filePath, const std::string content) {
    std::ofstream OsWrite(filePath, std::ofstream::app);
    OsWrite << content;
    OsWrite << std::endl;
    OsWrite.close();
    return 0;
}

} // namespace aura::vision