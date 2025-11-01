

#ifndef VISION_FILEUTIL_H
#define VISION_FILEUTIL_H

#include <iterator>
#include <string>
#include <vector>

namespace aura::vision {
/**
 * @brief 文件处理工具类
 * */
class FileUtil {
public:
    /**
     * @brief 读取文件信息数据
     * @param file_path 文件路径
     * @param content   输出的文件字符串信息
     * */
    static void read_file(char *file_path, std::string &content);

    static bool read_file(const std::string &file_path, void *buffer, int len);

    static bool read_file(const std::string &file_path, std::vector<char> &buffer);

    static int get_file_len(const std::string &file_path);

    /**
     * @brief 以追加的方法将内容写入到对应文件中
     * @param filePath  文件名称
     * @param content   追加内容
     * @return
     */
    static int writeToFileAppend(const std::string &filePath, const std::string content);

};

} // namespace aura::vision

#endif //VISION_FILEUTIL_H
