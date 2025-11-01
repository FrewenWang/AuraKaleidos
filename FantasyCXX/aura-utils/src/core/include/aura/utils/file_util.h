//
// Created by frewen on 1/4/23.
//
#pragma once

#include <iostream>
#include <vector>

namespace aura::utils
{
class FileUtil {
public:
    /**
     * 检测文件是否存在
     * @param fileName
     * @return
     */
    bool checkFileExists(const std::string &fileName);

    /**
     * 移动文件
     * @param currentName 当前文件
     * @param newName 移动之后的文件名称
     * @param overwrite 设置覆盖
     * @return
     */
    bool moveFile(const std::string &currentName, const std::string &newName, bool overwrite);

    /**
     * 删除文件
     * @param fileName
     * @return
     */
    bool deleteFile(const std::string &fileName);

    bool checkIsDir(const std::string &fileName);

    /**
     * 将文件从一个位置复制到另一个位置，如果目标已存在，则覆盖。
     * @param source
     * @param target
     * @return
     */
    static bool copyOverFile(const std::string &source, const std::string &target);

    /**
     * 读取文件内容
     * @param file_path
     * @param content
     * @return
     */
    static int readFile(const std::string &file_path, std::string &content);

    /**
     * 逐行读取文件内容
     * @param file_path
     * @param lines
     * @return
     */
    static int readFile(const std::string &file_path, std::vector<std::string> &lines);

    /**
     * 保存文本内容到Buffer中
     * @param file_path
     * @param content
     * @return
     */
    static int saveFile(const std::string &file_path, const std::string &content);

    /**
     * 读取文件内容
     * @param file_path
     * @param buffer
     * @return
     */
    static int readFile(const std::string &file_path, unsigned char *buffer);
};
}
