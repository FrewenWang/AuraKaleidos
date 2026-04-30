#pragma once

#include <fstream> //处理文件的头文件
#include <iostream>
#include <sstream>
#include <ostream>

#include <string>
#include <vector>

class IfstreamSimpleUnit {
public:
    explicit IfstreamSimpleUnit(std::string &filepath) {
        path = filepath;
        file.open(filepath, std::ios::binary);
    }

    ~IfstreamSimpleUnit() {
        if(file.is_open()) {
            file.close();
        }
    }

    bool read(unsigned char *buffer, int need_read_size = -1) {
        if(file.is_open() == false) {
            file.open(path);
        }

        if (file.bad() || file.is_open() == false) {
            return false;
        }

        int size = getFileSize();
        if(need_read_size > 0)
        if(size > need_read_size) {
            size = need_read_size;
        }

        file.read((char *)buffer, size);

        return true;
    }

    int getFileSize() {
        file.seekg(0, file.end);
        int size = file.tellg();
        file.seekg(0, file.beg);
        return size;
    }

private:
    std::ifstream file;
    std::string path;
};

class fileHelper
{
public:
    fileHelper();

    static bool readFile(std::string &filepath, unsigned char *buffer);
    static bool getDirFile(std::string &folder_path, std::vector<std::string> &file_vector);
};
