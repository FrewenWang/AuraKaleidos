//
// Created by frewen on 1/4/23.
//
#include "aura/aura_utils/utils/FileUtil.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <sys/stat.h>
#include <sys/file.h>
#include <sys/types.h>
#include <unistd.h>

namespace aura::utils {

typedef struct stat Stat_t;

bool FileUtil::checkFileExists(const std::string &fileName) {
    Stat_t sb;
    // 获取 FILE 的文件属性并将它们放入 BUF。
    if (stat(fileName.c_str(), &sb) == -1) {
        return false;
    }
    return true;
}


bool FileUtil::deleteFile(const std::string &fileName) {
    return (remove(fileName.c_str()) == 0);
}

bool FileUtil::moveFile(const std::string &currentName, const std::string &newName, bool overwrite) {
    // 如果允许复写，则避免原有文件存在，则删除原有文件
    if (overwrite) {
        remove(newName.c_str());
    }
    return (rename(currentName.c_str(), newName.c_str()) == 0);
}

bool FileUtil::checkIsDir(const std::string &fileName) {
    bool retVal = false;
    Stat_t sb;
    if (stat(fileName.c_str(), &sb) == 0) {
        if (sb.st_mode & S_IFDIR) {
            retVal = true;
        }
    }
    return retVal;
}

bool FileUtil::copyOverFile(const std::string &fromFile, const std::string &toFile) {
    bool rc = false;
    int readFd;
    int writeFd;
    struct stat statBuf;
    // Open the input file.
    readFd = ::open(fromFile.c_str(), O_RDONLY);
    if (readFd == -1) {
        close(readFd);
        return false;
    }

    // Stat the input file to obtain its size. */
    if (fstat(readFd, &statBuf) != 0) {
        close(readFd);
        return false;
    }

    // Open the output file for writing, with the same permissions as the input
    writeFd = ::open(toFile.c_str(), O_WRONLY | O_CREAT | O_TRUNC, statBuf.st_mode);
    if (writeFd == -1) {
        close(readFd);
        return false;
    }

    // Copy the file in a non-kernel specific way */
    char fileBuf[8192];
    ssize_t rBytes, wBytes;
    while (true) {
        rBytes = read(readFd, fileBuf, sizeof(fileBuf));

        if (!rBytes) {
            rc = true;
            break;
        }

        if (rBytes < 0) {
            rc = false;
            break;
        }

        wBytes = write(writeFd, fileBuf, (size_t) rBytes);

        if (!wBytes) {
            rc = true;
            break;
        }

        if (wBytes < 0) {
            rc = false;
            break;
        }
    }

    /* Close up. */
    close(readFd);
    close(writeFd);
    return rc;
}

int FileUtil::readFile(const std::string &file_path, std::string &content) {
    std::ifstream ifs(file_path);
    std::stringstream ss;
    int err = 0;
    if (ifs.is_open()) {
        std::string line;
        while (std::getline(ifs, line)) {
            ss << line << std::endl;
        }
        content = ss.str();
    } else {
        err = -1;
    }
    ifs.close();
    return err;
}

int FileUtil::readFile(const std::string &file_path, std::vector<std::string> &lines) {
    std::ifstream ifs(file_path);
    lines.clear();
    int err = 0;
    if (ifs.is_open()) {
        std::string line;
        while (std::getline(ifs, line)) {
            lines.emplace_back(line);
        }
    } else {
        err = -1;
    }
    ifs.close();
    return err;
}

int FileUtil::saveFile(const std::string &file_path, const std::string &content) {
    std::ofstream ofs(file_path);
    if (!ofs.is_open()) {
        return -1;
    }
    ofs << content;
    ofs.close();
    return 0;
}

int FileUtil::readFile(const std::string &file_path, unsigned char *buffer) {
    int size = 0;
    std::ifstream file(file_path, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        file.read((char *) buffer, size);
    }
    file.close();
    return 0;
}


}