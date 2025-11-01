//
// Created by weixuechao on 23-4-7.
//
#if ENABLE_PERF
#pragma once

#include "vision/core/request/VisionRequest.h"

#include <iostream>
#include <fstream>

namespace aura::vision {

class CsvFile {
public:
    CsvFile(int source) {
        mFileName = "/data/frame_rate_" + std::to_string(source) + ".csv";
        mOutFile.open(mFileName, std::ios::out | std::ios_base::app);
        if (!mOutFile.good() || mOutFile.fail()) {
            std::cout << "the 3 ways think open file failed" << std::endl;
        }

        if (!mOutFile.is_open()) {
            std::cout << "is_open think open file failed" << std::endl;
        }
    }

    ~CsvFile() {
        if (mOutFile.is_open()) {
            mOutFile.close();
        }
    }

    bool writeFile(int source, uint64_t time, float rate) {
        if (!mOutFile.is_open()) {
            mOutFile.open(mFileName, std::ios::out | std::ios_base::app);
        }
        if (!mOutFile.is_open() || !mOutFile.good() || mOutFile.fail()) {
            return false;
        }
        mOutFile << source << ',' << time << ',' << rate << std::endl;
        return true;
    }

private:
    std::string mFileName = "";
    // 写文件
    std::ofstream mOutFile;
};

class WriteCsvFile {
public:
    bool write(int source, uint64_t time, float rate) {
        if (source == Source::SOURCE_1) {
            sWriteCdcCsvFile1.writeFile(source, time, rate);
        } else if (source == Source::SOURCE_2) {
            sWriteCdcCsvFile2.writeFile(source, time, rate);
        }
        return true;
    }

private:
    CsvFile sWriteCdcCsvFile1 = {1};
    CsvFile sWriteCdcCsvFile2 = {2};
};

}
#endif
