//
// Created by Li,Wendong on 2019-01-13.
//

#pragma once

#include <unordered_map>
#include <string>
#include <ctime>
#include <vector>
#include <mutex>
#include <map>

#ifdef ENABLE_PERF
#define PERF_AUTO(perf, tag) PerfAuto auto_perf(perf, tag);

#define PERF_TICK(perf, tag)    \
do {                            \
    if (perf) (perf)->tick(tag);  \
} while(0);

#define PERF_TOCK(perf, tag)    \
do {                            \
    if (perf) (perf)->tock(tag);  \
} while(0);
#else
#define PERF_AUTO(perf, tag)
#define PERF_TICK(perf, tag)
#define PERF_TOCK(perf, tag)
#endif // ENABLE_PERF

namespace aura::vision {
/**
 * @brief 计算运行时间工具类
 * */
class PerfUtil {
public:

    static const std::string TAG_TOTAL;

    static PerfUtil *global(int tag = -1);

    static void globalClear(int tag = -1);

    static void globalPrint(int tag = -1);

    explicit PerfUtil(int &tag);

    explicit PerfUtil(std::string &tag);

    void setLogging(bool log);

    bool isLogging();

    /**
     * @brief 开始测试
     * @param key
     */
    void tick(const std::string &tag);

    void tick(int tag);

    /**
     * @brief 结束测试
     * @param key
     */
    void tock(const std::string &tag);

    void tock(int tag);

    /**
     * @brief 清空记录的数据
     * */
    void clear();

    /**
     * @brief 获取记录的数据
     * @return std::map<char*, long>  返回数据指针(key=键值,value=某段代码执行的时间)
     * */
    std::vector<std::tuple<std::string, std::uint64_t, int >> &get_records();

    /**
     * @brief 根据 tag 获取性能测试结果
     * @param tag
     * @return 运行时间
     */
    std::int64_t get_record(const std::string& tag);

    std::uint64_t getTotalTime();

    void printDetectRecords();

    void printRecords();

    void setPrintLoop(int loop = 10);

    int getPrintLoop();

    static int qnnLoopModel;

private:
    static std::mutex sMutexGlobal;
    static std::map<std::string, PerfUtil *> sGlobals;

    std::vector<std::tuple<std::string, std::uint64_t, int >> mRecords{};
    std::mutex mMutex{};
    int mPrintLoop = 1;
    int mCurLoop = 0;
    std::string mTag;
    bool mIsLogging = false;
};

class PerfAuto {
public:
    PerfAuto(PerfUtil* perf, const std::string& tag);
    explicit PerfAuto(double& duration);
    ~PerfAuto();

private:
    PerfUtil* _perf;
    std::string _tag;

    double* _duration;
    clock_t _start{};
};

} // namespace vision
