//
// Created by Frewen.Wong on 2023/1/15.
//
#pragma once

#include <unordered_map>
#include <string>
#include <ctime>
#include <vector>
#include <mutex>
#include <map>

#define PERF_AUTO(perf, tag) PerfAuto autoPerf(perf, tag);


namespace aura::perf_guard
{
class PerfGuard {
public:
    static const std::string TAG;

    /**
     * 使用TAG进行初始化PerfGuard
     * @param tag
     */
    explicit PerfGuard(int &tag);

    explicit PerfGuard(std::string &tag);

    static PerfGuard *gPerfGuard(int tag = -1);

    static void gPerfGuardClear(int tag = -1);

    static void gPrint(int tag = -1);

    /**
     * 设置是否启动调试模式，调试模型会进行日志打印
     * @param debugger
     * @return
     */
    bool setDebugger(bool debugger);

    bool isDebugger();

    /**
     * 开始耗时性能计时
     * @param tag
     * @return
     */
    bool tick(const std::string &tag);

    /**
     * 开始耗时性能计时
     * @param tag
     * @return
     */
    bool tick(int tag);

    bool tock(const std::string &tag);

    bool tock(int tag);

    bool clear();

    /**
     * @brief 获取记录的数据
     * @return std::map<char*, long>  返回数据指针(key=键值,value=某段代码执行的时间)
     * */
    std::vector<std::tuple<std::string, std::uint64_t, int>> &getRecords();

private:
    static std::mutex sMutexGlobal;
    static std::map<std::string, PerfGuard *> sGlobalGuard;
    /** PerfGuard的互斥锁 */
    std::mutex mMutex{};
    std::vector<std::tuple<std::string, std::uint64_t, int>> mRecords{};
};

class PerfAuto {
public:
    PerfAuto(PerfGuard *perf, const std::string &tag);

    explicit PerfAuto(double &duration);

    ~PerfAuto();

private:
    PerfGuard *perfGuard;
    std::string tag = "PerfAuto";

    std::int64_t duration;
    /**
     * https://blog.csdn.net/zx3517288/article/details/50553965
     * clock_t是一个长整形数。
     * 在time.h文件中，还定义了一个常量CLOCKS_PER_SEC，它用来表示一秒钟会有多少个时钟计时单元，其定义如下：
     * #define CLOCKS_PER_SEC ((clock_t)1000)
     * clock()返回单位是毫秒。如果想返用秒为单位可以用
     */
    // clock_t start{};

    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};
}
