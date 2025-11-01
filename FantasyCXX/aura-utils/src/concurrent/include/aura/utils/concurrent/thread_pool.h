#pragma once

#include "aura/utils/core/runnable.h"
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace aura::utils
{
/**
 * 代码参考：https://www.cnblogs.com/carsonzhu/p/16799198.html
 *
 */
class ThreadPool {
public:
    /**
     * explicit 防止隐式类型转换
     * 默认std::thread::hardware_concurrency()在新版C++标准库中是一个很有用的函数。
     *
     */
    explicit ThreadPool(size_t threads = std::thread::hardware_concurrency());

    ~ThreadPool();

    bool isStarted() const { return mIsStarted; }

    void start();

    void stop();

    void enqueueTask(Runnable *task);

    static int mInitThreadsSize;

private:
    /**
     * 将拷贝构造函数设置为私有的禁止线程池的拷贝构造
     */
    ThreadPool(const ThreadPool &) = delete;

    /**
     * 将复制运算符等号的重载设置为私有，让其不能进行线程池对象的赋值拷贝
     */
    const ThreadPool &operator=(const ThreadPool &) = delete;

    void threadLoop();

    Runnable *take();

    std::vector<std::thread> mWorkers;
    std::deque<Runnable *> mTasks;

    std::mutex mMutex;
    std::condition_variable mCondition;
    bool mIsStarted;
};
}
