#pragma once

#include <mutex>
#include <condition_variable>

namespace aura::utils
{
/**
 * 一次性的屏障，必须有count个线程到达(countdown)，才将await放行；
 **/
class CountDownLatch {
public:
    CountDownLatch();

    ~CountDownLatch() = default;

    void await();

    void countUp();

    void countDown();

    int getCount() const;

private:
    // 在 C++ 中，volatile 是一个类型修饰符，用于指示编译器：
    // 禁止优化：编译器不会对被 volatile 修饰的变量进行某些优化，确保每次访问都直接从内存读取或写入，而不是使用寄存器的缓存值。
    //
    // volatile int counter;
    std::atomic<int> counter;
    //getCount中mutex状态会有修改，因此加mutable关键字
    mutable std::mutex mutex;
    std::condition_variable condition;
};
}
