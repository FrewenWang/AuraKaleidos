
#pragma once

#include <mutex>
#include <condition_variable>

namespace aura::vision {

class CountLatch {
public:
    CountLatch();

    ~CountLatch() = default;

    void wait();

    void countUp();

    void countDown();

    int getCount() const;

private:
    volatile int counter;
    mutable std::mutex mutex;  //getCount中mutex状态会有修改，因此加mutable关键字
    std::condition_variable condition;
};

} // namespace aura::vision