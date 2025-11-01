#pragma once
#include <mutex>
#include <future>

namespace aura::utils
{
class PromiseDelayTask {
public:
    PromiseDelayTask();

    ~PromiseDelayTask();

    void RunAfterDelay(std::function<void()> task, int seconds);

    void Interrupt() {
        interrupted = true;
    }

    bool HasInterrupted() const {
        return interrupted.load();
    }

    bool HasProcessed() const {
        return processed.load();
    }

    void Reset();

private:
    std::future<void> worker;
    std::atomic<bool> interrupted;
    std::atomic<bool> processed;
};
}
