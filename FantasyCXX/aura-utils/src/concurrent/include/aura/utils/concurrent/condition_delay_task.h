#pragma once
#include <mutex>

namespace aura::utils
{
class ConditionDelayTask {
public:
    ConditionDelayTask();

    ~ConditionDelayTask();

    void RunAfterDelay(std::function<void()> task, int seconds);

    void Interrupt();

    bool HasInterrupted() const {
        return interrupted.load();
    }

    bool HasProcessed() const {
        return processed.load();
    }

    void Reset();

private:
    std::mutex mutex;
    std::condition_variable condition;
    std::atomic<bool> interrupted;
    std::atomic<bool> processed;
};
}
