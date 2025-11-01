//
// Created by Frewen.Wang on 25-6-22.
//

#include "aura/utils/concurrent/promise_delay_task.h"

#include <functional>
#include <mutex>
#include <thread>

namespace aura::utils
{
PromiseDelayTask::PromiseDelayTask() : processed(false), interrupted(false) {
}

PromiseDelayTask::~PromiseDelayTask() {
    if (worker.valid()) {
        worker.wait();
    }
}

void PromiseDelayTask::RunAfterDelay(std::function<void()> task, int seconds) {
    Reset();
    worker = std::async(std::launch::async, [this, task, seconds]() {
        auto start = std::chrono::steady_clock::now();
        auto end = start + std::chrono::seconds(seconds);
        // 分段睡眠，每100毫秒检查一次中断
        while (std::chrono::steady_clock::now() < end) {
            if (interrupted) {
                return; // 被中断则退出
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        if (!interrupted) {
            processed = true;
            task(); // 未被中断则执行任务
        }
    });
}

// 重置中断状态（允许新的延迟任务）
void PromiseDelayTask::Reset() {
    interrupted = false;
    processed = false;
}
}
