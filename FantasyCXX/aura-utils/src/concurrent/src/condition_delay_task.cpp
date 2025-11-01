//
// Created by Frewen.Wang on 25-6-22.
//

#include "aura/utils/concurrent/condition_delay_task.h"

#include <functional>
#include <mutex>
#include <thread>

namespace aura::utils
{
ConditionDelayTask::ConditionDelayTask() : interrupted(false) {
}

ConditionDelayTask::~ConditionDelayTask() {
}

void ConditionDelayTask::RunAfterDelay(std::function<void()> task, int seconds) {
  std::thread([this, task, seconds]() {
    std::unique_lock<std::mutex> lock(mutex);
    // 等待5秒或中断信号
    auto result = condition.wait_for(lock, std::chrono::seconds(seconds),
                                     [this] {
                                       return interrupted.load();
                                     });
    if (!result) {
      task(); // 执行任务
    }
  }).detach(); // 分离线程，任务在后台运行
}

// 中断所有延迟任务
void ConditionDelayTask::Interrupt() {
  interrupted = true;
  condition.notify_all(); // 唤醒所有等待的线程
}

// 重置中断状态（允许新的延迟任务）
void ConditionDelayTask::Reset() {
  interrupted = false;
}
}
