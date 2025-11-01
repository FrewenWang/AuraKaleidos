//
// Created by Frewen.Wang on 2024/6/30.
//
#pragma once

#include <functional>
#include <future>
#include <queue>

namespace aura::utils
{

class ThreadPoolV17
{
public:
    explicit ThreadPoolV17(size_t);

    template<class F, class... Args>
    decltype(auto) enqueue(F &&f, Args &&...args);

    ~ThreadPoolV17();

private:
    // need to keep track of threads so we can join them
    std::vector<std::thread> workers;
    // the task queue
    std::queue<std::packaged_task<void()>> tasks;

    // synchronization
    std::mutex queue_mutex;
    std::condition_variable condition;
    std::condition_variable condition_producers;
    bool stop;
};

} // namespace aura::utils
