// //
// // Created by Frewen.Wang on 2024/6/30.
// //
// #pragma once

// #include <condition_variable>
// #include <functional>
// #include <future>
// #include <memory>
// #include <mutex>
// #include <queue>
// #include <stdexcept>
// #include <thread>
// #include <vector>
// #include <string>

// namespace aura::utils
// {
// /**
//  * 基于C++11的来实现的线程池
//  */
// class ThreadPoolV11
// {
// public:
//     ThreadPoolV11(size_t size);

//     template<class F, class... Args>
//     auto enqueue(F &&f, Args &&...args) -> std::future<typename std::result_of<F(Args...)>::type>;

//     ~ThreadPoolV11();

// private:
//     // 创建一个workers线程向量，保存线程池里面的线程对象
//     std::vector<std::thread> workers;
//     // the task queue  线程池的任务队列
//     std::queue<std::function<void()>> tasks;

//     // synchronization
//     std::mutex queue_mutex;
//     std::condition_variable condition;
//     bool stop;
// };
