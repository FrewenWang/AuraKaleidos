// #include "aura/utils/concurrent/thread_pool_v11.h"
// #include <assert.h>
// #include <iostream>
// #include <sstream>
// #include <string>

// namespace aura::utils {

// ThreadPoolV11::ThreadPoolV11(size_t threads) : stop(false) {
//     // 构造函数进行初始化的时候，我们就进行线程池的构建
//     // 创建指定线程的（threads 是线程池的大小）。
//     // size_t 是无符号整数类型，通常用于表示大小或索引。
//     for(size_t i = 0;i<threads;++i) {
//         // 向 workers 容器中添加一个线程。
//         // workers 是一个存储线程的容器（如 std::vector<std::thread>）。
//         // emplace_back 是 C++11 引入的高效方法，直接在容器中构造对象，避免额外的拷贝或移动操作。
//         // 定义一个 Lambda 表达式，作为线程的执行函数。
//         // [this] 是捕获列表，表示 Lambda 表达式可以访问当前对象的成员变量。
//         // { ... } 是 Lambda 的函数体。
//         workers.emplace_back([this] {
//             for(;;) {
//                 std::function<void()> task;
//                 {
//                     std::unique_lock<std::mutex> lock(this->queue_mutex);
//                     this->condition.wait(lock,[this]{ return this->stop || !this->tasks.empty(); });

//                     ///
//                     if(this->stop && this->tasks.empty()) {
//                         return;
//                     }

//                     task = std::move(this->tasks.front());
//                     this->tasks.pop();
//                 }
//                 task();
//             }
//         });
//     }

// }

// template<class F, class... Args>
// auto ThreadPoolV11::enqueue(F&& f, Args&&... args)
//     -> std::future<typename std::result_of<F(Args...)>::type> {
//     using return_type = typename std::result_of<F(Args...)>::type;

//     auto task = std::make_shared< std::packaged_task<return_type()> >(
//             std::bind(std::forward<F>(f), std::forward<Args>(args)...)
//         );

//     std::future<return_type> res = task->get_future();
//     {
//         std::unique_lock<std::mutex> lock(queue_mutex);

//         // don't allow enqueueing after stopping the pool
//         if(stop)
//             throw std::runtime_error("enqueue on stopped ThreadPool");

//         tasks.emplace([task](){ (*task)(); });
//     }
//     condition.notify_one();
//     return res;
// }

// ThreadPoolV11::~ThreadPoolV11() {
//     {
//         std::unique_lock<std::mutex> lock(queue_mutex);
//         stop = true;
//     }
//     condition.notify_all();
//     for(std::thread &worker: workers) {
//         worker.join();
//     }
// }

// }