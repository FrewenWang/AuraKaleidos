#include "aura/utils/concurrent/thread_pool_v17.h"
#include <assert.h>
#include <iostream>
#include <sstream>

namespace aura::utils {
// the constructor just launches some amount of workers
inline ThreadPoolV17::ThreadPoolV17(size_t threads) :stop(false) {
    for(size_t i = 0;i<threads;++i)
        workers.emplace_back(
            [this]
            {
                for(;;) {
                    std::packaged_task<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock,
                            [this]{ return this->stop || !this->tasks.empty(); });
                        if(this->stop && this->tasks.empty())
                            return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                        if (tasks.empty()) {
                            condition_producers.notify_one(); // notify the destructor that the queue is empty
                        }
                    }

                    task();
                }
            }
        );
}

// add new work item to the pool
template<class F, class... Args>
decltype(auto) ThreadPoolV17::enqueue(F&& f, Args&&... args) {
    using return_type = std::invoke_result_t<F, Args...>;

    std::packaged_task<return_type()> task(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

    std::future<return_type> res = task.get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);

        // don't allow enqueueing after stopping the pool
        if(stop)
            throw std::runtime_error("enqueue on stopped ThreadPoolV17");

        tasks.emplace(std::move(task));
    }
    condition.notify_one();
    return res;
}

inline ThreadPoolV17::~ThreadPoolV17() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        condition_producers.wait(lock, [this] { return tasks.empty(); });
        stop = true;
    }
    condition.notify_all();
    for (std::thread& worker : workers) {
        worker.join();
    }
}

}