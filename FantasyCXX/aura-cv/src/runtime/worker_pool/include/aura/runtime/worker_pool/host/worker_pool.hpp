#ifndef AURA_RUNTIME_WORKER_POOL_HOST_WORKER_POOL_HPP__
#define AURA_RUNTIME_WORKER_POOL_HOST_WORKER_POOL_HPP__

#include "aura/runtime/context.h"
#include "aura/runtime/logger.h"
#include "aura/runtime/cpu_info.h"

#include <queue>
#include <map>
#include <list>
#include <thread>
#include <mutex>
#include <memory>
#include <future>

/**
 * @defgroup runtime Runtime
 * @{
 *      @defgroup worker Worker Pool
 *      @{
 *          @defgroup worker_pool_host Worker Pool Host
 *      @}
 * @}
*/

namespace aura
{
/**
 * @addtogroup worker_pool_host
 * @{
*/

/**
 * @brief Waits for a single token to complete.
 *
 * @tparam Tp Type of the token.
 *
 * @param token The token to wait for.
 */
template<typename Tp>
AURA_VOID WaitTokens(Tp &&token)
{
    if (token.valid())
    {
        token.wait();
    }
}

/**
 * @brief Waits for multiple tokens to complete using recursive variadic template.
 *
 * This function recursively waits for multiple tokens to complete.
 *
 * @tparam Tp0 Type of the first token.
 * @tparam Tp1 Type of the rest of the tokens.
 *
 * @param token The first token.
 * @param tokens The rest of the tokens.
 */
template<typename Tp0, typename ...Tp1>
AURA_VOID WaitTokens(Tp0 &&token, Tp1 &&...tokens)
{
    WaitTokens(token);
    WaitTokens(tokens...);
}

/**
 * @brief Waits for multiple tokens in a vector to complete.
 *
 * @tparam Tp Type of the tokens.
 *
 * @param tokens The vector of tokens to wait for.
 */
template<typename Tp>
AURA_VOID WaitTokens(std::vector<Tp> &tokens)
{
    for (auto &token : tokens)
    {
        WaitTokens(token);
    }
}

/**
 * @brief Set current thread to bind big cpu or littler cpu or not bind
 * 
 * @param affinity The cpu bind flag
 *
 * @return SetCpuAffinity status.
 */
AURA_EXPORTS Status SetCpuAffinity(CpuAffinity affinity);

/**
 * @brief Class for managing a pool of worker threads.
 *
 * The WorkerPool class provides functionality for asynchronous and parallel execution of tasks.
 * It allows users to run tasks asynchronously and in parallel, manage thread affinities, and stop the pool.
 */
class AURA_EXPORTS WorkerPool
{
public:
    /**
     * @brief Constructor for WorkerPool to create a worker threads pool based on specific parameters.
     *
     * @param ctx The pointer to the Context object.
     * @param tag The tag to identify the worker pool (optional).
     * @param compute_affinity CPU affinity for compute threads.
     * @param async_affinity CPU affinity for asynchronous threads.
     * @param compute_threads Number of threads for computing task in the pool.
     * @param async_threads Number of threads for asynchronous task in the pool.
     */
    WorkerPool(Context *ctx, const std::string &tag = AURA_TAG, CpuAffinity compute_affinity = CpuAffinity::ALL,
               CpuAffinity async_affinity = CpuAffinity::ALL, MI_S32 compute_threads = 0, MI_S32 async_threads = 0);

    /**
     * @brief Destructor for WorkerPool to terminates all threads and stops the worker pool.
     */
    ~WorkerPool();

    /**
     * @brief Disable copy and assignment constructor operations.
     */
    AURA_DISABLE_COPY_AND_ASSIGN(WorkerPool);

    /**
     * @brief Asynchronously runs a task and returns a future representing the result.
     *
     * @tparam FuncType Type of the task function.
     * @tparam ArgsType Types of the arguments to the task function.
     *
     * @param f The task function.
     * @param args The arguments to the task function.
     *
     * @return std::future representing the result of the task.
     */
    template<typename FuncType, typename ...ArgsType>
    auto AsyncRun(FuncType &&f, ArgsType &&...args) -> std::future<typename std::result_of<FuncType(ArgsType...)>::type>;

    /**
     * @brief Executes a parallelized task using a specified range.
     *
     * This function divides the specified range and executes the task function in parallel using
     * computing threads in the worker pool.
     *
     * @tparam RangeType Type of the range.
     * @tparam FuncType Type of the task function.
     * @tparam ArgsType Types of the arguments to the task function.
     *
     * @param start The start of the range.
     * @param end The end of the range.
     * @param f The task function.
     * @param args The arguments to the task function.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    template<typename RangeType, typename FuncType, typename ...ArgsType>
    Status ParallelFor(RangeType start, RangeType end, FuncType &&f, ArgsType &&...args);

    /**
     * @brief Executes a parallelized task using a specified range when there are data dependencies.
     *
     * Task will start from (0, 0) and continue to the right and bottom.
     * Task(i, j) will be executed only when Task(i-1, j) and Task(i, j-1) are done.
     * Generally, the task will be executed in a WaveFront manner. (From top-left to bottom-right).
     *
     * @tparam RangeType Type of the range.
     * @tparam FuncType Type of the task function.
     * @tparam ArgsType Types of the arguments to the task function.
     *
     * @param h The height of the range.
     * @param w The width of the range.
     * @param f The task function.
     * @param args The arguments to the task function.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    template<typename RangeType, typename FuncType, typename ...ArgsType>
    Status WaveFront(RangeType h, RangeType w, FuncType &&f, ArgsType &&...args);

    /**
     * @brief Gets the number of computing threads in the worker pool.
     *
     * @return The number of compute threads.
     */
    MI_S32 GetComputeThreadNum() {return m_compute_threads.size() + 1;}

    /**
     * @brief Gets the number of asynchronous threads in the worker pool.
     *
     * @return The number of asynchronous threads.
     */
    MI_S32 GetAsyncThreadNum() {return m_async_threads.size();}

    /**
     * @brief Gets the index of the current thread in the compute thread pool.
     *
     * @return The index of the current thread in the compute thread pool.
     */
    MI_S32 GetComputeThreadIdx()
    {
        auto search = m_compute_tid_map.find(std::this_thread::get_id());
        if (search != m_compute_tid_map.end())
        {
            return search->second; // return matched thread idx
        }
        else
        {
            return 0; // return main thread idx (main thread can be changeable)
        }
    }

    /**
    * @brief Retrieves a list of compute thread IDs.
    *
    * This function iterates over the map of compute thread IDs and collects the thread IDs into a vector.
    *
    * @return A vector of thread IDs representing the compute threads.
    */
#if !defined(AURA_BUILD_XPLORER)
    std::vector<std::thread::id> GetComputeThreadIDs()
    {
        std::vector<std::thread::id> ids;

        for (auto &tid : m_compute_tid_map)
        {
            ids.push_back(tid.first);
        }

        ids.push_back(std::this_thread::get_id());
        return ids;
    }
#else  // AURA_BUILD_XPLORER
    std::vector<MI_S32> GetComputeThreadIDs()
    {
        std::vector<MI_S32> ids;

        ids.push_back(0);
        return ids;
    }
#endif // AURA_BUILD_XPLORER

    /**
     * @brief Gets the index of the current thread in the asynchronous thread pool.
     *
     * @return The index of the current thread in the asynchronous thread pool.
     */
    MI_S32 GetAsyncThreadIdx()
    {
        auto search = m_async_tid_map.find(std::this_thread::get_id());
        if (search != m_async_tid_map.end())
        {
            return search->second; // return matched thread idx
        }
        else
        {
            return 0; // return main thread idx (main thread can be changeable)
        }
    }

    /**
     * @brief Send a stop signal to awaken all threads in preparation for exiting the thread callback function.
     */
    AURA_VOID Stop();

private:
    /**
     * @brief Set the name of the thread.
     *
     * @param tag The tag of name.
     * @param type The label to distinguish synchronous threads and asynchronous threads.
     * @param idx The index of the thread.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
    */
    Status SetThreadName(const std::string &tag, MI_CHAR type, MI_S32 idx);

    /**
     * @brief Function to be executed by asynchronous threads.
     *
     * @param async_affinity CPU affinity for asynchronous threads.
     */
    AURA_VOID AsyncThreadRun(CpuAffinity async_affinity);

    /**
     * @brief Function to be executed by compute threads.
     *
     * @param compute_affinity CPU affinity for compute threads.
     */
    AURA_VOID ComputeThreadRun(CpuAffinity compute_affinity);

    /**
     * @brief Check if compute threads are available.
     */
    AURA_VOID CheckComputeThreads();

    /**
     * @brief Check if asynchronous threads are available.
     */
    AURA_VOID CheckAsyncThreads();

public:
    /**
     * @brief Type representing the different types of tasks in the worker pool.
     */
    enum class TaskType
    {
        ASYNC_TASK   = 0,   /*!< Asynchronous task type. */
        COMPUTE_TASK = 1,   /*!< Compute task type. */
    };

    /**
     * @brief Adds a task to the appropriate queue based on the task type.
     *
     * This function creates a packaged task from the specified task function and arguments
     * and adds it to either the asynchronous or compute task queue.
     *
     * @tparam FuncType Type of the task function.
     * @tparam ArgsType Types of the arguments to the task function.
     *
     * @param type The type of the task (ASYNC_TASK or COMPUTE_TASK).
     * @param f The task function.
     * @param args The arguments to the task function.
     *
     * @return std::future representing the result of the task.
     */
    template<typename FuncType, typename ...ArgsType>
    auto AddTask(const TaskType type, FuncType &&f, ArgsType &&...args) -> std::future<typename std::result_of<FuncType(ArgsType...)>::type>;

private:

    struct ThreadTask;

    class AtomicQueue;

    template <typename FuncType, typename ...ArgsType>
    class WaveFrontHelper;

    Context                          *m_ctx;                  /*!< The context associated with the worker pool. */
    std::string                       m_tag;                  /*!< The tag to identify the threads of the worker pool. */
    CpuAffinity                       m_compute_affinity;     /*!< CPU affinity for compute threads. */
    CpuAffinity                       m_async_affinity;       /*!< CPU affinity for asynchronous threads. */
    std::atomic_bool                  m_stopped;              /*!< Atomic flag to indicate whether the worker pool has been stopped. */
    std::atomic_size_t                m_async_running_count;  /*!< Atomic counter for the number of running asynchronous tasks. */
    MI_S32                            m_max_compute_threads;  /*!< Maximum number of compute threads. */
    MI_S32                            m_max_async_threads;    /*!< Maximum number of asynchronous threads. */

    std::mutex                        m_async_mutex;          /*!< Mutex for synchronizing access to the asynchronous task queue. */
    std::atomic<MI_S32>               m_async_thread_idx;     /*!< Atomic index for tracking asynchronous thread indices. */
    std::condition_variable           m_async_wait_cv;        /*!< Condition variable for asynchronous thread synchronization. */
    std::vector<std::thread>          m_async_threads;        /*!< Vector of asynchronous threads. */
    std::map<std::thread::id, MI_S32> m_async_tid_map;        /*!< Map to store thread IDs and their corresponding asynchronous thread indices. */

    std::mutex                        m_compute_mutex;        /*!< Mutex for synchronizing access to the compute task queue. */
    std::atomic<MI_S32>               m_compute_thread_idx;   /*!< Atomic index for tracking compute thread indices. */
    std::condition_variable           m_compute_wait_cv;      /*!< Condition variable for compute thread synchronization. */
    std::vector<std::thread>          m_compute_threads;      /*!< Vector of compute threads. */
    std::map<std::thread::id, MI_S32> m_compute_tid_map;      /*!< Map to store thread IDs and their corresponding compute thread indices. */

    std::list<ThreadTask>             m_async_task_list;      /*!< List for asynchronous tasks. */
    std::list<ThreadTask>             m_compute_task_list;    /*!< List for compute tasks. */
};

struct WorkerPool::ThreadTask
{
    ThreadTask(std::function<AURA_VOID()> &&f) : func(std::move(f))
    {
        tid = std::this_thread::get_id();
    }

    ThreadTask(const std::thread::id &thread_id, std::function<AURA_VOID(AURA_VOID)> &&f) :
               tid(thread_id), func(std::move(f))
    {}

    std::thread::id          tid;
    std::function<AURA_VOID()> func;
};

class WorkerPool::AtomicQueue
{
public:
    AtomicQueue(MI_S32 head, MI_S32 tail): m_head_idx(head), m_tail_idx(tail) {};

    MI_BOOL Pop(MI_S32 &start_row, MI_S32 &end_row)
    {
        MI_S32 head = m_head_idx.fetch_add(1);

        if (head >= m_tail_idx.load())
        {
            return MI_FALSE;
        }

        start_row = head;
        end_row   = Min(head + 1, m_tail_idx.load());
        return MI_TRUE;
    }

    MI_BOOL PopChunk(MI_S32 &start_row, MI_S32 &end_row)
    {
        std::lock_guard<std::mutex> lock(m_mutex);

        MI_S32 tail = m_tail_idx.load();
        MI_S32 head = m_head_idx.load();
        MI_S32 steal_num = ((tail - head) + 1) / 2;
        if (steal_num <= 0)
        {
            return MI_FALSE;
        }

        head = m_head_idx.fetch_add(steal_num);
        if (head >= tail)
        {
            return MI_FALSE;
        }

        start_row = head;
        end_row   = Min(head + steal_num, tail);
        return MI_TRUE;
    }

    MI_BOOL IsEmpty()
    {
        return m_head_idx.load() >= m_tail_idx.load();
    }

public:
    alignas(128) std::atomic<MI_S32> m_head_idx;
    alignas(128) std::atomic<MI_S32> m_tail_idx;
    std::mutex m_mutex;
};

template <typename FuncType, typename ...ArgsType>
class WorkerPool::WaveFrontHelper
{
public:
    WaveFrontHelper(FuncType &&f, MI_S32 h, MI_S32 w) : m_f(std::forward<FuncType>(f)), m_h(h), m_w(w)
    {
        if ((w > 0) && (h > 0))
        {
            m_counters.resize(h * w);
            for (MI_S32 row = 0; row < h; row++)
            {
                for (MI_S32 col = 0; col < w; col++)
                {
                    m_counters[row * w + col] = static_cast<MI_U8>(row > 0) + 
                                                static_cast<MI_U8>(col > 0);
                }
            }
        }

        m_task_finished = MI_FALSE;
        m_task_failed   = MI_FALSE;
    }

    Status operator()(WorkerPool *wp, std::vector<std::shared_future<Status>> &tokens,
                      MI_S32 row, MI_S32 col, ArgsType &&...args)
    {
        if ((m_task_failed) || (row > m_h - 1) || (col > m_w - 1))
        {
            return Status::ERROR;
        }

        Status ret = m_f(std::forward<ArgsType>(args)..., row, col);
        if (ret != Status::OK)
        {
            m_task_failed = MI_TRUE;
            return ret;
        }

        while (MI_TRUE)
        {
            MI_BOOL to_east  = MI_FALSE;
            MI_BOOL to_south = MI_FALSE;

            {
                std::unique_lock<std::mutex> lock(m_mutex);
                to_south = ((row + 1 < m_h) && (--m_counters[(row + 1) * m_w + col] == 0));
                to_east  = ((col + 1 < m_w) && (--m_counters[row * m_w + col + 1] == 0));
            }

            if (to_south && to_east)
            {
                tokens.emplace_back(wp->AddTask(WorkerPool::TaskType::COMPUTE_TASK, std::ref(*this),
                                                wp, tokens, row + 1, col, std::forward<ArgsType>(args)...));
                ret = m_f(std::forward<ArgsType>(args)..., row, col + 1);
                if (ret != Status::OK)
                {
                    m_task_failed = MI_TRUE;
                    return ret;
                }

                col += 1;
            }
            else if (to_south)
            {
                ret = m_f(std::forward<ArgsType>(args)..., row + 1, col);
                if (ret != Status::OK)
                {
                    m_task_failed = MI_TRUE;
                    return ret;
                }

                row += 1;
            }
            else if (to_east)
            {
                ret = m_f(std::forward<ArgsType>(args)..., row, col + 1);
                if (ret != Status::OK)
                {
                    m_task_failed = MI_TRUE;
                    return ret;
                }

                col += 1;
            }
            else
            {
                break;
            }
        }

        // last task done
        if ((row == (m_h - 1)) && (col == (m_w - 1)))
        {
            m_task_finished = MI_TRUE;
        }

        return ret;
    }

public:
    MI_BOOL m_task_finished;
    MI_BOOL m_task_failed;

private:
    FuncType           m_f;
    MI_S32             m_h;
    MI_S32             m_w;
    std::vector<MI_U8> m_counters;
    std::mutex         m_mutex;
};

template<typename FuncType, typename ...ArgsType>
auto WorkerPool::AsyncRun(FuncType &&f, ArgsType &&...args) -> std::future<typename std::result_of<FuncType(ArgsType...)>::type>
{
    return AddTask(TaskType::ASYNC_TASK, std::forward<FuncType>(f), std::forward<ArgsType>(args)...);
}

template<typename RangeType, typename FuncType, typename ...ArgsType>
Status WorkerPool::ParallelFor(RangeType start, RangeType end, FuncType &&f, ArgsType &&...args)
{
    using RetType = typename std::result_of<FuncType(ArgsType..., RangeType, RangeType)>::type;

    static_assert(std::is_integral<RangeType>::value,
                  "parallel_for RangeType must be integral type");
    static_assert(std::is_same<RetType, Status>::value,
                  "parallel_for func's RetType must be Status");

    if (start >= end)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Invalid Range Info.");
        return Status::ERROR;
    }

    if (start + 1 == end)
    {
        return f(std::forward<ArgsType>(args)..., start, end);
    }

    std::vector<std::shared_future<RetType>> tokens;
    std::atomic_bool task_failed(MI_FALSE);

    auto functor = std::bind(std::forward<FuncType>(f), std::forward<ArgsType>(args)..., std::placeholders::_1, std::placeholders::_2);

    auto task_func = [&](std::vector<std::shared_ptr<AtomicQueue>> &wp_queue, MI_S32 thread_idx, MI_S32 thread_num) -> Status
    {
        Status ret = Status::OK;
        MI_S32 start_row, end_row;
        while (MI_TRUE)
        {
            while(!wp_queue[thread_idx]->IsEmpty())
            {
                while (wp_queue[thread_idx]->Pop(start_row, end_row)) // pop one row at a time
                {
                    ret |= functor(start_row, end_row);
                }
            }
            for (MI_S32 i = 0; i < thread_num; i++)
            {
                MI_S32 steal_thread = (thread_idx + i + 1) % thread_num;
                if (!wp_queue[steal_thread]->IsEmpty())
                {
                    if (wp_queue[steal_thread]->PopChunk(start_row, end_row)) //task stealing
                    {
                        std::lock_guard<std::mutex> lock(wp_queue[thread_idx]->m_mutex);

                        wp_queue[thread_idx]->m_head_idx.store(start_row);
                        wp_queue[thread_idx]->m_tail_idx.store(end_row);
                        break;
                    }
                }
            }

            if (wp_queue[thread_idx]->IsEmpty())
            {
                break;
            }
        }

        if (ret != Status::OK)
        {
            task_failed = MI_TRUE;
        }
        return ret;
    };

    MI_S32 thread_num = GetComputeThreadNum();
    RangeType cur = start;
    MI_S32 task_num_per_thread = ((end - start) + thread_num - 1) / thread_num;

    std::vector<std::shared_ptr<AtomicQueue>> wp_queue;
    for (MI_S32 thread_idx = 0; thread_idx < thread_num; thread_idx++)
    {
        MI_S32 head = cur;
        cur += task_num_per_thread;
        cur = (cur > end) ? end : cur;
        MI_S32 tail = cur;
        wp_queue.emplace_back(std::make_shared<AtomicQueue>(head, tail));
    }

    for (MI_S32 thread_idx = 0; thread_idx < thread_num; thread_idx++)
    {
        tokens.emplace_back(this->AddTask(TaskType::COMPUTE_TASK, task_func, wp_queue, thread_idx, thread_num));
    }

    while (MI_TRUE)
    {
        std::function<AURA_VOID()> task;

        {
            std::unique_lock<std::mutex> lock(m_compute_mutex);

            if (m_stopped || m_compute_task_list.empty())
            {
                break;
            }
            else
            {
                const std::thread::id tid = std::this_thread::get_id();
                for (auto iter = m_compute_task_list.begin(); iter != m_compute_task_list.end(); ++iter)
                {
                    if (tid == iter->tid)
                    {
                        task = std::move(iter->func);
                        m_compute_task_list.erase(iter);
                        break;
                    }
                }
            }
        }

        if (task)
        {
            task();
        }
        else
        {
            break;
        }
    
    }

    WaitTokens(tokens);

    if (task_failed)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Task failed.");
        return Status::ERROR;
    }
    else
    {
        return Status::OK;
    }
}

template<typename RangeType, typename FuncType, typename ...ArgsType>
Status WorkerPool::WaveFront(RangeType h, RangeType w, FuncType &&f, ArgsType &&...args)
{
    using RetType = typename std::result_of<FuncType(ArgsType..., RangeType, RangeType)>::type;

    static_assert(std::is_integral<RangeType>::value,
                  "parallel_for RangeType must be integral type");
    static_assert(std::is_same<RetType, Status>::value,
                  "parallel_for func's RetType must be Status");

    if ((h <= 0) || (w <= 0))
    {
        return Status::ERROR;
    }

    // start task from (0, 0)
    std::vector<std::shared_future<RetType>> tokens;

    WaveFrontHelper<FuncType, ArgsType...> task_helper(std::forward<FuncType>(f), h, w);

    tokens.emplace_back(this->AddTask(TaskType::COMPUTE_TASK, std::ref(task_helper),
                        this, tokens, 0, 0, std::forward<ArgsType>(args)...));

    while (MI_TRUE)
    {
        std::function<AURA_VOID()> task;

        {
            std::unique_lock<std::mutex> lock(m_compute_mutex);

            if (task_helper.m_task_finished || task_helper.m_task_failed || m_stopped)
            {
                break;
            }
            else
            {
                const std::thread::id tid = std::this_thread::get_id();

                for (auto iter = m_compute_task_list.begin(); iter != m_compute_task_list.end(); ++iter)
                {
                    if (tid == iter->tid)
                    {
                        task = std::move(iter->func);
                        m_compute_task_list.erase(iter);
                        break;
                    }
                }
            }
        }

        if (task)
        {
            task();
        }
    }

    WaitTokens(tokens);

    if (task_helper.m_task_failed || !task_helper.m_task_finished)
    {
        return Status::ERROR;
    }
    else
    {
        return Status::OK;
    }
}

template<typename FuncType, typename ...ArgsType>
auto WorkerPool::AddTask(const TaskType type, FuncType &&f, ArgsType &&...args) -> std::future<typename std::result_of<FuncType(ArgsType...)>::type>
{
    using RetType = typename std::result_of<FuncType(ArgsType...)>::type;

    auto task = std::make_shared<std::packaged_task<RetType()>>(std::bind(std::forward<FuncType>(f), std::forward<ArgsType>(args)...));

    std::future<RetType> result = task->get_future();

    if (!m_stopped)
    {
        if (TaskType::ASYNC_TASK == type)
        {
            CheckAsyncThreads();
            {
                std::unique_lock<std::mutex> lock(m_async_mutex);
                m_async_task_list.emplace_back([task]() { (*task)(); });
            }
            m_async_wait_cv.notify_one();
        }
        else
        {
            CheckComputeThreads();
            {
                std::unique_lock<std::mutex> lock(m_compute_mutex);
                m_compute_task_list.emplace_back([task]() { (*task)(); });
            }
            m_compute_wait_cv.notify_one();
        }

        return result;
    }
    else
    {
        task.reset();
        return result;
    }
}

/**
 * @}
*/
} // namespace aura
#endif // AURA_RUNTIME_WORKER_POOL_HOST_WORKER_POOL_HPP__