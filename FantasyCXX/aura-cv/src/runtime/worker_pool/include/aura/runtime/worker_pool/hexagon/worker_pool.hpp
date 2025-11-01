#ifndef AURA_RUNTIME_WORKER_POOL_HEXAGON_WORKER_POOL_HPP__
#define AURA_RUNTIME_WORKER_POOL_HEXAGON_WORKER_POOL_HPP__

/**
 * @cond AURA_BUILD_HEXAGON
*/

#include "aura/runtime/context.h"
#include "aura/runtime/logger.h"

#include <queue>
#include <thread>
#include <mutex>
#include <map>
#include <memory>
#include <future>

#define AURA_DEFAULT_THREAD_STACK_SZ                    (32 * 1024) // 32KB

/**
 * @defgroup runtime Runtime
 * @{
 *      @defgroup worker Worker Pool
 *      @{
 *          @defgroup worker_pool_hexagon Worker Pool Hexagon
 *      @}
 * @}
 */

namespace aura
{

/**
 * @addtogroup worker_pool_hexagon
 * @{
 */

/**
 * @brief Waits for a single token to complete.
 *
 * @tparam Tp Type of the token.
 * @param token The token to wait for.
 */
template<typename Tp>
AURA_VOID WaitTokens(Tp &&token)
{
    token.wait();
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
 * @brief Waits for a vector of tokens to complete.
 *
 * @tparam Tp Type of the tokens.
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
 * @brief Function to be executed by each thread.
 *
 * @param instance A pointer to the WorkerPool instance.
 */
AURA_VOID ThreadRun(AURA_VOID *instance);

/**
 * @brief Class for managing a pool of worker threads on Qualcomm CDSP platform.
 *
 * The WorkerPool class provides functionality for parallel execution of tasks.
 * It allows users to run tasks in parallel and stop the pool.
 */
class AURA_EXPORTS WorkerPool
{
public:
    /**
     * @brief Constructor for WorkerPool to create a worker threads pool based on specific parameters.
     *
     * @param ctx The pointer to the Context object.
     * @param stack_sz Size of the thread stack.
     * @param tag The tag for identifying the worker pool.
     */
    WorkerPool(Context *ctx, MI_S32 stack_sz = AURA_DEFAULT_THREAD_STACK_SZ, const std::string &tag = AURA_TAG);

    /**
     * @brief Destructor for WorkerPool to terminates all threads and stops the worker pool.
     */
    ~WorkerPool();

    /**
     * @brief Disable copy and assignment constructor operations.
     */
    AURA_DISABLE_COPY_AND_ASSIGN(WorkerPool);

    /**
     * @brief Adds a task to the worker pool.
     *
     * @tparam FuncType Type of the function to be executed.
     * @tparam ArgsType Types of the arguments for the function.
     *
     * @param f The function to be executed.
     * @param args The arguments for the function.
     *
     * @return std::future containing the result of the function.
     */
    template<typename FuncType, typename ...ArgsType>
    auto AddTask(FuncType &&f, ArgsType &&...args) -> std::future<typename std::result_of<FuncType(ArgsType...)>::type>;

    /**
     * @brief Executes a parallelized task using a specified range.
     *
     * This function divides the specified range and executes the task function in parallel in the worker pool.
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
     * @param w The width  of the range.
     * @param f The task function.
     * @param args The arguments to the task function.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    template<typename RangeType, typename FuncType, typename ...ArgsType>
    Status WaveFront(RangeType h, RangeType w, FuncType &&f, ArgsType &&...args);

    /**
     * @brief Gets the number of threads in the worker pool.
     *
     * @return The number of threads.
     */
    MI_S32 GetComputeThreadNum() {return m_threads.size();}

    /**
     * @brief Send a stop signal to awaken all threads in preparation for exiting the thread callback function.
     */
    AURA_VOID Stop();

    /**
     * @brief Gets the index of the current thread within the worker pool.
     *
     * @return The index of the current thread. (Return -1 if the thread is not in the thread pool.)
     */
    MI_S32 GetComputeThreadIdx()
    {
        auto search = m_tid_map.find(qurt_thread_get_id());
        if (search != m_tid_map.end())
        {
            return search->second;
        }
        else
        {
            return -1;
        }
    }

    /**
    * @brief Retrieves a list of thread IDs.
    *
    * This function iterates over the map of thread IDs and collects the thread IDs into a vector.
    *
    * @return A vector of thread IDs representing the threads.
    */
    std::vector<qurt_thread_t> GetComputeThreadIDs()
    {
        std::vector<qurt_thread_t> ids;

        for (auto &tid : m_tid_map)
        {
            ids.push_back(tid.first);
        }

        return ids;
    }

    /**
     * @brief Function to be executed by each thread in the worker pool.
     *
     * @param instance A pointer to the WorkerPool instance.
     */
    friend AURA_VOID ThreadRun(AURA_VOID *instance);

private:
    template <typename FuncType, typename ...ArgsType>
    class WaveFrontHelper;

    Context *m_ctx;                                     /*!< The context associated with the worker pool. */
    std::atomic_bool m_stopped;                         /*!< Atomic flag to indicate whether the worker pool has been stopped. */

    std::mutex m_mutex;                                 /*!< Mutex for synchronizing access to the task queue. */
    std::condition_variable m_wait_cv;                  /*!< Condition variable for thread synchronization. */
    std::string m_tag;                                  /*!< The tag to identify the threads of the worker pool. */
    std::vector<qurt_thread_t> m_threads;               /*!< Vector of compute threads. */
    MI_U8 *m_stack;                                     /*!< Pointer to the thread stack memory. */
    std::queue<std::function<AURA_VOID()>> m_task_queue;  /*!< Queue for tasks. */
    std::map<qurt_thread_t, MI_S32> m_tid_map;          /*!< Map to store thread IDs and their corresponding thread indices. */
};

template <typename FuncType, typename ...ArgsType>
class WorkerPool::WaveFrontHelper
{
public:
    WaveFrontHelper(FuncType &&f, MI_S32 h, MI_S32 w): m_f(std::forward<FuncType>(f)), m_h(h), m_w(w)
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
                tokens.emplace_back(wp->AddTask(std::ref(*this), wp, tokens, row + 1, col, std::forward<ArgsType>(args)...));
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

template<typename RangeType, typename FuncType, typename ...ArgsType>
Status WorkerPool::ParallelFor(RangeType start, RangeType end, FuncType &&f, ArgsType &&...args)
{
    using RetType = typename std::result_of<FuncType(ArgsType..., RangeType, RangeType)>::type;

    static_assert(is_integral<RangeType>::value,
                  "parallel_for RangeType must be integral type");
    static_assert(std::is_same<RetType, Status>::value,
                  "parallel_for func's RetType must be Status");

    if (start >= end)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Invalid Range Info.");
        return Status::ERROR;
    }

    RangeType step = 0;

    if (m_threads.empty())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Invalid Threads.");
        return Status::ERROR;
    }
    else
    {
        step = (end - start + static_cast<RangeType>(m_threads.size()) - 1) / static_cast<RangeType>(m_threads.size());
    }

    std::vector<std::future<RetType>> tokens;
    std::atomic_bool task_finished(MI_FALSE);
    std::atomic_bool task_failed(MI_FALSE);

    auto functor = std::bind(std::forward<FuncType>(f), std::forward<ArgsType>(args)..., std::placeholders::_1, std::placeholders::_2);

    auto task_func = [&functor, &task_failed](MI_S32 idx_start, MI_S32 idx_end) -> Status
    {
        if (task_failed)
        {
            return Status::ERROR;
        }

        if (functor(idx_start, idx_end) != Status::OK)
        {
            task_failed = MI_TRUE;
            return Status::ERROR;
        }

        return Status::OK;
    };

    auto last_task_func = [&functor, &task_failed, &task_finished](MI_S32 idx_start, MI_S32 idx_end) -> Status
    {
        if (task_failed)
        {
            return Status::ERROR;
        }

        if (functor(idx_start, idx_end) != Status::OK)
        {
            task_failed = MI_TRUE;
            return Status::ERROR;
        }

        task_finished = MI_TRUE;
        return Status::OK;
    };

    RangeType cur = start;

    for (; cur + step < end; cur += step)
    {
        tokens.emplace_back(this->AddTask(task_func, cur, cur + step));
    }

    tokens.emplace_back(this->AddTask(last_task_func, cur, end));

    WaitTokens(tokens);

    if (task_failed || !task_finished)
    {
        return Status::ERROR;
    }

    return Status::OK;
}

template<typename RangeType, typename FuncType, typename ...ArgsType>
Status WorkerPool::WaveFront(RangeType h, RangeType w, FuncType &&f, ArgsType &&...args)
{
    using RetType = typename std::result_of<FuncType(ArgsType..., RangeType, RangeType)>::type;

    static_assert(std::is_integral<RangeType>::value, "parallel_for RangeType must be integral type");
    static_assert(std::is_same<RetType, Status>::value, "parallel_for func's RetType must be Status");

    if ((h <= 0) || (w <= 0))
    {
        return Status::ERROR;
    }

    // start task from (0, 0)
    std::vector<std::shared_future<RetType>> tokens;

    WaveFrontHelper<FuncType, ArgsType...> task_helper(std::forward<FuncType>(f), h, w);

    tokens.emplace_back(this->AddTask(std::ref(task_helper),
                        this, tokens, 0, 0, std::forward<ArgsType>(args)...));

    while (MI_TRUE)
    {
        std::function<AURA_VOID()> task;

        {
            std::unique_lock<std::mutex> lock(m_mutex);

            if (task_helper.m_task_finished || task_helper.m_task_failed || m_stopped)
            {
                break;
            }
            else
            {
                if (m_task_queue.empty())
                {
                    continue;
                }
                task = std::move(m_task_queue.front());
                m_task_queue.pop();
            }
        }

        task();
    }

    WaitTokens(tokens);

    if (task_helper.m_task_failed || !task_helper.m_task_finished)
    {
        return Status::ERROR;
    }

    return Status::OK;
}

template<typename FuncType, typename ...ArgsType>
auto WorkerPool::AddTask(FuncType &&f, ArgsType &&...args) ->  std::future<typename std::result_of<FuncType(ArgsType...)>::type>
{
    using RetType = typename std::result_of<FuncType(ArgsType...)>::type;

    auto task = std::make_shared<std::packaged_task<RetType()>>(std::bind(std::forward<FuncType>(f), std::forward<ArgsType>(args)...));

    std::future<RetType> result = task->get_future();

    if (!m_stopped)
    {

        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_task_queue.emplace([task]() { (*task)(); });
        }
        m_wait_cv.notify_one();

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

/**
 * @endcond
*/

#endif // AURA_RUNTIME_WORKER_POOL_HEXAGON_WORKER_POOL_HPP__