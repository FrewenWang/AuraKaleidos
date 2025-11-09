#include "aura/runtime/worker_pool/host/worker_pool.hpp"
#include "aura/runtime/logger.h"

#if defined(AURA_BUILD_ANDROID)
#  include <sys/syscall.h>
#  include <unistd.h>
#  include <linux/prctl.h>
#  include <sys/prctl.h>
#endif

#include <fstream>
#include <algorithm>
#include <numeric>
#include <cstring>

#define AURA_MAX_COMPUTE_THREADS  (128)
#define AURA_MIN_COMPUTE_THREADS  (1)

#define AURA_MAX_ASYNC_THREADS    (128)
#define AURA_MIN_ASYNC_THREADS    (1)

namespace aura
{

Status SetCpuAffinity(CpuAffinity affinity)
{
    Status status = Status::OK;

#if defined(AURA_BUILD_ANDROID)

#  if !defined(AURA_NCPUBITS)
#    define AURA_NCPUBITS       (8 * sizeof(uint64_t))
#    define CPU_SET_SIZE        (1024)
    struct CpuSet
    {
        DT_U64 __bits[CPU_SET_SIZE / AURA_NCPUBITS];
    };
#    define CPU_SET_MASK(cpu, cpusetp)      ((cpusetp)->__bits[(cpu) / AURA_NCPUBITS] |= (1UL << ((cpu) % AURA_NCPUBITS)))
#    define CPU_ZERO_MASK(cpusetp)           memset((cpusetp), 0, sizeof(CpuSet))

#  endif // AURA_NCPUBITS

#  if defined(__GLIBC__)
    pid_t pid = syscall(SYS_gettid);
#  else
    pid_t pid = gettid();
#  endif // __GLIBC__

    std::vector<DT_S32> cpu_core_idxs = CpuInfo::Get().GetCpuIdxs(affinity);

    if (!cpu_core_idxs.empty())
    {
        CpuSet mask;
        CPU_ZERO_MASK(&mask);
        for (auto id : cpu_core_idxs)
        {
            CPU_SET_MASK(id, &mask);
        }

        if ((syscall(__NR_sched_setaffinity, pid, sizeof(mask), &mask)) != 0)
        {
            status = Status::ERROR;
        }
    }
#else
    AURA_UNUSED(affinity);
#endif // defined(AURA_BUILD_ANDROID)

    return status;
}

WorkerPool::WorkerPool(Context *ctx, const std::string &tag, CpuAffinity compute_affinity, CpuAffinity async_affinity, DT_S32 compute_threads, DT_S32 async_threads)
                       :m_ctx(ctx), m_tag(tag), m_compute_affinity(compute_affinity), m_async_affinity(async_affinity),
                        m_stopped(DT_FALSE), m_async_running_count(0), m_async_thread_idx(0), m_compute_thread_idx(0)
{
    // define maximum number of threads
    std::vector<DT_S32> compute_core_idxs = CpuInfo::Get().GetCpuIdxs(compute_affinity);
    std::vector<DT_S32> async_core_idxs   = CpuInfo::Get().GetCpuIdxs(async_affinity);

    m_max_compute_threads = static_cast<DT_S32>(compute_core_idxs.size());
    m_max_async_threads   = static_cast<DT_S32>(async_core_idxs.size());
    m_max_compute_threads = Clamp(m_max_compute_threads, AURA_MIN_COMPUTE_THREADS, AURA_MAX_COMPUTE_THREADS);
    m_max_async_threads   = Clamp(m_max_async_threads,   AURA_MIN_ASYNC_THREADS,   AURA_MAX_ASYNC_THREADS);

    m_compute_threads.reserve(m_max_compute_threads);
    m_async_threads.reserve(m_max_async_threads);

    // set number of threads to launch
    compute_threads = Clamp(compute_threads, 0, m_max_compute_threads);
    async_threads   = Clamp(async_threads,   0, m_max_async_threads);

    // if explicit number of thread is set, launch threads
    if (compute_threads > 0)
    {
        for (DT_S32 n = 0; n < compute_threads; ++n)
        {
            std::function<DT_VOID(DT_VOID)> compute_thread_run = std::bind(&WorkerPool::ComputeThreadRun, this, m_compute_affinity);
            m_compute_threads.emplace_back(std::move(compute_thread_run));
            m_compute_tid_map[m_compute_threads.back().get_id()] = n + 1; // 0 is reserved for main thread
        }
    }

    if (async_threads > 0)
    {
        for (DT_S32 n = 0; n < async_threads; ++n)
        {
            std::function<DT_VOID(DT_VOID)> async_thread_run = std::bind(&WorkerPool::AsyncThreadRun, this, async_affinity);
            m_async_threads.emplace_back(std::move(async_thread_run));
            m_async_tid_map[m_async_threads.back().get_id()] =  n + 1; // 0 is reserved for main thread
        }
    }
}

DT_VOID WorkerPool::CheckComputeThreads()
{
    // if compute threads are empty, relaunch
    if (m_compute_threads.empty())
    {
        std::unique_lock<std::mutex> lock(m_compute_mutex);

        for (DT_S32 n = 0; n < m_max_compute_threads; ++n)
        {
            std::function<DT_VOID(DT_VOID)> compute_thread_run = std::bind(&WorkerPool::ComputeThreadRun, this, m_compute_affinity);
            m_compute_threads.emplace_back(std::move(compute_thread_run));
            m_compute_tid_map[m_compute_threads.back().get_id()] = n + 1; // 0 is reserved for main thread
        }
    }
}

DT_VOID WorkerPool::CheckAsyncThreads()
{
    DT_S32 num_async_threads = static_cast<DT_S32>(m_async_threads.size());

    if (num_async_threads == m_max_async_threads)
    {
        return;
    }

    if (num_async_threads > static_cast<DT_S32>(m_async_running_count.load()))
    {
        return;
    }

    // add new async thread
    {
        std::unique_lock<std::mutex> lock(m_async_mutex);

        // double check
        num_async_threads = static_cast<DT_S32>(m_async_threads.size());
        if (num_async_threads < m_max_async_threads)
        {
            std::function<DT_VOID(DT_VOID)> async_thread_run = std::bind(&WorkerPool::AsyncThreadRun, this, m_async_affinity);
            m_async_threads.emplace_back(std::move(async_thread_run));
            m_async_tid_map[m_async_threads.back().get_id()] =  num_async_threads + 1;
        }
    }
}

WorkerPool::~WorkerPool()
{
    Stop();

    for (auto &thd : m_async_threads)
    {
        if (thd.joinable())
        {
            thd.join();
        }
    }

    for (auto &thd : m_compute_threads)
    {
        if (thd.joinable())
        {
            thd.join();
        }
    }
}

Status WorkerPool::SetThreadName(const std::string &tag, DT_CHAR type, DT_S32 idx)
{
    // 16 bytes: [tag][type][idx]['\0']; 12 + 1 + 2 + 1
    std::string thread_name = tag.size() > 12 ? tag.substr(0, 12) : tag;
    thread_name += type;

#if defined(ANDROID)
    DT_CHAR name_buf[16] = {0};
    sprintf(name_buf, "%s%02d", thread_name.c_str(), idx % 100);
    prctl(PR_SET_NAME, name_buf);
#else
    AURA_UNUSED(idx);
#endif // ANDROID

    return Status::OK;
}

DT_VOID WorkerPool::AsyncThreadRun(CpuAffinity async_affinity)
{
    if (SetThreadName(m_tag, 'A', m_async_thread_idx++) != Status::OK)
    {
        AURA_LOGE(m_ctx, AURA_TAG, "AsyncThreadRun SetThreadName failed.\n");
    }

    SetCpuAffinity(async_affinity);

    while (DT_TRUE)
    {
        std::function<DT_VOID()> task;

        {
            std::unique_lock<std::mutex> lock(m_async_mutex);
            m_async_wait_cv.wait(lock, [this]{ return m_stopped || !m_async_task_list.empty();});

            if (m_stopped && m_async_task_list.empty())
            {
                return;
            }

            task = std::move(m_async_task_list.front().func);
            m_async_task_list.pop_front();
        }

        ++m_async_running_count;
        task();
        --m_async_running_count;
    }
}

DT_VOID WorkerPool::ComputeThreadRun(CpuAffinity compute_affinity)
{
    if (SetThreadName(m_tag, 'C', m_compute_thread_idx++) != Status::OK)
    {
        AURA_LOGE(m_ctx, AURA_TAG, "ComputeThreadRun SetThreadName failed.\n");
    }

    SetCpuAffinity(compute_affinity);

    while (DT_TRUE)
    {
        std::function<DT_VOID()> task;

        {
            std::unique_lock<std::mutex> lock(m_compute_mutex);
            m_compute_wait_cv.wait(lock, [this]{ return m_stopped || !m_compute_task_list.empty();});

            if (m_stopped && m_compute_task_list.empty())
            {
                return;
            }

            task = std::move(m_compute_task_list.front().func);
            m_compute_task_list.pop_front();
        }

        task();
    }
}

DT_VOID WorkerPool::Stop()
{
    if (m_stopped)
    {
        AURA_LOGI(m_ctx, AURA_TAG, "WorkerPool is already stopped.\n");
        return;
    }

    m_stopped = DT_TRUE;

    {
        std::unique_lock<std::mutex> lock(m_async_mutex);
        m_async_task_list.clear();
    }

    {
        std::unique_lock<std::mutex> lock(m_compute_mutex);
        m_compute_task_list.clear();
    }

    m_async_wait_cv.notify_all();
    m_compute_wait_cv.notify_all();
}

} // namespace aura
