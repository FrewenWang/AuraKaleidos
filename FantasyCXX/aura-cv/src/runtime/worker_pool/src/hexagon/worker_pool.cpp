#include "aura/runtime/worker_pool/hexagon/worker_pool.hpp"
#include "aura/runtime/logger.h"
#include "aura/runtime/memory.h"

#include "AEEStdDef.h"
#include "HAP_power.h"

#include "qurt.h"
#include "qurt_hvx.h"

#include <fstream>
#include <algorithm>
#include <numeric>
#include <cstring>

#define LOWEST_USABLE_QURT_PRIO             (254)

namespace aura
{

static MI_S32 GetEffectiveNumThreads()
{
    MI_S32 hvx_units = (qurt_hvx_get_units() >> 8) & 0xFF;
    return Max(hvx_units, (MI_S32)1);
}

static Status HTPSetPower(Context *ctx, MI_BOOL flag)
{
#if (__HEXAGON_ARCH__ < 68)
    HAP_power_request_t request;
    request.type = HAP_power_set_HVX;
    request.hvx.power_up = flag;

    if (HAP_power_set(ctx, &request) != 0)
    {
        return Status::ERROR;
    }
    else
    {
        return Status::OK;
    }
#else
    AURA_UNUSED(ctx);
    AURA_UNUSED(flag);
#endif
    return Status::OK;
}

AURA_VOID ThreadRun(AURA_VOID *instance)
{
    if (MI_NULL == instance)
    {
        return;
    }

    WorkerPool *wp = reinterpret_cast<WorkerPool *>(instance);

    while (MI_TRUE)
    {
        std::function<AURA_VOID()> task;

        {
            std::unique_lock<std::mutex> lock(wp->m_mutex);
            wp->m_wait_cv.wait(lock, [wp]{ return wp->m_stopped || !wp->m_task_queue.empty();});

            if (wp->m_stopped && wp->m_task_queue.empty())
            {
                return;
            }

            task = std::move(wp->m_task_queue.front());
            wp->m_task_queue.pop();
        }

        qurt_hvx_lock(QURT_HVX_MODE_128B);
        task();
        qurt_hvx_unlock();
    }
}

WorkerPool::WorkerPool(Context *ctx, MI_S32 stack_sz, const std::string &tag) : m_ctx(ctx), m_stopped(MI_FALSE), m_tag(tag), m_stack(MI_NULL)
{
    if (HTPSetPower(ctx, MI_TRUE) != Status::OK)
    {
        AURA_LOGE(ctx, AURA_TAG, "HTPSetPower on failed.\n");
    }

    MI_S32 thread_count = GetEffectiveNumThreads();

    m_threads.reserve(thread_count);

    stack_sz = Max(stack_sz, 4096l);
    // stack pointer must aligned to 8 bytes
    stack_sz = AURA_ALIGN(stack_sz, 8);
    m_stack = static_cast<MI_U8*>(AURA_ALLOC(ctx, stack_sz * thread_count));

    if (!m_stack)
    {
        AURA_LOGE(ctx, AURA_TAG, "WorkerPool create thread stack failed.\n");
    }
    else
    {
        MI_CHAR name_buffer[128] = {0};
        qurt_thread_attr_t attr;
        qurt_thread_attr_init(&attr);

        for (MI_S32 n = 0; n < thread_count; ++n)
        {
            snprintf(name_buffer, 128, "%sC%02ld", m_tag.c_str(), n);
            qurt_thread_attr_set_stack_addr(&attr, m_stack + n * stack_sz);
            qurt_thread_attr_set_stack_size(&attr, stack_sz);
            qurt_thread_attr_set_name(&attr, name_buffer);

            MI_S32 priority = qurt_thread_get_priority(qurt_thread_get_id());

            if (priority < 1)
            {
                priority = 1;
            }

            if (priority > LOWEST_USABLE_QURT_PRIO)
            {
                priority = LOWEST_USABLE_QURT_PRIO;
            }

            qurt_thread_attr_set_priority(&attr, priority);

            qurt_thread_t tid = 0;

            if (qurt_thread_create(&tid, &attr, ThreadRun, this) != 0)
            {
                AURA_LOGE(ctx, AURA_TAG, "Could not launch worker threads!\n");
            }
            else
            {
                m_threads.push_back(tid);
                m_tid_map[m_threads.back()] = n;
            }
        }
    }
}

WorkerPool::~WorkerPool()
{
    Stop();

    for (size_t n = 0; n < m_threads.size(); n++)
    {
        if (m_threads[n] != 0)
        {
            qurt_thread_join(m_threads[n], MI_NULL);
        }
    }

    AURA_FREE(m_ctx, m_stack);

    if (HTPSetPower(m_ctx, MI_FALSE) != Status::OK)
    {
        AURA_LOGE(m_ctx, AURA_TAG, "HTPSetPower off failed.\n");
    }
}

AURA_VOID WorkerPool::Stop()
{
    if (m_stopped)
    {
        AURA_LOGI(m_ctx, AURA_TAG, "WorkerPool is already stopped.\n");
        return;
    }

    m_stopped = MI_TRUE;

    {
        std::queue<std::function<AURA_VOID()>> empty_queue;
        std::unique_lock<std::mutex> lock(m_mutex);
        m_task_queue.swap(empty_queue);
    }

    m_wait_cv.notify_all();
}

} // namespace aura