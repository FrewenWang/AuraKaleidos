#include "aura/runtime/thread_buffer.h"
#include "aura/runtime/worker_pool.h"
#include "aura/tools/unit_test.h"

using namespace aura;

static Status ThreadTask(Context *ctx, ThreadBuffer &buffer, DT_S32 start_idx, DT_S32 stop_idx)
{
    DT_S32 idx = ctx->GetWorkerPool()->GetComputeThreadIdx();

    std::thread::id thread_id = std::this_thread::get_id();
    std::stringstream ss_thread_id;
    ss_thread_id << thread_id;

    AURA_LOGI(ctx, AURA_TAG, "Thread[%d], ID: %s, process task: %2d~%2d\n", idx, ss_thread_id.str().c_str(), start_idx, stop_idx);

    auto thread_data = buffer.GetThreadData<DT_S32>();

    for (DT_S32 i = 0; i < buffer.GetThreadBuffer().m_size / (DT_S32)sizeof(DT_S32); i++)
    {
        thread_data[i]++;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    return Status::OK;
}

NEW_TESTCASE(runtime_utils_thread_buffer_test)
{
    Context *ctx = UnitTest::GetInstance()->GetContext();

    DT_S32 max_task_count = 128;
    DT_S32 buffer_bytes   = 64 * sizeof(DT_S32);
    auto thread_ids       = ctx->GetWorkerPool()->GetComputeThreadIDs();

    ThreadBuffer thread_buffer0(ctx, buffer_bytes);
    ThreadBuffer thread_buffer1(ctx, thread_ids, buffer_bytes);

    std::vector<ThreadBuffer*> thread_buffers = {&thread_buffer0, &thread_buffer1};

    Status status = Status::OK;

    for (auto &thread_buffer : thread_buffers)
    {
        for (auto &thread_id : thread_ids)
        {
            DT_U8 *thread_data = thread_buffer->GetThreadData<DT_U8>(thread_id);
            memset(thread_data, 0, buffer_bytes);
        }

        ctx->GetWorkerPool()->ParallelFor((DT_S32)0, max_task_count, ThreadTask, ctx, std::ref(*thread_buffer));

        DT_S32 elem_count = buffer_bytes / (DT_S32)sizeof(DT_S32);

        for (DT_S32 i = 0; i < static_cast<DT_S32>(thread_ids.size()); i++)
        {
            AURA_LOGD(ctx, AURA_TAG, "Buffer[%d] Status: \n", i);

            DT_S32 *cur_data = thread_buffer->GetThreadData<DT_S32>(thread_ids[i]);

            for (DT_S32 j = 0; j < elem_count; j++)
            {
                if (cur_data[j] != cur_data[0])
                {
                    status |= Status::ERROR;
                    AURA_LOGE(ctx, AURA_TAG, "Racing happened.\n");
                    break;
                }
            }
        }

    }

    AddTestResult(AURA_GET_TEST_STATUS(status));
}
