#include "aura/runtime/worker_pool.h"
#include "aura/tools/unit_test.h"

#include <numeric>
#include <cmath>

using namespace aura;

static AURA_VOID TestFunc(Context *ctx, Status &ret, WorkerPool *wp)
{
    Mat mat(ctx, ElemType::F32, Sizes3(1024, 1024, 1));
    Sizes3 size = mat.GetSizes();
    memset(mat.GetData(), 0, mat.GetTotalBytes());

    auto func_benchmark = [&]() -> Status
    {
        for (MI_S32 y = 0; y < size.m_height; ++y)
        {
            for (MI_S32 x = 0; x < size.m_width; ++x)
            {
                mat.At<MI_F32>(y, x, 0) += static_cast<MI_F32>(std::sqrt(y * size.m_width + x));
            }
        }
        return Status::OK;
    };

    auto func = [&](MI_S32 start, MI_S32 end) -> Status
    {
        for (MI_S32 y = start; y < end; ++y)
        {
            for (MI_S32 x = 0; x < size.m_width; ++x)
            {
                mat.At<MI_F32>(y, x, 0) -= static_cast<MI_F32>(std::sqrt(y * size.m_width + x));
            }
        }
        return Status::OK;
    };

    auto perf_func = [&]() -> Status
    {
        return wp->ParallelFor((MI_S32)0, size.m_height, func);
    };

    TestTime time;

    Executor(10, 5, time, func_benchmark);
    AURA_LOGI(ctx, AURA_TAG, "Benchmark run with single thread cost time: (%s)\n", time.ToString().c_str());

    Executor(10, 5, time, perf_func);
    AURA_LOGI(ctx, AURA_TAG, "ParallelFor cost time: (%s)\n", time.ToString().c_str());

    for (MI_S32 y = 0; y < size.m_height; ++y)
    {
        for (MI_S32 x = 0; x < size.m_width; ++x)
        {
            ret |= AURA_CHECK_EQ(ctx, static_cast<MI_S32>(mat.At<MI_F32>(y, x, 0)), (MI_S32)0, "check mat memory failed\n");
        }
    }
}

NEW_TESTCASE(runtime_worker_pool_add_task_test)
{
    Status ret = Status::OK;
    Context *ctx = UnitTest::GetInstance()->GetContext();

    WorkerPool *wp = ctx->GetWorkerPool();

    Mat mat(ctx, ElemType::S32, Sizes3(1024, 1024, 1));
    Sizes3 size = mat.GetSizes();

    auto func = [&](MI_S32 start, MI_S32 end) -> Status
    {
        for (MI_S32 y = start; y < end; ++y)
        {
            for (MI_S32 x = 0; x < size.m_width; ++x)
            {
                mat.At<MI_S32>(y, x, 0) = y * size.m_width + x;
            }
        }
        return Status::OK;
    };

    auto token0 = wp->AddTask(func, 0, 256);
    auto token1 = wp->AddTask(func, 256, 512);
    auto token2 = wp->AddTask(func, 512, 768);
    auto token3 = wp->AddTask(func, 768, 1024);

    WaitTokens(token0, token1, token2, token3);

    for (MI_S32 y = 0; y < size.m_height; ++y)
    {
        for (MI_S32 x = 0; x < size.m_width; ++x)
        {
            ret |= AURA_CHECK_EQ(ctx, mat.At<MI_S32>(y, x, 0), y * size.m_width + x, "check mat memory failed\n");
        }
    }

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}

NEW_TESTCASE(runtime_worker_pool_constructor_test)
{
    Status ret = Status::OK;
    Context *ctx = UnitTest::GetInstance()->GetContext();

    TestFunc(ctx, ret, ctx->GetWorkerPool());

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}

class ExampleClass
{
public:
    ExampleClass(Context *ctx) : value(0), m_ctx(ctx)
    {}

    ~ExampleClass()
    {}

    ExampleClass(const ExampleClass& src)
    {
        this->value = src.value;
        this->m_ctx = src.m_ctx;
        AURA_LOGI(m_ctx, AURA_TAG, "ExampleClass's copy construct function is called.\n");
    }

    ExampleClass& operator= (const ExampleClass& src)
    {
        this->value = src.value;
        AURA_LOGI(m_ctx, AURA_TAG, "ExampleClass's operator= is called.\n");
        return *this;
    }

    AURA_VOID Print()
    {
        AURA_LOGI(m_ctx, AURA_TAG, "ExampleClass's Print functions is called, value is: %d\n", this->value);
    }

    MI_S32 value;
    Context *m_ctx;
};

static Status InputParallelForFunc(Context *ctx, ExampleClass &obj, MI_S32 start, MI_S32 end)
{
    AURA_LOGI(ctx, AURA_TAG, "InputParallelForFunc with args: %d %d\n", start, end);
    obj.value = 777;

    return Status::OK;
}

NEW_TESTCASE(runtime_worker_pool_ref_param_test)
{
    Status ret = Status::OK;
    Context *ctx = UnitTest::GetInstance()->GetContext();

    ExampleClass src0(ctx);
    ExampleClass src1(ctx);
    /// Input params with reference type must use std::ref because of std::bind make a copy default
    /// https://stackoverflow.com/questions/31810985/why-does-bind-not-work-with-pass-by-reference
    ctx->GetWorkerPool()->ParallelFor(0, 1, InputParallelForFunc, ctx, std::ref(src0));
    AURA_LOGI(ctx, AURA_TAG, "ParallelFor with std::ref(src) result: %d\n", src0.value);

    ctx->GetWorkerPool()->ParallelFor(0, 1, InputParallelForFunc, ctx, src1);
    AURA_LOGI(ctx, AURA_TAG, "ParallelFor with src result: %d\n", src1.value);
    AddTestResult(AURA_GET_TEST_STATUS(ret));
}

NEW_TESTCASE(runtime_worker_pool_stack_size)
{
    Status ret = Status::OK;
    Context *ctx = UnitTest::GetInstance()->GetContext();

    {
        const MI_S32 stack_size = 16;
        Time start = Time::Now();
        WorkerPool worker_pool(ctx, stack_size * 1024, "CustomTag");
        AURA_LOGI(ctx, AURA_TAG, "Create thread with stack size: %ldKB cost time: %s\n", stack_size, (Time::Now() - start).ToString().c_str());
    }

    {
        const MI_S32 stack_size = 32;
        Time start = Time::Now();
        WorkerPool worker_pool(ctx, stack_size * 1024, "CustomTag");
        AURA_LOGI(ctx, AURA_TAG, "Create thread with stack size: %ldKB cost time: %s\n", stack_size, (Time::Now() - start).ToString().c_str());
    }

    {
        const MI_S32 stack_size = 64;
        Time start = Time::Now();
        WorkerPool worker_pool(ctx, stack_size * 1024, "CustomTag");
        AURA_LOGI(ctx, AURA_TAG, "Create thread with stack size: %ldKB cost time: %s\n", stack_size, (Time::Now() - start).ToString().c_str());
    }

    {
        const MI_S32 stack_size = 128;
        Time start = Time::Now();
        WorkerPool worker_pool(ctx, stack_size * 1024, "CustomTag");
        AURA_LOGI(ctx, AURA_TAG, "Create thread with stack size: %ldKB cost time: %s\n", stack_size, (Time::Now() - start).ToString().c_str());
    }

    {
        const MI_S32 stack_size = 512;
        Time start = Time::Now();
        WorkerPool worker_pool(ctx, stack_size * 1024, "CustomTag");
        AURA_LOGI(ctx, AURA_TAG, "Create thread with stack size: %ldKB cost time: %s\n", stack_size, (Time::Now() - start).ToString().c_str());
    }

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}

NEW_TESTCASE(runtime_worker_pool_get_tid_test)
{
    Status ret = Status::OK;
    Context *ctx = UnitTest::GetInstance()->GetContext();
    const MI_S32 thread_num = ctx->GetWorkerPool()->GetComputeThreadNum();

    AURA_LOGI(ctx, AURA_TAG, "ThreadTest (thread number: %d)\n", thread_num);
    auto FuncGetThreadId = [ctx](MI_S32 start, MI_S32 end) -> Status
    {
        MI_S32 thread_id = ctx->GetWorkerPool()->GetComputeThreadIdx();
        AURA_LOGI(ctx, AURA_TAG, "running %2d - %2d: thread_id = %d, qurt_thread_get_id() = %d\n",
                  start, end, thread_id, qurt_thread_get_id());

        if (-1 == thread_id)
        {
            AURA_LOGE(ctx, AURA_TAG, "thread cannot be found\n");
            return Status::ERROR;
        }

        return Status::OK;
    };

    for (MI_S32 loop = 0; loop < 1000; loop++)
    {
        AURA_LOGI(ctx, AURA_TAG, "ThreadTest one:\n");
        ret |= ctx->GetWorkerPool()->ParallelFor((MI_S32)0, thread_num, FuncGetThreadId);

        AURA_LOGI(ctx, AURA_TAG, "ThreadTest two:\n");
        ret |= ctx->GetWorkerPool()->ParallelFor((MI_S32)0, thread_num * 8, FuncGetThreadId);
    }

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}

NEW_TESTCASE(runtime_worker_pool_wave_front_test)
{
    Status ret   = Status::OK;
    Context *ctx = UnitTest::GetInstance()->GetContext();

    Mat mat(ctx, ElemType::U16, Sizes3(2048, 4096, 1));
    Mat ref(ctx, ElemType::U16, Sizes3(2048, 4096, 1));

    Sizes3 size = mat.GetSizes();

    for (MI_S32 row = 0; row < size.m_height; row++)
    {
        for (MI_S32 col = 0; col < size.m_width; col++)
        {
            mat.At<MI_U16>(row, col, 0) = (MI_U16)(rand() % 256);
            ref.At<MI_U16>(row, col, 0) = mat.At<MI_U16>(row, col, 0);
        }
    }

    auto func_wpf = [&](MI_S32 row_step, MI_S32 col_step, MI_S32 row, MI_S32 col) -> Status
    {
        MI_S32 row_start = row * row_step + 1;
        MI_S32 row_end   = Min(row_start + row_step, mat.GetSizes().m_height);
        MI_S32 col_start = col * col_step + 1;
        MI_S32 col_end   = Min(col_start + col_step, mat.GetSizes().m_width);

        for (MI_S32 y = row_start; y < row_end; ++y)
        {
            for (MI_S32 x = col_start; x < col_end; ++x)
            {
                mat.At<MI_U16>(y, x, 0) = (MI_U16)(mat.At<MI_U16>(y, x - 1, 0) + mat.At<MI_U16>(y - 1, x, 0));
            }
        }
        return Status::OK;
    };

    auto func_ref = [&]() -> Status
    {
        for (MI_S32 y = 1; y < ref.GetSizes().m_height; ++y)
        {
            for (MI_S32 x = 1; x < ref.GetSizes().m_width; ++x)
            {
                ref.At<MI_U16>(y, x, 0) = (MI_U16)(ref.At<MI_U16>(y, x - 1, 0) + ref.At<MI_U16>(y - 1, x, 0));
            }
        }
        return Status::OK;
    };

    MI_S32 block_height = 256;
    MI_S32 block_width  = 256;

    // test GetComputeThreadIdx
    AURA_LOGI(ctx, AURA_TAG, "WaveFront test begins\n");
    ctx->GetWorkerPool()->WaveFront((size.m_height + block_height - 2) / block_height,
                                    (size.m_width  + block_width  - 2) / block_width,
                                    func_wpf, block_height, block_width);
    func_ref();

    MatCmpResult cmp_result;
    MatCompare(ctx, mat, ref, cmp_result, 1);

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}