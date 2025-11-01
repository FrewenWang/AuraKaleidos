#include "aura/runtime/worker_pool.h"
#include "aura/tools/unit_test.h"

#include <numeric>
#include <cmath>
#include <fstream>

using namespace aura;

static AURA_VOID TestFunc(Context *ctx, Status &ret, WorkerPool *wp)
{
    Mat mat(ctx, ElemType::F64, Sizes3(1024, 1024, 1));
    Sizes3 size = mat.GetSizes();
    memset(mat.GetData(), 0, mat.GetTotalBytes());

    auto func_benchmark = [&]() -> Status
    {
        for (MI_S32 y = 0; y < size.m_height; ++y)
        {
            for (MI_S32 x = 0; x < size.m_width; ++x)
            {
                mat.At<MI_F64>(y, x, 0) += static_cast<MI_F64>(std::sqrt(y * size.m_width + x));
            }
        }
        return Status::OK;
    };

    auto func = [&](MI_S32 task_count, MI_S32 start, MI_S32 end) -> Status
    {
        start = start * task_count;
        end = Min(end * task_count, size.m_height);
        for (MI_S32 y = start; y < end; ++y)
        {
            for (MI_S32 x = 0; x < size.m_width; ++x)
            {
                mat.At<MI_F64>(y, x, 0) -= static_cast<MI_F64>(std::sqrt(y * size.m_width + x));
            }
        }
        return Status::OK;
    };

    auto perf_func = [&](MI_S32 task_count) -> Status
    {
        return wp->ParallelFor(0, AURA_ALIGN(size.m_height, task_count) / task_count, func, task_count);
    };

    TestTime time;

    Executor(110, 5, time, func_benchmark);
    AURA_LOGI(ctx, AURA_TAG, "Benchmark run with single thread cost time: (%s)\n", time.ToString().c_str());

    for (MI_S32 count = 1; count <= 1024; count *= 2)
    {
        Executor(10, 5, time, perf_func, count);
        AURA_LOGI(ctx, AURA_TAG, "ParallelFor with args: partial_count: [%4d] step: [%8d] cost time: (%s)\n", count, size.m_height / count, time.ToString().c_str());
    }
    for (MI_S32 y = 0; y < size.m_height; ++y)
    {
        for (MI_S32 x = 0; x < size.m_width; ++x)
        {
            ret |= AURA_CHECK_EQ(ctx, static_cast<MI_S32>(mat.At<MI_F64>(y, x, 0)), 0, "check mat memory failed\n");
        }
    }
}

NEW_TESTCASE(runtime_worker_pool_constructor_test)
{
    Status ret = Status::OK;
    Context *ctx = UnitTest::GetInstance()->GetContext();

    {
        AURA_LOGI(ctx, AURA_TAG, "Test WorkerPool : CpuAffinity::ALL\n");
        WorkerPool wp(ctx, AURA_TAG, CpuAffinity::ALL, CpuAffinity::ALL);
        auto token = wp.AsyncRun(TestFunc, ctx, std::ref(ret), &wp);
        WaitTokens(token);
        AURA_LOGI(ctx, AURA_TAG, "---------------------------------------------\n");
    }
    {
        AURA_LOGI(ctx, AURA_TAG, "Test WorkerPool : CpuAffinity::BIG\n");
        WorkerPool wp(ctx, AURA_TAG, CpuAffinity::BIG, CpuAffinity::BIG);
        auto token = wp.AsyncRun(TestFunc, ctx, std::ref(ret), &wp);
        WaitTokens(token);
        AURA_LOGI(ctx, AURA_TAG, "---------------------------------------------\n");
    }
    {
        AURA_LOGI(ctx, AURA_TAG, "Test WorkerPool : CpuAffinity::LITTLE\n");
        WorkerPool wp(ctx, AURA_TAG, CpuAffinity::LITTLE, CpuAffinity::LITTLE);
        auto token = wp.AsyncRun(TestFunc, ctx, std::ref(ret), &wp);
        WaitTokens(token);
        AURA_LOGI(ctx, AURA_TAG, "---------------------------------------------\n");
    }

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}

NEW_TESTCASE(runtime_worker_pool_async_run_test)
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

    auto token0 = wp->AsyncRun(func, 0, 256);
    auto token1 = wp->AsyncRun(func, 256, 512);
    auto token2 = wp->AsyncRun(func, 512, 768);
    auto token3 = wp->AsyncRun(func, 768, 1024);

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

NEW_TESTCASE(runtime_worker_pool_stop_test)
{
    Status ret = Status::OK;
    Context *ctx = UnitTest::GetInstance()->GetContext();

    WorkerPool wp(ctx, AURA_TAG, CpuAffinity::ALL, CpuAffinity::ALL, 4, 1);
    WorkerPool *wp_ptr = &wp;

    MI_S32 total_task_count = 50;
    std::atomic_int task_count_cur(total_task_count);

    auto func0 = [=, &task_count_cur]()
    {
        while(MI_TRUE)
        {
            if (task_count_cur < total_task_count / 2)
            {
                AURA_LOGI(ctx, AURA_TAG, "task_count < total_task_count / 2, stop the worker_pool.\n");
                wp_ptr->Stop();
                break;
            }
        }
    };

    auto func1 = [&](MI_S32 start, MI_S32 stop) -> Status
    {
        AURA_UNUSED(start);
        AURA_UNUSED(stop);

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        --task_count_cur;
        MI_S32 result = task_count_cur;
        AURA_LOGI(ctx, AURA_TAG, "task_count current : [%2d].\n", result);
        return Status::OK;
    };

    auto token0 = wp_ptr->AsyncRun(func0);
    AURA_LOGI(ctx, AURA_TAG, "AsyncRun func0, main thread continue run\n");
    wp_ptr->ParallelFor(0, total_task_count, func1);

    WaitTokens(token0);

    AURA_LOGI(ctx, AURA_TAG, "AsyncRun add task to an stopped worker_pool\n");

    auto token = wp_ptr->AsyncRun([&]()
    {
        AURA_LOGI(ctx, AURA_TAG, "Add new task to a stopped worker_pool.\n");
    });
    WaitTokens(token);

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

NEW_TESTCASE(runtime_worker_pool_set_name_test)
{
    Status ret = Status::OK;

    Context *ctx = UnitTest::GetInstance()->GetContext();

    Config algo_ctx_cfg;
    // Set thread tag for algo1 context
    algo_ctx_cfg.SetWorkerPool("AlgoTag", CpuAffinity::BIG, CpuAffinity::LITTLE);
    std::shared_ptr<Context> ctx1(new Context(algo_ctx_cfg));

    ret = ctx1->Initialize();
    if (ret != Status::OK)
    {
        AURA_LOGE(ctx, AURA_TAG, "Initialize algo1 context failed.\n");
    }
    // Use shell cmd to get aura_test_main thread names:
    // eg: adb shell 'ps -T -p $(pgrep aura_test_main)'
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    AddTestResult(AURA_GET_TEST_STATUS(ret));
}

NEW_TESTCASE(runtime_worker_pool_get_tid_test)
{
    Status ret = Status::OK;

    Context *ctx = UnitTest::GetInstance()->GetContext();
    MI_S32 n_compute_threads = ctx->GetWorkerPool()->GetComputeThreadNum();

    // lambda function : GetComputeThreadIdx
    auto func_get_compute_tid = [ctx](MI_S32 start, MI_S32 end) -> Status
    {
        MI_S32 tid = ctx->GetWorkerPool()->GetComputeThreadIdx();
        std::ostringstream oss;
        oss << "ParallelFor with args: " << start << " " << end << ", tid: " << tid << ", thread_id: " << std::this_thread::get_id() << "\n";
        AURA_LOGI(ctx, AURA_TAG, "%s", oss.str().c_str());
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        return Status::OK;
    };

    // lambda function : GetAsyncThreadIdx
    auto func_get_async_tid = [ctx]() -> Status
    {
        MI_S32 tid = ctx->GetWorkerPool()->GetAsyncThreadIdx();
        std::ostringstream oss;
        oss << "AsyncRun, tid: " << tid << ", thread_id: " << std::this_thread::get_id() << "\n";
        AURA_LOGI(ctx, AURA_TAG, "%s", oss.str().c_str());
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        return Status::OK;
    };

    // test GetComputeThreadIdx
    AURA_LOGI(ctx, AURA_TAG, "ComputeThreadTest (n_compute_threads: %d)\n", n_compute_threads);
    ctx->GetWorkerPool()->ParallelFor(0, n_compute_threads, func_get_compute_tid);

    // test GetAsyncThreadIdx
    MI_S32 n_async_threads = ctx->GetWorkerPool()->GetAsyncThreadNum();
    AURA_LOGI(ctx, AURA_TAG, "AsyncThreadTest (n_async_threads: %d)\n", n_async_threads);

    // sync_number = n_async_threads
    {
        std::vector<std::future<Status>> tokens;
        for (MI_S32 n = 0; n < n_async_threads; ++n)
        {
            auto token = ctx->GetWorkerPool()->AsyncRun(func_get_async_tid);
            tokens.push_back(std::move(token));
        }
        WaitTokens(tokens);
    }

    // sync_number = n_async_threads * 4
    {
        std::vector<std::future<Status>> tokens;
        for (MI_S32 n = 0; n < n_async_threads * 4; ++n)
        {
            auto token = ctx->GetWorkerPool()->AsyncRun(func_get_async_tid);
            tokens.push_back(std::move(token));
        }
        WaitTokens(tokens);
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
                                    (size.m_width + block_width  - 2) / block_width,
                                    func_wpf, block_height, block_width);
    func_ref();

    MatCmpResult cmp_result;
    MatCompare(ctx, mat, ref, cmp_result, 1);

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}

static AURA_VOID AsyncTestFunc0(Context *ctx)
{
    std::thread::id tid = std::this_thread::get_id();

    std::stringstream ss;
    ss << "AsyncTestFunc0 is called from thread_id: " << tid << "\n";
    AURA_LOGI(ctx, AURA_TAG, "%s\n", ss.str().c_str());

    // define lambda function
    auto func = [&](MI_S32 start, MI_S32 end) -> Status
    {
        std::thread::id tid = std::this_thread::get_id();
        std::stringstream ss;
        ss << "Lambda function 0 is called from thread_id: " << tid << ": " << start << " ~ " << end <<  "\n";
        AURA_LOGI(ctx, AURA_TAG, "%s\n", ss.str().c_str());

        std::this_thread::sleep_for(std::chrono::milliseconds(20));

        return Status::OK;
    };

    WorkerPool *wp = ctx->GetWorkerPool();
    wp->ParallelFor(0, 10, func);
}

static AURA_VOID AsyncTestFunc1(Context *ctx)
{
    std::thread::id tid = std::this_thread::get_id();

    std::stringstream ss;
    ss << "AsyncTestFunc1 is called from thread_id: " << tid << "\n";
    AURA_LOGI(ctx, AURA_TAG, "%s\n", ss.str().c_str());

    // define lambda function
    auto func = [&](MI_S32 start, MI_S32 end) -> Status
    {
        std::thread::id tid = std::this_thread::get_id();
        std::stringstream ss;
        ss << "Lambda function 1 is called from thread_id: " << tid << ": " << start << " ~ " << end <<  "\n";
        AURA_LOGI(ctx, AURA_TAG, "%s\n", ss.str().c_str());

        std::this_thread::sleep_for(std::chrono::milliseconds(20));

        return Status::OK;
    };

    WorkerPool *wp = ctx->GetWorkerPool();
    wp->ParallelFor(0, 10, func);
}

NEW_TESTCASE(runtime_worker_pool_async_parallelfor_test)
{
    Status ret = Status::OK;
    Context *ctx = UnitTest::GetInstance()->GetContext();

    WorkerPool *wp = ctx->GetWorkerPool();

    auto token0 = wp->AsyncRun(AsyncTestFunc0, ctx);
    auto token1 = wp->AsyncRun(AsyncTestFunc1, ctx);
    WaitTokens(token0, token1);

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}

#define  CHECK_COUNT_VAL(val, expect)                                 \
{                                                                     \
    if (val.load() != expect)                                         \
    {                                                                 \
        AURA_LOGE(g_ctx, AURA_TAG, "count value is not correct\n");   \
        AddTestResult(TestStatus::FAILED);                            \
        return;                                                       \
    }                                                                 \
}
NEW_TESTCASE(runtime_worker_pool_thread_num_test)
{
    Context *g_ctx = UnitTest::GetInstance()->GetContext();

    std::atomic<MI_S32> count(0);

    // define lambda function
    auto func = [&](MI_S32 start, MI_S32 end) -> Status
    {
        std::thread::id tid = std::this_thread::get_id();
        std::stringstream ss;
        ss << "Lambda function is called from thread_id: " << tid << ": " << start << " ~ " << end <<  "\n";
        AURA_LOGI(g_ctx, AURA_TAG, "%s\n", ss.str().c_str());

        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        count.fetch_add(1);
        AURA_LOGI(g_ctx, AURA_TAG, "function done (%d ~ %d)\n", start, end);
        return Status::OK;
    };

    auto test0 = [&]() -> AURA_VOID
    {
        Config cfg;
        cfg.SetWorkerPool("dynamic_wp_test", CpuAffinity::ALL, CpuAffinity::ALL);

        std::shared_ptr<aura::Context> ctx = std::make_shared<aura::Context>(cfg);
        if (MI_NULL == ctx)
        {
            return;
        }

        if (ctx->Initialize() != aura::Status::OK)
        {
            AURA_LOGE(ctx.get(), AURA_TAG, "aura::Context::Initialize() failed\n");
            return;
        }

        aura::WorkerPool *wp = ctx->GetWorkerPool();

        AURA_LOGD(ctx.get(), AURA_TAG, "compute thread num: %d async thread num: %d\n",
                                wp->GetComputeThreadNum(), wp->GetAsyncThreadNum());
        
        AURA_LOGD(ctx.get(), AURA_TAG, "parallel for test \n");
        count.store(0);
        wp->ParallelFor(0, 10, func);
        CHECK_COUNT_VAL(count, 10);

        AURA_LOGD(ctx.get(), AURA_TAG, "async run test \n");
        count.store(0);
        auto token0  = wp->AsyncRun(func, 0,  1);       std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto token1  = wp->AsyncRun(func, 1,  2);       std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto token2  = wp->AsyncRun(func, 2,  3);       std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto token3  = wp->AsyncRun(func, 3,  4);       std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto token4  = wp->AsyncRun(func, 4,  5);       std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto token5  = wp->AsyncRun(func, 5,  6);       std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto token6  = wp->AsyncRun(func, 6,  7);       std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto token7  = wp->AsyncRun(func, 7,  8);       std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto token8  = wp->AsyncRun(func, 8,  9);       std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto token9  = wp->AsyncRun(func, 9,  10);      std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto token10 = wp->AsyncRun(func, 10, 11);      std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto token11 = wp->AsyncRun(func, 11, 12);      std::this_thread::sleep_for(std::chrono::milliseconds(10));
        WaitTokens(token0, token1, token2, token3, token4, token5,
                   token6, token7, token8, token9, token10, token11);
        CHECK_COUNT_VAL(count, 12);
    };

    auto test1 = [&]() -> AURA_VOID
    {
        Config cfg;
        cfg.SetWorkerPool("dynamic_wp_test", CpuAffinity::ALL, CpuAffinity::ALL, -1, -1);

        std::shared_ptr<aura::Context> ctx = std::make_shared<aura::Context>(cfg);
        if (MI_NULL == ctx)
        {
            return;
        }

        if (ctx->Initialize() != aura::Status::OK)
        {
            AURA_LOGE(ctx.get(), AURA_TAG, "aura::Context::Initialize() failed\n");
            return;
        }

        aura::WorkerPool *wp = ctx->GetWorkerPool();

        AURA_LOGD(ctx.get(), AURA_TAG, "compute thread num: %d async thread num: %d\n",
                                wp->GetComputeThreadNum(), wp->GetAsyncThreadNum());
        
        AURA_LOGD(ctx.get(), AURA_TAG, "parallel for test \n");
        count.store(0);
        wp->ParallelFor(0, 10, func);
        CHECK_COUNT_VAL(count, 10);

        AURA_LOGD(ctx.get(), AURA_TAG, "async run test \n");
        count.store(0);
        auto token0  = wp->AsyncRun(func, 0,  1);       std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto token1  = wp->AsyncRun(func, 1,  2);       std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto token2  = wp->AsyncRun(func, 2,  3);       std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto token3  = wp->AsyncRun(func, 3,  4);       std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto token4  = wp->AsyncRun(func, 4,  5);       std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto token5  = wp->AsyncRun(func, 5,  6);       std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto token6  = wp->AsyncRun(func, 6,  7);       std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto token7  = wp->AsyncRun(func, 7,  8);       std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto token8  = wp->AsyncRun(func, 8,  9);       std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto token9  = wp->AsyncRun(func, 9,  10);      std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto token10 = wp->AsyncRun(func, 10, 11);      std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto token11 = wp->AsyncRun(func, 11, 12);      std::this_thread::sleep_for(std::chrono::milliseconds(10));
        WaitTokens(token0, token1, token2, token3, token4, token5,
                   token6, token7, token8, token9, token10, token11);
        CHECK_COUNT_VAL(count, 12);
    };

    auto test2 = [&]() -> AURA_VOID
    {
        Config cfg;
        cfg.SetWorkerPool("dynamic_wp_test", CpuAffinity::ALL, CpuAffinity::ALL, 4, 4);

        std::shared_ptr<aura::Context> ctx = std::make_shared<aura::Context>(cfg);
        if (MI_NULL == ctx)
        {
            return;
        }

        if (ctx->Initialize() != aura::Status::OK)
        {
            AURA_LOGE(ctx.get(), AURA_TAG, "aura::Context::Initialize() failed\n");
            return;
        }

        aura::WorkerPool *wp = ctx->GetWorkerPool();

        AURA_LOGD(ctx.get(), AURA_TAG, "compute thread num: %d async thread num: %d\n",
                                wp->GetComputeThreadNum(), wp->GetAsyncThreadNum());
        
        AURA_LOGD(ctx.get(), AURA_TAG, "parallel for test \n");
        count.store(0);
        wp->ParallelFor(0, 10, func);
        CHECK_COUNT_VAL(count, 10);

        AURA_LOGD(ctx.get(), AURA_TAG, "async run test \n");
        count.store(0);
        auto token0  = wp->AsyncRun(func, 0,  1);       std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto token1  = wp->AsyncRun(func, 1,  2);       std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto token2  = wp->AsyncRun(func, 2,  3);       std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto token3  = wp->AsyncRun(func, 3,  4);       std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto token4  = wp->AsyncRun(func, 4,  5);       std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto token5  = wp->AsyncRun(func, 5,  6);       std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto token6  = wp->AsyncRun(func, 6,  7);       std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto token7  = wp->AsyncRun(func, 7,  8);       std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto token8  = wp->AsyncRun(func, 8,  9);       std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto token9  = wp->AsyncRun(func, 9,  10);      std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto token10 = wp->AsyncRun(func, 10, 11);      std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto token11 = wp->AsyncRun(func, 11, 12);      std::this_thread::sleep_for(std::chrono::milliseconds(10));
        WaitTokens(token0, token1, token2, token3, token4, token5,
                   token6, token7, token8, token9, token10, token11);
        CHECK_COUNT_VAL(count, 12);
    };

    auto test3 = [&]() -> AURA_VOID
    {
        Config cfg;
        cfg.SetWorkerPool("dynamic_wp_test", CpuAffinity::ALL, CpuAffinity::ALL, 20000, 20000);

        std::shared_ptr<aura::Context> ctx = std::make_shared<aura::Context>(cfg);
        if (MI_NULL == ctx)
        {
            return;
        }

        if (ctx->Initialize() != aura::Status::OK)
        {
            AURA_LOGE(ctx.get(), AURA_TAG, "aura::Context::Initialize() failed\n");
            return;
        }

        aura::WorkerPool *wp = ctx->GetWorkerPool();

        AURA_LOGD(ctx.get(), AURA_TAG, "compute thread num: %d async thread num: %d\n",
                                wp->GetComputeThreadNum(), wp->GetAsyncThreadNum());
        
        AURA_LOGD(ctx.get(), AURA_TAG, "parallel for test \n");
        count.store(0);
        wp->ParallelFor(0, 10, func);
        CHECK_COUNT_VAL(count, 10);

        AURA_LOGD(ctx.get(), AURA_TAG, "async run test \n");
        count.store(0);
        auto token0  = wp->AsyncRun(func, 0,  1);       std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto token1  = wp->AsyncRun(func, 1,  2);       std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto token2  = wp->AsyncRun(func, 2,  3);       std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto token3  = wp->AsyncRun(func, 3,  4);       std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto token4  = wp->AsyncRun(func, 4,  5);       std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto token5  = wp->AsyncRun(func, 5,  6);       std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto token6  = wp->AsyncRun(func, 6,  7);       std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto token7  = wp->AsyncRun(func, 7,  8);       std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto token8  = wp->AsyncRun(func, 8,  9);       std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto token9  = wp->AsyncRun(func, 9,  10);      std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto token10 = wp->AsyncRun(func, 10, 11);      std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto token11 = wp->AsyncRun(func, 11, 12);      std::this_thread::sleep_for(std::chrono::milliseconds(10));
        WaitTokens(token0, token1, token2, token3, token4, token5,
                   token6, token7, token8, token9, token10, token11);
        CHECK_COUNT_VAL(count, 12);
    };

    MI_S32 test_num = 1;
    for (MI_S32 i = 0; i < test_num; i++)
    {
        AURA_LOGI(g_ctx, AURA_TAG, "============================\n");
        AURA_LOGI(g_ctx, AURA_TAG, "   Test0: default config\n");
        AURA_LOGI(g_ctx, AURA_TAG, "============================\n");
        test0();

        AURA_LOGI(g_ctx, AURA_TAG, "======================================\n");
        AURA_LOGI(g_ctx, AURA_TAG, "   Test1: Set Thread Num as Negative\n");
        AURA_LOGI(g_ctx, AURA_TAG, "=====================================\n");
        test1();

        AURA_LOGI(g_ctx, AURA_TAG, "======================================\n");
        AURA_LOGI(g_ctx, AURA_TAG, "   Test2: Set Thread Num as Positive\n");
        AURA_LOGI(g_ctx, AURA_TAG, "=====================================\n");
        test2();

        AURA_LOGI(g_ctx, AURA_TAG, "======================================\n");
        AURA_LOGI(g_ctx, AURA_TAG, "   Test3: Set Thread Num Very Large\n");
        AURA_LOGI(g_ctx, AURA_TAG, "=====================================\n");
        test3();
    }

    AddTestResult(TestStatus::PASSED);
}