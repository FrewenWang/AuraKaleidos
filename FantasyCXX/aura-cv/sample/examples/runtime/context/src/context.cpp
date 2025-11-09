#include "sample_runtime.hpp"

DT_S32 main()
{
    if (!aura::Context::IsPlatformSupported())
    {
        return -1;
    }

    // create context for sample
    std::shared_ptr<aura::Context> ctx = CreateContext();
    if (nullptr == ctx)
    {
        return -1;
    }

    AURA_LOGI(ctx, SAMPLE_TAG, "=================== aura::Context Sample Test Begin ===================\n");

    aura::Logger *log = ctx->GetLogger();
    if (DT_NULL == log)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "Context::GetLogger() failed\n");
        return -1;
    }
    else
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "Context::GetLogger() successed\n");
    }

    aura::MemPool *mem_pool = ctx->GetMemPool();
    if (DT_NULL == mem_pool)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "Context::GetMemPool() failed\n");
        return -1;
    }
    else
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "Context::GetMemPool() successed\n");
    }

    aura::WorkerPool *work_pool = ctx->GetWorkerPool();
    if (DT_NULL == work_pool)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "Context::GetWorkerPool() failed\n");
        return -1;
    }
    else
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "Context::GetWorkerPool() successed\n");
    }

#if defined(AURA_BUILD_ANDROID)
    aura::Systrace *sys_trace = ctx->GetSystrace();
    if (DT_NULL == sys_trace)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "Context::GetSystrace() failed\n");
        return -1;
    }
    else
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "Context::GetSystrace() successed\n");
    }
#endif // AURA_BUILD_ANDROID

#if defined(AURA_ENABLE_OPENCL)
    aura::CLEngine *cl_engine = ctx->GetCLEngine();
    if (DT_NULL == cl_engine)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "Context::GetCLEngine() failed\n");
        return -1;
    }
    else
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "Context::GetCLEngine() successed\n");
    }
#endif // AURA_ENABLE_OPENCL

#if defined(AURA_ENABLE_HEXAGON)
    aura::HexagonEngine *hex_engine = ctx->GetHexagonEngine();
    if (DT_NULL == hex_engine)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "Context::GetHexagonEngine() failed\n");
        return -1;
    }
    else
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "Context::GetHexagonEngine() successed\n");
    }
#endif // AURA_ENABLE_HEXAGON

    AURA_LOGD(ctx, SAMPLE_TAG, "=================== Context Sample Test Successed ===================\n");

    return 0;
}