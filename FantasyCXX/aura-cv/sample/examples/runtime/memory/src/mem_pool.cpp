#include "sample_runtime.hpp"

DT_S32 main()
{
    // create context for sample
    std::shared_ptr<aura::Context> ctx = CreateContext();
    if (nullptr == ctx)
    {
        return -1;
    }

    aura::Status ret = aura::Status::OK;

    AURA_LOGI(ctx, SAMPLE_TAG, "=================== aura::MemPool Sample Test Begin ===================\n");

    aura::MemPool *mem_pool = ctx->GetMemPool();
    if (DT_NULL == mem_pool)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "Context::GetMemPool() failed\n");
        return -1;
    }
    else
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "Context::GetMemPool() succeeded\n");
    }

    // allocate and free
    DT_VOID *ptr = mem_pool->Allocate(AURA_MEM_HEAP, 100, 0, __FILE__, __FUNCTION__, __LINE__);
    auto buffer  = mem_pool->GetBuffer(ptr);
    if (buffer.IsValid())
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "MemPool::Allocate() succeeded, memory type is AURA_MEM_HEAP \n");
    }
    else
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "MemPool::Allocate() failed\n");
        return -1;
    }

    mem_pool->Free(ptr);

#if defined(AURA_BUILD_HEXAGON)
    ptr    = mem_pool->Allocate(AURA_MEM_VTCM, 128, 128, __FILE__, __FUNCTION__, __LINE__);
    buffer = mem_pool->GetBuffer(ptr);
    if (buffer.IsValid())
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "MemPool::Allocate() succeeded, memory type is AURA_MEM_VTCM \n");
    }
    else
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "MemPool::Allocate() failed\n");
        return -1;
    }

    mem_pool->Free(ptr);
#endif // AURA_BUILD_HEXAGON

#if defined(AURA_BUILD_ANDROID)
    ptr    = mem_pool->Allocate(AURA_MEM_DMA_BUF_HEAP, 100, 128, __FILE__, __FUNCTION__, __LINE__);
    buffer = mem_pool->GetBuffer(ptr);
    if (buffer.IsValid())
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "MemPool::Allocate() succeeded, memory type is AURA_MEM_DMA_BUF_HEAP \n");
    }
    else
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "MemPool::Allocate() failed\n");
        return -1;
    }

    mem_pool->Free(ptr);
#endif // AURA_BUILD_ANDROID

#if defined(AURA_ENABLE_OPENCL)
    ptr    = mem_pool->Allocate(AURA_MEM_SVM, 100, 128, __FILE__, __FUNCTION__, __LINE__);
    buffer = mem_pool->GetBuffer(ptr);
    if (buffer.IsValid())
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "MemPool::Allocate() succeeded, memory type is AURA_MEM_SVM \n");
    }
    else
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "MemPool::Allocate() failed\n");
        return -1;
    }

    mem_pool->Free(ptr);
#endif // AURA_ENABLE_OPENCL

    // AURA_ALLOC
    ptr = AURA_ALLOC(ctx.get(), 100);
    buffer = ctx->GetMemPool()->GetBuffer(ptr);
    if (buffer.IsValid())
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "AURA_ALLOC succeeded \n");
    }
    else
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "AURA_ALLOC failed\n");
        return -1;
    }

    AURA_FREE(ctx.get(), ptr);

    // Map
    ptr    = mem_pool->Allocate(AURA_MEM_HEAP, 100, 128, __FILE__, __FUNCTION__, __LINE__);
    buffer = mem_pool->GetBuffer(ptr);
    ret    = mem_pool->Map(buffer);
    if (aura::Status::OK == ret)
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "MemPool::Map() succeeded\n");
    }
    else
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "MemPool::Map() failed\n");
        return -1;
    }

    ret = mem_pool->Unmap(buffer);
    if (aura::Status::OK == ret)
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "MemPool::Unmap() succeeded\n");
    }
    else
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "MemPool::Unmap() failed\n");
        return -1;
    }

    mem_pool->Free(ptr);

    AURA_LOGD(ctx, SAMPLE_TAG, "=================== MemPool Sample Test Succeeded ===================\n");

    return 0;
}