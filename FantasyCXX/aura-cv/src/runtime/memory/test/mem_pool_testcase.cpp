#include "aura/runtime/memory.h"
#include "aura/tools/unit_test.h"

using namespace aura;

using namespace aura;

struct BufferMembers
{
    BufferMembers(DT_BOOL valid, DT_S32 type,
                  DT_S64 capacity, DT_S64 size)
                  : valid(valid), type(type),
                    capacity(capacity), size(size)
    {}

    DT_BOOL valid;
    DT_S32  type;
    DT_S64  capacity;
    DT_S64  size;
};

static Status CheckBuffer(Context *ctx, const Buffer &src, const BufferMembers &buffer_members,
                          const DT_CHAR *file, const DT_CHAR *func, DT_S32 line)
{
    Status ret = Status::OK;
    ret |= TestCheckEQ(ctx, src.IsValid(), buffer_members.valid, "check Buffer::IsValid() failed\n", file, func, line);
    ret |= TestCheckEQ(ctx, src.m_type, buffer_members.type, "check Buffer::m_type failed\n", file, func, line);
    ret |= TestCheckEQ(ctx, src.m_capacity, buffer_members.capacity, "check Buffer::m_capacity failed\n", file, func, line);
    ret |= TestCheckEQ(ctx, src.m_size, buffer_members.size, "check Buffer::m_size failed\n", file, func, line);
    return ret;
}

#define CHECK_BUFFER(ctx, src, ref)    CheckBuffer(ctx, src, ref, __FILE__, __FUNCTION__, __LINE__)

NEW_TESTCASE(runtime_mem_pool_allocate_test)
{
    Status ret = Status::OK;
    Context *ctx = UnitTest::GetInstance()->GetContext();

    MemPool *mp = ctx->GetMemPool();

    DT_VOID *ptr = mp->Allocate(AURA_MEM_INVALID, 100, 128, __FILE__, __FUNCTION__, __LINE__);
    auto buffer = mp->GetBuffer(ptr);
    ret |= AURA_CHECK_EQ(ctx, buffer.IsValid(), DT_FALSE, "check Buffer::IsValid() failed\n");
    mp->Free(ptr);

    ptr = mp->Allocate(AURA_MEM_HEAP, 100, 0, __FILE__, __FUNCTION__, __LINE__);
    buffer = mp->GetBuffer(ptr);
    ret |= CHECK_BUFFER(ctx, buffer, BufferMembers(DT_TRUE, AURA_MEM_HEAP, 100, 100));
    mp->Free(ptr);

    ptr = mp->Allocate(AURA_MEM_HEAP, 0, 0, __FILE__, __FUNCTION__, __LINE__);
    buffer = mp->GetBuffer(ptr);
    ret |= AURA_CHECK_EQ(ctx, buffer.IsValid(), DT_FALSE, "check Buffer::IsValid() failed\n");
    mp->Free(ptr);

#if defined(AURA_BUILD_HEXAGON)
    ptr = mp->Allocate(AURA_MEM_VTCM, 128, 128, __FILE__, __FUNCTION__, __LINE__);
    buffer = mp->GetBuffer(ptr);
    ret |= CHECK_BUFFER(ctx, buffer, BufferMembers(DT_TRUE, AURA_MEM_VTCM, 128, 128));
    mp->Free(ptr);
#endif // AURA_BUILD_HEXAGON

#if defined(AURA_BUILD_ANDROID)
    ptr = mp->Allocate(AURA_MEM_DMA_BUF_HEAP, 100, 128, __FILE__, __FUNCTION__, __LINE__);
    buffer = mp->GetBuffer(ptr);
    ret |= CHECK_BUFFER(ctx, buffer, BufferMembers(DT_TRUE, AURA_MEM_DMA_BUF_HEAP, 100, 100));
    mp->Free(ptr);
#endif // AURA_BUILD_ANDROID

#if defined(AURA_ENABLE_OPENCL)
    ptr = mp->Allocate(AURA_MEM_SVM, 100, 128, __FILE__, __FUNCTION__, __LINE__);
    buffer = mp->GetBuffer(ptr);
    ret |= CHECK_BUFFER(ctx, buffer, BufferMembers(DT_TRUE, AURA_MEM_SVM, 100, 100));
    mp->Free(ptr);
#endif // AURA_ENABLE_OPENCL

    DT_VOID *ptr0 = AURA_ALLOC(ctx, 100);
    DT_VOID *ptr1 = AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, 100, 128);
    Buffer buffer0 = ctx->GetMemPool()->GetBuffer(ptr0);
    Buffer buffer1 = ctx->GetMemPool()->GetBuffer(ptr1);
#if defined(AURA_BUILD_ANDROID)
    DT_S32 type = AURA_MEM_DMA_BUF_HEAP;
#else // AURA_BUILD_ANDROID
    DT_S32 type = AURA_MEM_HEAP;
#endif // AURA_BUILD_ANDROID
    ret |= CHECK_BUFFER(ctx, buffer0, BufferMembers(DT_TRUE, type, 100, 100));
    ret |= CHECK_BUFFER(ctx, buffer1, BufferMembers(DT_TRUE, AURA_MEM_HEAP, 100, 100));
    ret |= AURA_CHECK_EQ(ctx, AURA_FREE(ctx, ptr0), Status::OK, "check AURA_FREE() failed\n");
    ret |= AURA_CHECK_EQ(ctx, AURA_FREE(ctx, ptr1), Status::OK, "check AURA_FREE() failed\n");
    ret |= AURA_CHECK_EQ(ctx, AURA_FREE(ctx, ptr1), Status::ERROR, "check AURA_FREE() failed\n");

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}

NEW_TESTCASE(runtime_mem_pool_map_test)
{
    Status ret = Status::OK;
    Context *ctx = UnitTest::GetInstance()->GetContext();

    MemPool *mp = ctx->GetMemPool();

    Buffer buffer;
    ret |= AURA_CHECK_EQ(ctx, mp->Map(buffer), Status::ERROR, "check MemPool::Map() failed\n");
    ret |= AURA_CHECK_EQ(ctx, mp->Unmap(buffer), Status::ERROR, "check MemPool::Unmap() failed\n");

    DT_VOID *ptr = mp->Allocate(AURA_MEM_HEAP, 100, 128, __FILE__, __FUNCTION__, __LINE__);
    buffer = mp->GetBuffer(ptr);
    ret |= AURA_CHECK_EQ(ctx, mp->Map(buffer), Status::OK, "check MemPool::Map() failed\n");
    ret |= AURA_CHECK_EQ(ctx, mp->Unmap(buffer), Status::OK, "check MemPool::Unmap() failed\n");
    ret |= AURA_CHECK_EQ(ctx, mp->Map(buffer), Status::OK, "check MemPool::Map() failed\n");
    ret |= AURA_CHECK_EQ(ctx, mp->Map(buffer), Status::OK, "check MemPool::Map() failed\n");
    ret |= AURA_CHECK_EQ(ctx, mp->Unmap(buffer), Status::OK, "check MemPool::Unmap() failed\n");
    ret |= AURA_CHECK_EQ(ctx, mp->Unmap(buffer), Status::OK, "check MemPool::Unmap() failed\n");
    mp->Free(ptr);

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}

class TestOp
{
public:
    TestOp(Context* ctx, DT_S32 a, DT_F32 b) : m_ctx(ctx), m_a(a), m_b(b)
    {
        AURA_LOGD(m_ctx, AURA_TAG, "TestOp Constructor with %d, %f\n", a, b);
    }

    ~TestOp()
    {
        AURA_LOGD(m_ctx, AURA_TAG, "TestOp Destructor\n");
    }

    Context *m_ctx;
    DT_S32 m_a;
    DT_F32 m_b;
};

NEW_TESTCASE(create_op_test)
{
    Context *ctx = UnitTest::GetInstance()->GetContext();

    TestOp *op = Create<TestOp>(ctx, 1, 2.0f);
    Delete(ctx, &op);
}

NEW_TESTCASE(runtime_mem_pool_mem_stat_in_order)
{
    Context *ctx = UnitTest::GetInstance()->GetContext();
    ctx->GetMemPool()->MemTraceSet(DT_TRUE);
    ctx->GetMemPool()->MemTraceBegin("SampleAlgo");
    ctx->GetMemPool()->MemTraceBegin("SectionA");
    {
        auto ptr1 = AURA_ALLOC(ctx, 1024);
        AURA_LOGI(ctx, AURA_TAG, "SectionA allocate 1024 bytes, addr: %p.\n", ptr1);
        AURA_FREE(ctx, ptr1);
    }
    ctx->GetMemPool()->MemTraceEnd("SectionA");

    ctx->GetMemPool()->MemTraceBegin("SectionB");
    {
        auto ptr1 = AURA_ALLOC(ctx, 4096);
        AURA_LOGI(ctx, AURA_TAG, "SectionB allocate 4096 bytes, addr: %p.\n", ptr1);
        AURA_FREE(ctx, ptr1);
    }
    ctx->GetMemPool()->MemTraceEnd("SectionB");
    
    ctx->GetMemPool()->MemTraceBegin("SectionC");
    {
        auto ptr1 = AURA_ALLOC(ctx, 2048);
        AURA_LOGI(ctx, AURA_TAG, "SectionC allocate 2048 bytes, addr: %p.\n", ptr1);
        AURA_FREE(ctx, ptr1);
    }
    ctx->GetMemPool()->MemTraceEnd("SectionC");

    ctx->GetMemPool()->MemTraceEnd("SampleAlgo");
    std::cout << ctx->GetMemPool()->MemTraceReport() << std::endl;
    ctx->GetMemPool()->MemTraceSet(DT_FALSE);
}

DT_VOID FuncA(Context *ctx)
{
    ctx->GetMemPool()->MemTraceBegin("FuncA");
    auto ptr1 = AURA_ALLOC(ctx, 1024);
    AURA_LOGI(ctx, AURA_TAG, "Func A allocate 1024 bytes, addr: %p.\n", ptr1);
    AURA_FREE(ctx, ptr1);
    ctx->GetMemPool()->MemTraceEnd("FuncA");
}

DT_VOID FuncB(Context *ctx)
{
    ctx->GetMemPool()->MemTraceBegin("FuncB");
    auto ptr1 = AURA_ALLOC(ctx, 4096);
    AURA_LOGI(ctx, AURA_TAG, "Func B allocate 4096 bytes, addr: %p.\n", ptr1);
    AURA_FREE(ctx, ptr1);
    ctx->GetMemPool()->MemTraceEnd("FuncB");
}

DT_VOID FuncC(Context *ctx)
{
    ctx->GetMemPool()->MemTraceBegin("FuncC");
    auto ptr1 = AURA_ALLOC(ctx, 2048);
    AURA_LOGI(ctx, AURA_TAG, "Func C allocate 2048 bytes, addr: %p.\n", ptr1);
    FuncA(ctx);
    FuncB(ctx);
    AURA_FREE(ctx, ptr1);
    ctx->GetMemPool()->MemTraceEnd("FuncC");
}

DT_VOID FuncD(Context *ctx)
{
    ctx->GetMemPool()->MemTraceBegin("FuncD");
    auto ptr1 = AURA_ALLOC(ctx, 2048);
    AURA_LOGI(ctx, AURA_TAG, "Func D allocate 2048 bytes, addr: %p.\n", ptr1);
    AURA_FREE(ctx, ptr1);
    ctx->GetMemPool()->MemTraceEnd("FuncD");
}

NEW_TESTCASE(runtime_mem_pool_mem_stat_recursive)
{
    Context *ctx = UnitTest::GetInstance()->GetContext();
    ctx->GetMemPool()->MemTraceSet(DT_TRUE);

    ctx->GetMemPool()->MemTraceBegin("Algorithm");

    ctx->GetMemPool()->MemTraceBegin("ModuleA");
        FuncD(ctx);
        FuncC(ctx);
    ctx->GetMemPool()->MemTraceEnd("ModuleA");

    ctx->GetMemPool()->MemTraceBegin("ModuleB");
        ctx->GetMemPool()->MemTraceBegin("CreateOp");
            TestOp *op = Create<TestOp>(ctx, 1, 2.0f);
            Delete(ctx, &op);
        ctx->GetMemPool()->MemTraceEnd("CreateOp");

        ctx->GetMemPool()->MemTraceBegin("EmptyFunc");
        ctx->GetMemPool()->MemTraceEnd("EmptyFunc");
    ctx->GetMemPool()->MemTraceEnd("ModuleB");

    ctx->GetMemPool()->MemTraceBegin("ModuleC");
        ctx->GetMemPool()->MemTraceBegin("AlolocateDmaBuf");
#if defined(AURA_BUILD_ANDROID)
            auto ptr = ctx->GetMemPool()->Allocate(AURA_MEM_DMA_BUF_HEAP, 100, 128, __FILE__, __FUNCTION__, __LINE__);
            Buffer buffer = ctx->GetMemPool()->GetBuffer(ptr);
            ctx->GetMemPool()->Free(ptr);
#endif // AURA_BUILD_ANDROID
        ctx->GetMemPool()->MemTraceEnd("AlolocateDmaBuf");
    ctx->GetMemPool()->MemTraceEnd("ModuleC");
    ctx->GetMemPool()->MemTraceEnd("Algorithm");

    std::cout << ctx->GetMemPool()->MemTraceReport() << std::endl;
    ctx->GetMemPool()->MemTraceSet(DT_FALSE);

    AddTestResult(AURA_GET_TEST_STATUS(Status::OK));
}

NEW_TESTCASE(runtime_mem_pool_mem_stat_halfway)
{
    Context *ctx = UnitTest::GetInstance()->GetContext();
    ctx->GetMemPool()->MemTraceSet(DT_TRUE);

    ctx->GetMemPool()->MemTraceBegin("Algorithm");

    ctx->GetMemPool()->MemTraceBegin("ModuleA");
        FuncD(ctx);
        FuncC(ctx);
    ctx->GetMemPool()->MemTraceEnd("ModuleA");

    std::cout << ctx->GetMemPool()->MemTraceReport() << std::endl;

    ctx->GetMemPool()->MemTraceEnd("Algorithm");
    std::cout << ctx->GetMemPool()->MemTraceReport() << std::endl;

    ctx->GetMemPool()->MemTraceSet(DT_FALSE);
    AddTestResult(AURA_GET_TEST_STATUS(Status::OK));
}

NEW_TESTCASE(runtime_mem_pool_mem_stat_dismatch)
{
    Context *ctx = UnitTest::GetInstance()->GetContext();
    ctx->GetMemPool()->MemTraceSet(DT_TRUE);

    ctx->GetMemPool()->MemTraceBegin("ModuleB");
        FuncD(ctx);
    ctx->GetMemPool()->MemTraceBegin("ModuleA");
        FuncC(ctx);
    ctx->GetMemPool()->MemTraceEnd("ModuleB");
    ctx->GetMemPool()->MemTraceBegin("ModuleA");

    std::cout << ctx->GetMemPool()->MemTraceReport() << std::endl;

    ctx->GetMemPool()->MemTraceSet(DT_FALSE);
    AddTestResult(AURA_GET_TEST_STATUS(Status::OK));
}

NEW_TESTCASE(runtime_mem_pool_mem_stat_reset)
{
    Context *ctx = UnitTest::GetInstance()->GetContext();
    ctx->GetMemPool()->MemTraceSet(DT_TRUE);

    ctx->GetMemPool()->MemTraceBegin("ModuleB");
        FuncD(ctx);
    ctx->GetMemPool()->MemTraceEnd("ModuleA");

    std::cout << ctx->GetMemPool()->MemTraceReport() << std::endl;

    // reset
    ctx->GetMemPool()->MemTraceClear();
    ctx->GetMemPool()->MemTraceBegin("ModuleA");
        FuncC(ctx);
    ctx->GetMemPool()->MemTraceEnd("ModuleA");
    std::cout << ctx->GetMemPool()->MemTraceReport() << std::endl;

    ctx->GetMemPool()->MemTraceSet(DT_FALSE);
    AddTestResult(AURA_GET_TEST_STATUS(Status::OK));
}

NEW_TESTCASE(runtime_mem_pool_mem_stat_empty)
{
    Context *ctx = UnitTest::GetInstance()->GetContext();
    ctx->GetMemPool()->MemTraceSet(DT_TRUE);

    ctx->GetMemPool()->MemTraceBegin("ModuleA");

    std::cout << ctx->GetMemPool()->MemTraceReport() << std::endl;

    ctx->GetMemPool()->MemTraceSet(DT_FALSE);
    AddTestResult(AURA_GET_TEST_STATUS(Status::OK));
}

NEW_TESTCASE(runtime_mem_pool_mem_stat_duplicate)
{
    Context *ctx = UnitTest::GetInstance()->GetContext();
    ctx->GetMemPool()->MemTraceSet(DT_TRUE);

    ctx->GetMemPool()->MemTraceBegin("ModuleA");
        FuncC(ctx);
    ctx->GetMemPool()->MemTraceBegin("ModuleA");
        FuncD(ctx);
    ctx->GetMemPool()->MemTraceBegin("ModuleA");

    ctx->GetMemPool()->MemTraceEnd("ModuleA");
    ctx->GetMemPool()->MemTraceEnd("ModuleA");
    ctx->GetMemPool()->MemTraceEnd("ModuleA");

    std::cout << ctx->GetMemPool()->MemTraceReport() << std::endl;

    ctx->GetMemPool()->MemTraceSet(DT_FALSE);
    AddTestResult(AURA_GET_TEST_STATUS(Status::OK));
}