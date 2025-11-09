#include "aura/runtime/context.h"
#include "aura/tools/unit_test.h"

#include <memory>

using namespace aura;

NEW_TESTCASE(runtime_context_constructor_test)
{
    Status ret = Status::OK;

    Config cfg;
    cfg.SetLog(LogOutput::STDOUT, LogLevel::DEBUG, "log");
    cfg.SetWorkerPool("Aura", CpuAffinity::ALL, CpuAffinity::ALL);
    cfg.SetCLConf(DT_TRUE, "", "");

    std::shared_ptr<Context> ctx(new Context(cfg));
    if (DT_NULL == ctx)
    {
        AddTestResult(AURA_GET_TEST_STATUS(Status::ERROR));
        return;
    }

    ret |= ctx->Initialize();
    if (ret != Status::OK)
    {
        AddTestResult(AURA_GET_TEST_STATUS(ret));
        return;
    }

    AURA_CHECK_IEQ(ctx.get(), ctx->GetLogger(), static_cast<Logger*>(DT_NULL), "check Context::GetLogger() failed\n");
    AURA_CHECK_IEQ(ctx.get(), ctx->GetMemPool(), static_cast<MemPool*>(DT_NULL), "check Context::GetMemPool() failed\n");
#if !defined(AURA_BUILD_XPLORER)
    AURA_CHECK_IEQ(ctx.get(), ctx->GetWorkerPool(), static_cast<WorkerPool*>(DT_NULL), "check Context::GetWorkerPool() failed\n");
#endif // AURA_BUILD_XPLORER
#if defined(AURA_BUILD_ANDROID)
    AURA_CHECK_IEQ(ctx.get(), ctx->GetSystrace(), static_cast<Systrace*>(DT_NULL), "check Context::GetSystrace() failed\n");
#endif // AURA_BUILD_ANDROID
#if defined(AURA_ENABLE_OPENCL)
    AURA_CHECK_IEQ(ctx.get(), ctx->GetCLEngine(), static_cast<CLEngine*>(DT_NULL), "check Context::GetCLEngine() failed\n");
#endif // AURA_ENABLE_OPENCL
#if defined(AURA_ENABLE_HEXAGON)
    AURA_CHECK_IEQ(ctx.get(), ctx->GetHexagonEngine(), static_cast<HexagonEngine*>(DT_NULL), "check Context::HexagonEngine() failed\n");
#endif // AURA_ENABLE_HEXAGON
#if defined(AURA_ENABLE_XTENSA)
    AURA_CHECK_IEQ(ctx.get(), ctx->GetXtensaEngine(), static_cast<XtensaEngine*>(DT_NULL), "check Context::XtensaEngine() failed\n");
#endif // AURA_ENABLE_XTENSA
    AURA_LOGI(ctx.get(), AURA_TAG, "aura version : %s\n", ctx->GetVersion().c_str());

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}