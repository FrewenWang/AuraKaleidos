#include "aura/runtime/context.h"
#include "aura/tools/unit_test.h"

#include <memory>

using namespace aura;

NEW_TESTCASE(runtime_context_constructor_test)
{
    Status ret = Status::OK;

    std::shared_ptr<Context> ctx(new Context());

    ret |= ctx->Initialize(LogOutput::STDOUT, LogLevel::DEBUG, "./log.txt");
    if (ret != Status::OK)
    {
        AddTestResult(AURA_GET_TEST_STATUS(ret));
        return;
    }

    AURA_CHECK_IEQ(ctx.get(), ctx->GetLogger(),     static_cast<Logger*>(MI_NULL),     "check Context::GetLogger() failed\n");
    AURA_CHECK_IEQ(ctx.get(), ctx->GetMemPool(),    static_cast<MemPool*>(MI_NULL),    "check Context::GetMemPool() failed\n");
    AURA_CHECK_IEQ(ctx.get(), ctx->GetWorkerPool(), static_cast<WorkerPool*>(MI_NULL), "check Context::GetWorkerPool() failed\n");

    AURA_LOGI(ctx.get(), AURA_TAG, "aura version : %s\n", ctx->GetVersion().c_str());

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}