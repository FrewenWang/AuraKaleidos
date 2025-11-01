#include "device/include/hexagon_instructions_unit_test.hpp"

NEW_TESTCASE(runtime_hexagon_instructions_memcpy_test)
{
    Status ret = Status::OK;
    Context *ctx = UnitTest::GetInstance()->GetContext();

    MI_S32 len = 1024 * 1024;
    MI_U8 *src = static_cast<MI_U8*>(AURA_ALLOC(ctx, len));
    MI_U8 *dst = static_cast<MI_U8*>(AURA_ALLOC(ctx, len));
    MI_U8 *ref = static_cast<MI_U8*>(AURA_ALLOC(ctx, len));

    for (MI_S32 i = 0; i < len; i++)
    {
        src[i] = i;
    }

    auto func_memcpy = [=]() -> Status
    {
        memcpy(ref, src, len);
        return Status::OK;
    };

    auto func_memcpy_asm = [=]() -> Status
    {
        AuraMemCopy(dst, src, len);
        return Status::OK;
    };

    TestTime time, time_asm;
    Executor(10, 2, time, func_memcpy);
    Executor(10, 2, time_asm, func_memcpy_asm);
    AURA_LOGI(ctx, AURA_TAG, "memcpy cost time: (%s), AuraMemCopy cost time: (%s)\n", time.ToString().c_str(), time_asm.ToString().c_str());

    ret |= CHECK_CMP_VECTOR(ctx, ref, dst, len);

    AURA_FREE(ctx, src);
    AURA_FREE(ctx, dst);
    AURA_FREE(ctx, ref);

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}