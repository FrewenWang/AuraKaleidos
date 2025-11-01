#include "aura/runtime/core.h"
#include "aura/tools/unit_test.h"

using namespace aura;

template <typename Tp>
AURA_VOID PrintTest(MI_S32 id)
{
    constexpr MI_S32 total_num = AURA_HVLEN / sizeof(Tp);
    Tp src[total_num];
    for (MI_S32 i = 0; i < total_num; i++)
    {
        src[i] = i;
    }

    HVX_Vector v = vmemu(src);
    Q6_vprint_V<Tp>(v, id);
}

NEW_TESTCASE(runtime_hexagon_instructions_print_test)
{
    Status ret = Status::OK;

    PrintTest<MI_U8>(0);
    PrintTest<MI_S8>(1);
    PrintTest<MI_U16>(2);
    PrintTest<MI_S16>(3);
    PrintTest<MI_U32>(4);
    PrintTest<MI_S32>(5);
    PrintTest<MI_F32>(6);
    PrintTest<MI_U64>(7);
    PrintTest<MI_S64>(8);

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}