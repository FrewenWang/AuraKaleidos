#include "aura/runtime/core.h"
#include "aura/tools/unit_test.h"

using namespace aura;

template <typename Tp>
DT_VOID PrintTest(DT_S32 id)
{
    constexpr DT_S32 total_num = AURA_HVLEN / sizeof(Tp);
    Tp src[total_num];
    for (DT_S32 i = 0; i < total_num; i++)
    {
        src[i] = i;
    }

    HVX_Vector v = vmemu(src);
    Q6_vprint_V<Tp>(v, id);
}

NEW_TESTCASE(runtime_hexagon_instructions_print_test)
{
    Status ret = Status::OK;

    PrintTest<DT_U8>(0);
    PrintTest<DT_S8>(1);
    PrintTest<DT_U16>(2);
    PrintTest<DT_S16>(3);
    PrintTest<DT_U32>(4);
    PrintTest<DT_S32>(5);
    PrintTest<DT_F32>(6);
    PrintTest<DT_U64>(7);
    PrintTest<DT_S64>(8);

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}