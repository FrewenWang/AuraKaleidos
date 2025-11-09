#include "device/include/hexagon_instructions_unit_test.hpp"

template <typename Tp>
Status SplatTest(Context *ctx)
{
    Status ret = Status::OK;

    constexpr DT_S32 total_num = AURA_HVLEN / sizeof(Tp);
    Tp ref[total_num], dst[total_num];
    for (DT_S32 i = 0; i < total_num; i++)
    {
        ref[i] = 1;
    }

    HVX_Vector v_splat = vsplat<Tp>(1);
    vmemu(dst) = v_splat;

    ret |= CHECK_CMP_VECTOR(ctx, dst, ref, 128);
    return ret;
}

NEW_TESTCASE(runtime_hexagon_instructions_splat_test)
{
    Status ret = Status::OK;
    Context *ctx = UnitTest::GetInstance()->GetContext();

    ret |= SplatTest<DT_U8>(ctx);
    ret |= SplatTest<DT_S8>(ctx);
    ret |= SplatTest<DT_U16>(ctx);
    ret |= SplatTest<DT_S16>(ctx);
    ret |= SplatTest<DT_U32>(ctx);
    ret |= SplatTest<DT_S32>(ctx);
    ret |= SplatTest<DT_F32>(ctx);

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}