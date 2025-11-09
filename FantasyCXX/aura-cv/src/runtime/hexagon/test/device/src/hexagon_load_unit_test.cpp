#include "device/include/hexagon_instructions_unit_test.hpp"

template <typename Tp, DT_S32 C>
Status LoadStoreTest(Context *ctx)
{
    using MVType = typename MVHvxVector<C>::Type;

    Status ret = Status::OK;

    constexpr DT_S32 total_num = AURA_HVLEN / sizeof(Tp) * C;
    Tp src[total_num], dst[total_num];
    for (DT_S32 i = 0; i < total_num; i++)
    {
        src[i] = i;
    }

    MVType mv;
    vload(src, mv);
    for (DT_S32 ch = 0; ch < C; ch++)
    {
        constexpr DT_S32 num = AURA_HVLEN / sizeof(Tp);
        Tp dst_tmp[num], ref_tmp[num];
        vmemu(dst_tmp) = mv.val[ch];

        for (DT_S32 i = 0; i < num; i++)
        {
            ref_tmp[i] = C * i + ch;
        }

        ret |= CHECK_CMP_VECTOR(ctx, dst_tmp, ref_tmp, AURA_HVLEN);
    }
    vstore(dst, mv);

    ret |= CHECK_CMP_VECTOR(ctx, src, dst, AURA_HVLEN * C);
    return ret;
}

NEW_TESTCASE(runtime_hexagon_instructions_load_test)
{
    Status ret = Status::OK;
    Context *ctx = UnitTest::GetInstance()->GetContext();

    ret |= LoadStoreTest<DT_U8, 1>(ctx);
    ret |= LoadStoreTest<DT_U8, 2>(ctx);
    ret |= LoadStoreTest<DT_U8, 3>(ctx);
    ret |= LoadStoreTest<DT_S8, 1>(ctx);
    ret |= LoadStoreTest<DT_S8, 2>(ctx);
    ret |= LoadStoreTest<DT_S8, 3>(ctx);
    ret |= LoadStoreTest<DT_U16, 1>(ctx);
    ret |= LoadStoreTest<DT_U16, 2>(ctx);
    ret |= LoadStoreTest<DT_U16, 3>(ctx);
    ret |= LoadStoreTest<DT_S16, 1>(ctx);
    ret |= LoadStoreTest<DT_S16, 2>(ctx);
    ret |= LoadStoreTest<DT_S16, 3>(ctx);

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}