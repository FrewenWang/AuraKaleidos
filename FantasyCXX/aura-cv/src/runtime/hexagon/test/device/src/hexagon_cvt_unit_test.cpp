#include "device/include/hexagon_instructions_unit_test.hpp"

NEW_TESTCASE(runtime_hexagon_instructions_cvt_test)
{
    Status ret = Status::OK;
    Context *ctx = UnitTest::GetInstance()->GetContext();

    // Q6_Vsf_vcvt_Vuw
    {
        DT_U32 src[32] = {0, 1, 100, 10000, 1000000, 100000000};
        DT_F32 dst[32], ref[32];

        HVX_Vector vu32 = vmemu(src);
        HVX_Vector vf32 = Q6_Vsf_vcvt_Vuw(vu32);
        vmemu(dst) = vf32;

        for (DT_S32 i = 0; i < 32; i++)
        {
            ref[i] = (DT_F32)src[i];
        }

        ret |= CHECK_CMP_VECTOR(ctx, dst, ref, 128);
    }

    // Q6_Vsf_vcvt_Vw
    {
        DT_S32 src[32] = {-100000000, -1000000, -10000, -100, -1, 0, 1, 100, 10000, 1000000, 100000000};
        DT_F32 dst[32], ref[32];

        HVX_Vector vs32 = vmemu(src);
        HVX_Vector vf32 = Q6_Vsf_vcvt_Vw(vs32);
        vmemu(dst) = vf32;

        for (DT_S32 i = 0; i < 32; i++)
        {
            ref[i] = (DT_F32)src[i];
        }

        ret |= CHECK_CMP_VECTOR(ctx, dst, ref, 128);
    }

    // Q6_Vuw_vcvt_Vsf
    {
        DT_F32 src[32] = {0.0, 1.0, 1.1, 100.5, 10000.1, 1000000.99, 100000000.123456};
        DT_U32 dst[32], ref[32];

        HVX_Vector vf32 = vmemu(src);
        HVX_Vector vu32 = Q6_Vuw_vcvt_Vsf(vf32);
        vmemu(dst) = vu32;

        for (DT_S32 i = 0; i < 32; i++)
        {
            ref[i] = (DT_U32)src[i];
        }

        ret |= CHECK_CMP_VECTOR(ctx, dst, ref, 128);
    }

    // Q6_Vw_vcvt_Vsf
    {
        DT_F32 src[32] = {-100000000.123456, -1000000.99, -10000.1, -100.5, -1.1, 0.0, 1.0, 1.1, 100.5, 10000.1, 1000000.99, 100000000.123456};
        DT_S32 dst[32], ref[32];

        HVX_Vector vf32 = vmemu(src);
        HVX_Vector vs32 = Q6_Vw_vcvt_Vsf(vf32);
        vmemu(dst) = vs32;

        for (DT_S32 i = 0; i < 32; i++)
        {
            ref[i] = (DT_S32)src[i];
        }

        ret |= CHECK_CMP_VECTOR(ctx, dst, ref, 128);
    }

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}