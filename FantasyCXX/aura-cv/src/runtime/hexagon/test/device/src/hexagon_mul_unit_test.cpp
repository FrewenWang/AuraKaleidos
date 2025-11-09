#include "device/include/hexagon_instructions_unit_test.hpp"

NEW_TESTCASE(runtime_hexagon_instructions_mul_test)
{
    Status ret = Status::OK;
    Context *ctx = UnitTest::GetInstance()->GetContext();

    // Q6_Wd_vmul_VwVw
    {
        DT_S32 src_u[32] = {0x7fffffff, 0x7fffffff, (DT_S32)0x80000000, (DT_S32)0x80000000, 0x7fffffff, 0x7ffffffe};
        DT_S32 src_v[32] = {0x7fffffff, 0x7ffffffe, (DT_S32)0x80000000, 0x7fffffff, (DT_S32)0x80000000, (DT_S32)0x80000001};

        DT_S64 dst[32], ref[32];

        HVX_Vector vs32_u = vmemu(src_u);
        HVX_Vector vs32_v = vmemu(src_v);
        HVX_VectorPair ws64 = Q6_Wd_vmul_VwVw(vs32_u, vs32_v);
        ws64 = Q6_W_vshuff_VVR(Q6_V_hi_W(ws64), Q6_V_lo_W(ws64), -4);
        vmemu(dst) = Q6_V_lo_W(ws64);
        vmemu(dst + 16) = Q6_V_hi_W(ws64);

        for (DT_S32 i = 0; i < 32; i++)
        {
            ref[i] = (DT_S64)src_u[i] * (DT_S64)src_v[i];
        }

        ret |= CHECK_CMP_VECTOR(ctx, dst, ref, 256);
    }

    // Q6_Wud_vmul_VuwVuw
    {
        DT_U32 src_u[32] = {0xffffffff, 0xfffffffe, 0xffffffff, 0xffffffff, 1};
        DT_U32 src_v[32] = {0xffffffff, 0xffffffff, 0xfffffffe, 0, 0xffffffff};

        DT_U64 dst[32], ref[32];

        HVX_Vector vu32_u = vmemu(src_u);
        HVX_Vector vu32_v = vmemu(src_v);
        HVX_VectorPair wu64 = Q6_Wud_vmul_VuwVuw(vu32_u, vu32_v);
        wu64 = Q6_W_vshuff_VVR(Q6_V_hi_W(wu64), Q6_V_lo_W(wu64), -4);
        vmemu(dst) = Q6_V_lo_W(wu64);
        vmemu(dst + 16) = Q6_V_hi_W(wu64);

        for (DT_S32 i = 0; i < 32; i++)
        {
            ref[i] = (DT_U64)src_u[i] * (DT_U64)src_v[i];
        }

        ret |= CHECK_CMP_VECTOR(ctx, dst, ref, 256);
    }

    // Q6_Vw_vmul32xhi16_VwVw
    {
        DT_S32 src_u[32] = {0x7fffffff, (DT_S32)0x80000000, 0xffff, 0x10000, (DT_S32)0xfffefffe};
        DT_S16 src_v[64] = {0, 1, 0, 1, 0, 0x7fff, 0, (DT_S16)0x8000, 0, 0x7fff};

        DT_S32 dst[32], ref[32];

        HVX_Vector vs32_u = vmemu(src_u);
        HVX_Vector vs16_v = vmemu(src_v);
        HVX_Vector vs32 = Q6_Vw_vmul32xhi16_VwVw(vs32_u, vs16_v);
        vmemu(dst) = vs32;

        for (DT_S32 i = 0; i < 32; i++)
        {
            ref[i] = src_u[i] * src_v[i * 2 + 1];
        }

        ret |= CHECK_CMP_VECTOR(ctx, dst, ref, 128);
    }

    // Q6_Vw_vmul32xlo16_VwVuw
    {
        DT_S32 src_u[32] = {0x7fffffff, (DT_S32)0x80000000, 0x8000, (DT_S32)0xffff8000, (DT_S32)0x80000000};
        DT_U16 src_v[64] = {1, 0, 1, 0, 0xffff, 0, 0xffff, 0, 0, 0};

        DT_S32 dst[32], ref[32];

        HVX_Vector vs32_u = vmemu(src_u);
        HVX_Vector vu16_v = vmemu(src_v);
        HVX_Vector vs32 = Q6_Vw_vmul32xlo16_VwVuw(vs32_u, vu16_v);
        vmemu(dst) = vs32;

        for (DT_S32 i = 0; i < 32; i++)
        {
            ref[i] = src_u[i] * (DT_S32)src_v[i * 2];
        }

        ret |= CHECK_CMP_VECTOR(ctx, dst, ref, 128);
    }

    // Q6_Wud_vmul32xlo16_VuwVuw
    {
        DT_U32 src_u[32] = {0xffffffff, 0, 0xffffffff, 0xffffffff};
        DT_U16 src_v[64] = {0xffff, 0, 0, 0, 0xfffe, 0, 1, 0};

        DT_U64 dst[32], ref[32];

        HVX_Vector vu32_u = vmemu(src_u);
        HVX_Vector vu16_v = vmemu(src_v);
        HVX_VectorPair wu64 = Q6_Wud_vmul32xlo16_VuwVuw(vu32_u, vu16_v);
        wu64 = Q6_W_vshuff_VVR(Q6_V_hi_W(wu64), Q6_V_lo_W(wu64), -4);
        vmemu(dst) = Q6_V_lo_W(wu64);
        vmemu(dst + 16) = Q6_V_hi_W(wu64);

        for (DT_S32 i = 0; i < 32; i++)
        {
            ref[i] = (DT_U64)src_u[i] * (DT_U64)src_v[i * 2];
        }

        ret |= CHECK_CMP_VECTOR(ctx, dst, ref, 256);
    }

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}