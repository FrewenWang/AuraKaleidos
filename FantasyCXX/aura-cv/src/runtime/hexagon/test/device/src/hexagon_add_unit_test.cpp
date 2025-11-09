#include "device/include/hexagon_instructions_unit_test.hpp"

NEW_TESTCASE(runtime_hexagon_instructions_add_test)
{
    Status ret = Status::OK;
    Context *ctx = UnitTest::GetInstance()->GetContext();

    // Q6_Wud_vadd_VuwVuw
    {
        DT_U32 src_u[32] = {0xffffffff, 0xFFFFFFFE, 0, 0xffffffff, 1, 1};
        DT_U32 src_v[32] = {0xffffffff, 0xffffffff, 0xfffffffe, 0, 0xffffffff, 0xfffffffe};

        DT_U64 ref[32];
        DT_U64 dst[32];

        HVX_Vector vu32_u = vmemu(src_u);
        HVX_Vector vu32_v = vmemu(src_v);
        HVX_VectorPair wu64 = Q6_Wud_vadd_VuwVuw(vu32_u, vu32_v);
        wu64 = Q6_W_vshuff_VVR(Q6_V_hi_W(wu64), Q6_V_lo_W(wu64), -4);
        vmemu(dst) = Q6_V_lo_W(wu64);
        vmemu(dst + 16) = Q6_V_hi_W(wu64);

        for (DT_S32 i = 0; i < 32; i++)
        {
            ref[i] = static_cast<DT_U64>(src_u[i]) + static_cast<DT_U64>(src_v[i]);
        }

        ret |= CHECK_CMP_VECTOR(ctx, dst, ref, 256);
    }

    // Q6_Wd_vadd_VwVw
    {
        DT_S32 src_u[32] = {(DT_S32)0x80000000, 0x7fffffff, 0x7fffffff, 0, 0x7fffffff, (DT_S32)0x80000001};
        DT_S32 src_v[32] = {(DT_S32)0x80000000, 0x7fffffff, (DT_S32)0x80000000, (DT_S32)0x80000000, 0, (DT_S32)0x80000000};

        DT_S64 ref[32];
        DT_S64 dst[32];

        HVX_Vector vs32_u = vmemu(src_u);
        HVX_Vector vs32_v = vmemu(src_v);
        HVX_VectorPair ws64 = Q6_Wd_vadd_VwVw(vs32_u, vs32_v);
        ws64 = Q6_W_vshuff_VVR(Q6_V_hi_W(ws64), Q6_V_lo_W(ws64), -4);
        vmemu(dst) = Q6_V_lo_W(ws64);
        vmemu(dst + 16) = Q6_V_hi_W(ws64);

        for (DT_S32 i = 0; i < 32; i++)
        {
            ref[i] = static_cast<DT_S64>(src_u[i]) + static_cast<DT_S64>(src_v[i]);
        }

        ret |= CHECK_CMP_VECTOR(ctx, dst, ref, 256);
    }

    // Q6_Wd_vadd_WdWd
    {
        DT_S64 src_u[32] = {0x80000000, 0x7fffffff, 0x7fffffff, (DT_S64)0x8000000000000000, 0x3fffffffffffffff};
        DT_S64 src_v[32] = {0x80000000, 0x7fffffff, 0x80000000, 0x7fffffffffffffff, (DT_S64)0x4000000000000000};

        DT_S64 ref[32];
        DT_S64 dst[32];

        HVX_Vector vs64_u0 = vmemu(src_u);
        HVX_Vector vs64_u1 = vmemu(src_u + 16);
        HVX_Vector vs64_v0 = vmemu(src_v);
        HVX_Vector vs64_v1 = vmemu(src_v + 16);

        HVX_VectorPair ws64_u = Q6_W_vdeal_VVR(vs64_u1, vs64_u0, -4);
        HVX_VectorPair ws64_v = Q6_W_vdeal_VVR(vs64_v1, vs64_v0, -4);
        HVX_VectorPair ws64 = Q6_Wd_vadd_WdWd(ws64_u, ws64_v);
        ws64 = Q6_W_vshuff_VVR(Q6_V_hi_W(ws64), Q6_V_lo_W(ws64), -4);
        vmemu(dst) = Q6_V_lo_W(ws64);
        vmemu(dst + 16) = Q6_V_hi_W(ws64);

        for (DT_S32 i = 0; i < 32; i++)
        {
            ref[i] = src_u[i] + src_v[i];
        }

        ret |= CHECK_CMP_VECTOR(ctx, dst, ref, 256);
    }

    // Q6_Wud_vadd_WudWud
    {
        DT_U64 src_u[32] = {0xffffffff, 0xffffffffffffffff, 0xfffffffffffffffe, 0x7fffffffffffffff};
        DT_U64 src_v[32] = {0xffffffff, 0, 1, 0x8000000000000000};

        DT_U64 ref[32];
        DT_U64 dst[32];

        HVX_Vector vu64_u0 = vmemu(src_u);
        HVX_Vector vu64_u1 = vmemu(src_u + 16);
        HVX_Vector vu64_v0 = vmemu(src_v);
        HVX_Vector vu64_v1 = vmemu(src_v + 16);

        HVX_VectorPair wu64_u = Q6_W_vdeal_VVR(vu64_u1, vu64_u0, -4);
        HVX_VectorPair wu64_v = Q6_W_vdeal_VVR(vu64_v1, vu64_v0, -4);
        HVX_VectorPair wu64 = Q6_Wud_vadd_WudWud(wu64_u, wu64_v);
        wu64 = Q6_W_vshuff_VVR(Q6_V_hi_W(wu64), Q6_V_lo_W(wu64), -4);
        vmemu(dst) = Q6_V_lo_W(wu64);
        vmemu(dst + 16) = Q6_V_hi_W(wu64);

        for (DT_S32 i = 0; i < 32; i++)
        {
            ref[i] = src_u[i] + src_v[i];
        }

        ret |= CHECK_CMP_VECTOR(ctx, dst, ref, 256);
    }

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}