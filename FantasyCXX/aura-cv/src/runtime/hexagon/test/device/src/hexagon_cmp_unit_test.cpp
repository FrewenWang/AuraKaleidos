#include "device/include/hexagon_instructions_unit_test.hpp"

NEW_TESTCASE(runtime_hexagon_instructions_cmp_test)
{
    Status ret = Status::OK;
    Context *ctx = UnitTest::GetInstance()->GetContext();

    MI_U8 ones[128];
    MI_U8 zeros[128];
    memset(ones, 1, 128);
    memset(zeros, 0, 128);
    HVX_Vector vu8_ones = vmemu(ones);
    HVX_Vector vu8_zeros = vmemu(zeros);

    // Q6_Q_vcmp_ge_VbVb
    {
        MI_S8 src_u[128] = {0, -128, 127, 0, -127, 127};
        MI_S8 src_v[128] = {0, -127, -128, -128, 0, 127};

        MI_U8 dst[128];
        MI_U8 ref[128];

        HVX_Vector vs8_u = vmemu(src_u);
        HVX_Vector vs8_v = vmemu(src_v);
        HVX_VectorPred q = Q6_Q_vcmp_ge_VbVb(vs8_u, vs8_v);
        HVX_VectorPair w = Q6_W_vswap_QVV(q, vu8_ones, vu8_zeros);
        vmemu(dst) = Q6_V_lo_W(w);

        for (MI_S32 i = 0; i < 128; i++)
        {
            ref[i] = (src_u[i] >= src_v[i]) ? 1 : 0;
        }

        ret |= CHECK_CMP_VECTOR(ctx, dst, ref, 128);
    }

    // Q6_Q_vcmp_ge_VubVub
    {
        MI_U8 src_u[128] = {0, 128, 255, 0, 254};
        MI_U8 src_v[128] = {0, 127, 255, 255, 255};

        MI_U8 dst[128];
        MI_U8 ref[128];

        HVX_Vector vu8_u = vmemu(src_u);
        HVX_Vector vu8_v = vmemu(src_v);
        HVX_VectorPred q = Q6_Q_vcmp_ge_VubVub(vu8_u, vu8_v);
        HVX_VectorPair w = Q6_W_vswap_QVV(q, vu8_ones, vu8_zeros);
        vmemu(dst) = Q6_V_lo_W(w);

        for (MI_S32 i = 0; i < 128; i++)
        {
            ref[i] = (src_u[i] >= src_v[i]) ? 1 : 0;
        }

        ret |= CHECK_CMP_VECTOR(ctx, dst, ref, 128);
    }

    // Q6_Q_vcmp_ge_VhVh
    {
        MI_S16 src_u[64] = {0, 0x7fff, 0x7fff, (MI_S16)0x8000, 0x7ffe, (MI_S16)0x8000};
        MI_S16 src_v[64] = {0, (MI_S16)0x8000, 0x7fff, (MI_S16)0x8000, 0x7fff, (MI_S16)0x8001};

        MI_U8 dst[128];
        MI_U8 ref[128];

        HVX_Vector vs16_u = vmemu(src_u);
        HVX_Vector vs16_v = vmemu(src_v);
        HVX_VectorPred q = Q6_Q_vcmp_ge_VhVh(vs16_u, vs16_v);
        HVX_VectorPair w = Q6_W_vswap_QVV(q, vu8_ones, vu8_zeros);
        vmemu(dst) = Q6_V_lo_W(w);

        for (MI_S32 i = 0; i < 64; i++)
        {
            if (src_u[i] >= src_v[i])
            {
                ref[i * 2] = 1;
                ref[i * 2 + 1] = 1;
            }
            else
            {
                ref[i * 2] = 0;
                ref[i * 2 + 1] = 0;
            }
        }
    }

    // Q6_Q_vcmp_ge_VuhVuh
    {
        MI_U16 src_u[64] = {0, 0xfffe, 0xffff, 0x7fff, 1};
        MI_U16 src_v[64] = {0, 0xffff, 0xffff, 0x7ffe, 0};

        MI_U8 dst[128];
        MI_U8 ref[128];

        HVX_Vector vu16_u = vmemu(src_u);
        HVX_Vector vu16_v = vmemu(src_v);
        HVX_VectorPred q = Q6_Q_vcmp_ge_VuhVuh(vu16_u, vu16_v);
        HVX_VectorPair w = Q6_W_vswap_QVV(q, vu8_ones, vu8_zeros);
        vmemu(dst) = Q6_V_lo_W(w);

        for (MI_S32 i = 0; i < 64; i++)
        {
            if (src_u[i] >= src_v[i])
            {
                ref[i * 2] = 1;
                ref[i * 2 + 1] = 1;
            }
            else
            {
                ref[i * 2] = 0;
                ref[i * 2 + 1] = 0;
            }
        }

        ret |= CHECK_CMP_VECTOR(ctx, dst, ref, 128);
    }

    // Q6_Q_vcmp_ge_VwVw
    {
        MI_S32 src_u[32] = {0, 0x7fffffff, 0x7fffffff, (MI_S32)0x80000000, 0x7ffffffe, (MI_S32)0x80000000};
        MI_S32 src_v[32] = {0, (MI_S32)0x80000000, 0x7fffffff, (MI_S32)0x80000000, 0x7fffffff, (MI_S32)0x80000001};

        MI_U8 dst[128];
        MI_U8 ref[128];
        memset(ref, 0, 128);

        HVX_Vector vs32_u = vmemu(src_u);
        HVX_Vector vs32_v = vmemu(src_v);
        HVX_VectorPred q = Q6_Q_vcmp_ge_VwVw(vs32_u, vs32_v);
        HVX_VectorPair w = Q6_W_vswap_QVV(q, vu8_ones, vu8_zeros);
        vmemu(dst) = Q6_V_lo_W(w);

        for (MI_S32 i = 0; i < 32; i++)
        {
            if (src_u[i] >= src_v[i])
            {
                ref[i * 4] = 1;
                ref[i * 4 + 1] = 1;
                ref[i * 4 + 2] = 1;
                ref[i * 4 + 3] = 1;
            }
        }

        ret |= CHECK_CMP_VECTOR(ctx, dst, ref, 128);
    }

    // Q6_Q_vcmp_ge_VuwVuw
    {

        MI_U32 src_u[32] = {0, 0xfffffffe, 0xffffffff, 0x7fffffff, 1};
        MI_U32 src_v[32] = {0, 0xffffffff, 0xffffffff, 0x7ffffffe, 0};

        MI_U8 dst[128];
        MI_U8 ref[128];
        memset(ref, 0, 128);

        HVX_Vector vu32_u = vmemu(src_u);
        HVX_Vector vu32_v = vmemu(src_v);
        HVX_VectorPred q = Q6_Q_vcmp_ge_VwVw(vu32_u, vu32_v);
        HVX_VectorPair w = Q6_W_vswap_QVV(q, vu8_ones, vu8_zeros);
        vmemu(dst) = Q6_V_lo_W(w);

        for (MI_S32 i = 0; i < 32; i++)
        {
            if (src_u[i] >= src_v[i])
            {
                ref[i * 4] = 1;
                ref[i * 4 + 1] = 1;
                ref[i * 4 + 2] = 1;
                ref[i * 4 + 3] = 1;
            }
        }

        ret |= CHECK_CMP_VECTOR(ctx, dst, ref, 128);
    }

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}