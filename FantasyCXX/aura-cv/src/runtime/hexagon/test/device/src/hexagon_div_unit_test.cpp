#include "device/include/hexagon_instructions_unit_test.hpp"

NEW_TESTCASE(runtime_hexagon_instructions_div_test)
{
    Status ret = Status::OK;
    Context *ctx = UnitTest::GetInstance()->GetContext();

    // Q6_Vub_vdiv_VubVub
    {
        MI_U8 src_u[128] = {255, 255, 255, 1, 1, 254, 255};
        MI_U8 src_v[128] = {1, 2, 255, 255, 1, 255, 254};
        memset(src_u + 7, 0, 121);
        memset(src_v + 7, 255, 121);
        MI_U8 dst[128], ref[128];

        HVX_Vector vu8_u = vmemu(src_u);
        HVX_Vector vu8_v = vmemu(src_v);

        HVX_Vector vu8_dst = Q6_Vub_vdiv_VubVub(vu8_u, vu8_v);
        vmemu(dst) = vu8_dst;

        for (MI_S32 i = 0; i < 128; i++)
        {
            ref[i] = src_u[i] / src_v[i];
        }

        ret |= CHECK_CMP_VECTOR(ctx, dst, ref, 128);
    }

    // Q6_Vb_vdiv_VbVb
    {
        MI_S8 src_u[128] = {-128, 127, -128, -127, 127, -128, 1, 127, 1};
        MI_S8 src_v[128] = {-128, 127, 1, -1, -128, 127, -128, 1, 127};
        memset(src_u + 9, 0, 119);
        memset(src_v + 9, 255, 119);
        MI_S8 dst[128], ref[128];

        HVX_Vector vs8_u = vmemu(src_u);
        HVX_Vector vs8_v = vmemu(src_v);

        HVX_Vector vs8_dst = Q6_Vb_vdiv_VbVb(vs8_u, vs8_v);
        vmemu(dst) = vs8_dst;

        for (MI_S32 i = 0; i < 128; i++)
        {
            ref[i] = src_u[i] / src_v[i];
        }

        ret |= CHECK_CMP_VECTOR(ctx, dst, ref, 128);
    }

    // Q6_Vuh_vdiv8_VuhVuh
    {
        MI_U16 src_u[64] = {0xffff, 0xffff, 0, 0xfffe, 0xffff, 0xff};
        MI_U16 src_v[64] = {0xffff, 0x0101, 0xffff, 0xffff, 0xfffe, 1};
        memset(src_u + 6, 0, 116);
        memset(src_v + 6, 255, 116);
        MI_U8 dst[128], ref[128];

        HVX_Vector vu16_u = vmemu(src_u);
        HVX_Vector vu16_v = vmemu(src_v);

        HVX_Vector vu8_dst = Q6_Vuh_vdiv8_VuhVuh(vu16_u, vu16_v);
        vmemu(dst) = vu8_dst;

        for (MI_S32 i = 0; i < 64; i++)
        {
            ref[i] = src_u[i] / src_v[i];
            dst[i] = dst[i * 2];
        }

        ret |= CHECK_CMP_VECTOR(ctx, dst, ref, 64);
    }

    // Q6_Vh_vdiv8_VhVh
    {
        MI_S16 src_u[64] = {0x7fff, (MI_S16)0x8000, 0x7fff, (MI_S16)0x8000, 0x7fff, 0x7fff, (MI_S16)0x8000, (MI_S16)0x8000, -128, 127};
        MI_S16 src_v[64] = {0x7fff, (MI_S16)0x8000, (MI_S16)0x8000, 0x7fff, 256, -255, 256, -257, 1, 1};
        memset(src_u + 10, 0, 108);
        memset(src_v + 10, 255, 108);
        MI_S8 dst[128], ref[128];

        HVX_Vector vs16_u = vmemu(src_u);
        HVX_Vector vs16_v = vmemu(src_v);

        HVX_Vector vs8_dst = Q6_Vh_vdiv8_VhVh(vs16_u, vs16_v);
        vmemu(dst) = vs8_dst;

        for (MI_S32 i = 0; i < 64; i++)
        {
            ref[i] = src_u[i] / src_v[i];
            dst[i] = dst[i * 2];
        }

        ret |= CHECK_CMP_VECTOR(ctx, dst, ref, 64);
    }

    // Q6_Vuh_vdiv_VuhVuh
    {
        MI_U16 src_u[64] = {0xffff, 0xffff, 1, 0, 0xffff, 0xfffe};
        MI_U16 src_v[64] = {0xffff, 1, 0xffff, 0xffff, 0xfffe, 0xffff};
        memset(src_u + 6, 0, 116);
        memset(src_v + 6, 255, 116);
        MI_U16 dst[64], ref[64];

        HVX_Vector vu16_u = vmemu(src_u);
        HVX_Vector vu16_v = vmemu(src_v);

        HVX_Vector vu16_dst = Q6_Vuh_vdiv_VuhVuh(vu16_u, vu16_v);
        vmemu(dst) = vu16_dst;

        for (MI_S32 i = 0; i < 64; i++)
        {
            ref[i] = src_u[i] / src_v[i];
        }

        ret |= CHECK_CMP_VECTOR(ctx, dst, ref, 128);
    }

    // Q6_Vh_vdiv_VhVh
    {
        MI_S16 src_u[64] = {0x7fff, (MI_S16)0x8000, 0x7fff, (MI_S16)0x8000, 0x7fff, (MI_S16)0x8000, 1, 1, 0x7fff, (MI_S16)0x8000};
        MI_S16 src_v[64] = {0x7fff, (MI_S16)0x8000, (MI_S16)0x8000, 0x7fff, 1, 1, 0x7fff, (MI_S16)0x8000, -1, 2};
        memset(src_u + 10, 0, 108);
        memset(src_v + 10, 255, 108);
        MI_S16 dst[64], ref[64];

        HVX_Vector vs16_u = vmemu(src_u);
        HVX_Vector vs16_v = vmemu(src_v);

        HVX_Vector vs16_dst = Q6_Vh_vdiv_VhVh(vs16_u, vs16_v);
        vmemu(dst) = vs16_dst;

        for (MI_S32 i = 0; i < 64; i++)
        {
            ref[i] = src_u[i] / src_v[i];
        }

        ret |= CHECK_CMP_VECTOR(ctx, dst, ref, 128);
    }

    // Q6_Vuh_vdiv8_WuwVuh
    {
        MI_U32 src_u[64] = {0xffff, 0xfeff01, 0xffff, 0xff, 1, 1};
        MI_U16 src_v[64] = {0xffff, 0xffff, 0x101, 1, 0xffff, 1};
        memset(src_u + 6, 0, 232);
        memset(src_v + 6, 255, 116);
        MI_U8 dst[128], ref[128];

        HVX_Vector vu32_u0 = vmemu(src_u);
        HVX_Vector vu32_u1 = vmemu(src_u + 32);
        HVX_VectorPair vu32_u = Q6_W_vdeal_VVR(vu32_u1, vu32_u0, -4);
        HVX_Vector vu16_v = vmemu(src_v);

        HVX_Vector vu8_dst = Q6_Vuh_vdiv8_WuwVuh(vu32_u, vu16_v);
        vmemu(dst) = vu8_dst;

        for (MI_S32 i = 0; i < 64; i++)
        {
            ref[i] = src_u[i] / (MI_U32)src_v[i];
            dst[i] = dst[i * 2];
        }

        ret |= CHECK_CMP_VECTOR(ctx, dst, ref, 64);
    }

    // Q6_Vh_vdiv8_WwVh
    {
        MI_S32 src_u[64] = {0x3f7f81, (MI_S32)0xffc08000, (MI_S32)0xffbf8082, 0x407fff, 127, -128, 1, 1, 1, (MI_S32)0xffff8000, 0x7fff};
        MI_S16 src_v[64] = {0x7fff, (MI_S16)0x8000, 0x7fff, (MI_S16)0x8000, 1, 1, 1, 0x7fff, (MI_S16)0x8000, (MI_S16)0x8000, 0x7fff};
        memset(src_u + 11, 0, 212);
        memset(src_v + 11, 255, 106);
        MI_S8 dst[128], ref[128];

        HVX_Vector vs32_u0 = vmemu(src_u);
        HVX_Vector vs32_u1 = vmemu(src_u + 32);
        HVX_VectorPair vs32_u = Q6_W_vdeal_VVR(vs32_u1, vs32_u0, -4);
        HVX_Vector vs16_v = vmemu(src_v);

        HVX_Vector vs8_dst = Q6_Vh_vdiv8_WwVh(vs32_u, vs16_v);
        vmemu(dst) = vs8_dst;

        for (MI_S32 i = 0; i < 64; i++)
        {
            ref[i] = src_u[i] / (MI_S32)src_v[i];
            dst[i] = dst[i * 2];
        }

        ret |= CHECK_CMP_VECTOR(ctx, dst, ref, 64);
    }

    // Q6_Vuh_vdiv_WuwVuh
    {
        MI_U32 src_u[64] = {0xffff, 1, 0xfffe0001, 0xffff};
        MI_U16 src_v[64] = {0xffff, 1, 0xffff, 1};
        memset(src_u + 4, 0, 240);
        memset(src_v + 4, 255, 120);
        MI_U16 dst[64], ref[64];

        HVX_Vector vu32_u0 = vmemu(src_u);
        HVX_Vector vu32_u1 = vmemu(src_u + 32);
        HVX_VectorPair vu32_u = Q6_W_vdeal_VVR(vu32_u1, vu32_u0, -4);
        HVX_Vector vu16_v = vmemu(src_v);

        HVX_Vector vu16_dst = Q6_Vuh_vdiv_WuwVuh(vu32_u, vu16_v);
        vmemu(dst) = vu16_dst;

        for (MI_S32 i = 0; i < 64; i++)
        {
            ref[i] = src_u[i] / (MI_U32)src_v[i];
        }

        ret |= CHECK_CMP_VECTOR(ctx, dst, ref, 128);
    }

    // Q6_Vh_vdiv_WwVh
    {
        MI_S32 src_u[64] = {0x7fff, (MI_S32)0xffff8000, 0x3fff0001, (MI_S32)0xc0008000, (MI_S32)0xc0008000,
                            0x40000000, 0x7fff, (MI_S32)0xffff8000, 1, 1};
        MI_S16 src_v[64] = {0x7fff, (MI_S16)0x8000, 0x7fff, (MI_S16)0x8000, 0x7fff, (MI_S16)0x8000,
                            1, 1, 0x7fff, (MI_S16)0x8000};
        memset(src_u + 10, 0, 216);
        memset(src_v + 10, 255, 108);
        MI_S16 dst[64], ref[64];

        HVX_Vector vs32_u0 = vmemu(src_u);
        HVX_Vector vs32_u1 = vmemu(src_u + 32);
        HVX_VectorPair vs32_u = Q6_W_vdeal_VVR(vs32_u1, vs32_u0, -4);
        HVX_Vector vs16_v = vmemu(src_v);

        HVX_Vector vs16_dst = Q6_Vh_vdiv_WwVh(vs32_u, vs16_v);
        vmemu(dst) = vs16_dst;

        for (MI_S32 i = 0; i < 64; i++)
        {
            ref[i] = src_u[i] / (MI_S32)src_v[i];
        }

        ret |= CHECK_CMP_VECTOR(ctx, dst, ref, 128);
    }

    // Q6_Vuw_vdiv8_VuwVuw
    {
        MI_U32 src_u[32] = {0xffffffff, 0xffffffff, 255, 0xffffffff, 0xfffffffe};
        MI_U32 src_v[32] = {0xffffffff, 0x1010101, 1, 0xfffffffe, 0xffffffff};
        memset(src_u + 5, 0, 108);
        memset(src_v + 5, 255, 108);
        MI_U8 dst[128], ref[128];

        HVX_Vector vu32_u = vmemu(src_u);
        HVX_Vector vu32_v = vmemu(src_v);
        HVX_Vector vu8_dst = Q6_Vuw_vdiv8_VuwVuw(vu32_u, vu32_v);
        vmemu(dst) = vu8_dst;

        for (MI_S32 i = 0; i < 32; i++)
        {
            ref[i] = src_u[i] / src_v[i];
            dst[i] = dst[i * 4];
        }

        ret |= CHECK_CMP_VECTOR(ctx, dst, ref, 32);
    }

    // Q6_Vw_vdiv8_VwVw
    {
        MI_S32 src_u[32] = {0x7fffffff, (MI_S32)0x80000000, 0x7fffffff, (MI_S32)0x80000000, 0x7fffffff, (MI_S32)0x80000000};
        MI_S32 src_v[32] = {0x7fffffff, (MI_S32)0x80000000, 0x1020408, (MI_S32)0xfefdfbf8, (MI_S32)0xff000001, 0x1000000};
        memset(src_u + 6, 0, 104);
        memset(src_v + 6, 255, 104);
        MI_S8 dst[128], ref[128];

        HVX_Vector vs32_u = vmemu(src_u);
        HVX_Vector vs32_v = vmemu(src_v);
        HVX_Vector vs8_dst = Q6_Vw_vdiv8_VwVw(vs32_u, vs32_v);
        vmemu(dst) = vs8_dst;

        for (MI_S32 i = 0; i < 32; i++)
        {
            ref[i] = src_u[i] / src_v[i];
            dst[i] = dst[i * 4];
        }

        ret |= CHECK_CMP_VECTOR(ctx, dst, ref, 32);
    }

    // Q6_Vuw_vdiv16_VuwVuw
    {
        MI_U32 src_u[32] = {0xffffffff, 0xffffffff, 0xffff, 0xffffffff, 0xfffffffe};
        MI_U32 src_v[32] = {0xffffffff, 0x10001, 1, 0xfffffffe, 0xffffffff};
        memset(src_u + 5, 0, 108);
        memset(src_v + 5, 255, 108);
        MI_U16 dst[64], ref[64];

        HVX_Vector vu32_u = vmemu(src_u);
        HVX_Vector vu32_v = vmemu(src_v);
        HVX_Vector vu16_dst = Q6_Vuw_vdiv16_VuwVuw(vu32_u, vu32_v);
        vmemu(dst) = vu16_dst;

        for (MI_S32 i = 0; i < 32; i++)
        {
            ref[i] = src_u[i] / src_v[i];
            dst[i] = dst[i * 2];
        }

        ret |= CHECK_CMP_VECTOR(ctx, dst, ref, 64);
    }

    // Q6_Vw_vdiv16_VwVw
    {
        MI_S32 src_u[32] = {0x7fffffff, (MI_S32)0x80000000, 0x7fffffff, (MI_S32)0x80000000, 0x7fffffff, (MI_S32)0x80000000};
        MI_S32 src_v[32] = {0x7fffffff, (MI_S32)0x80000000, 0x10002, (MI_S32)0xfffefffe, (MI_S32)0xffff0000, 0x10000};
        memset(src_u + 6, 0, 104);
        memset(src_v + 6, 255, 104);
        MI_S16 dst[64], ref[64];

        HVX_Vector vs32_u = vmemu(src_u);
        HVX_Vector vs32_v = vmemu(src_v);
        HVX_Vector vs16_dst = Q6_Vw_vdiv16_VwVw(vs32_u, vs32_v);
        vmemu(dst) = vs16_dst;

        for (MI_S32 i = 0; i < 32; i++)
        {
            ref[i] = src_u[i] / src_v[i];
            dst[i] = dst[i * 2];
        }

        ret |= CHECK_CMP_VECTOR(ctx, dst, ref, 64);
    }

    // Q6_Vuw_vdiv_VuwVuw
    {
        MI_U32 src_u[32] = {0xffffffff, 0xffffffff, 0xffffffff, 0xfffffffe};
        MI_U32 src_v[32] = {0xffffffff, 1, 0xfffffffe, 0xffffffff};
        memset(src_u + 4, 0, 112);
        memset(src_v + 4, 255, 112);
        MI_U32 dst[32], ref[32];

        HVX_Vector vu32_u = vmemu(src_u);
        HVX_Vector vu32_v = vmemu(src_v);
        HVX_Vector vu32_dst = Q6_Vuw_vdiv_VuwVuw(vu32_u, vu32_v);
        vmemu(dst) = vu32_dst;

        for (MI_S32 i = 0; i < 32; i++)
        {
            ref[i] = src_u[i] / src_v[i];
        }

        ret |= CHECK_CMP_VECTOR(ctx, dst, ref, 128);
    }

    // Q6_Vw_vdiv_VwVw
    {
        MI_S32 src_u[32] = {0x7fffffff, (MI_S32)0x80000000, 0x7fffffff, (MI_S32)0x80000000, 0x7fffffff, (MI_S32)0x80000001};
        MI_S32 src_v[32] = {0x7fffffff, (MI_S32)0x80000000, 1, 1, -1, -1};
        memset(src_u + 6, 0, 104);
        memset(src_v + 6, 255, 104);
        MI_S32 dst[32], ref[32];

        HVX_Vector vs32_u = vmemu(src_u);
        HVX_Vector vs32_v = vmemu(src_v);
        HVX_Vector vs32_dst = Q6_Vw_vdiv_VwVw(vs32_u, vs32_v);
        vmemu(dst) = vs32_dst;

        for (MI_S32 i = 0; i < 32; i++)
        {
            ref[i] = src_u[i] / src_v[i];
        }

        ret |= CHECK_CMP_VECTOR(ctx, dst, ref, 128);
    }

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}