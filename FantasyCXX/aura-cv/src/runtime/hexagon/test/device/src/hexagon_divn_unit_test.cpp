#include "device/include/hexagon_instructions_unit_test.hpp"

#define TESTDIVN_FAST(type, divisor, item_number)                                                   \
    for (MI_S32 item = 0; item < item_number; item += 2)                                            \
    {                                                                                               \
        constexpr MI_S32 compute_num = 128 / sizeof(type);                                          \
        type dst[(compute_num * 2)], ref[(compute_num * 2)];                                        \
        type *p_src = src_u + item * compute_num;                                                   \
        HVX_Vector v_src0        = vmemu(p_src);                                                    \
        HVX_Vector v_src1        = vmemu(p_src + compute_num);                                      \
        HVX_VectorPair v_src     = Q6_W_vcombine_VV(v_src1, v_src0);                                \
        HVX_VectorPair v_result  = vdiv_n<type, divisor>(v_src);                                    \
        vmemu(dst)               = Q6_V_lo_W(v_result);                                             \
        vmemu(dst + compute_num) = Q6_V_hi_W(v_result);                                             \
                                                                                                    \
        for (MI_S32 i = 0; i < compute_num * 2; i++)                                                \
        {                                                                                           \
            ref[i] = (0 == divisor) ? 0 : p_src[i] / divisor;                                       \
        }                                                                                           \
        Status ret_cmp = CHECK_CMP_VECTOR(ctx, dst, ref, 256);                                      \
        if (Status::OK == ret_cmp)                                                                  \
        {                                                                                           \
            AURA_LOGD(ctx, AURA_TAG, "divsor is %d item %ld :pass !\n", divisor, (MI_S32)item);     \
        }                                                                                           \
        else                                                                                        \
        {                                                                                           \
            AURA_LOGE(ctx, AURA_TAG, "divsor is %d item %ld :fail !\n", divisor, (MI_S32)item);     \
        }                                                                                           \
        ret |= ret_cmp;                                                                             \
    }

NEW_TESTCASE(runtime_hexagon_instructions_divn_test_U8)
{
    Status ret   = Status::OK;
    Context *ctx = UnitTest::GetInstance()->GetContext();

    MI_U8 src_u[256];
    MI_U8 divisor[256];
    for (MI_S32 i = 0; i < 256; i++)
    {
        src_u[i]   = i;
        divisor[i] = i;
    }

    TESTDIVN_FAST(MI_U8, 3,   2);
    TESTDIVN_FAST(MI_U8, 5,   2);
    TESTDIVN_FAST(MI_U8, 7,   2);
    TESTDIVN_FAST(MI_U8, 9,   2);
    TESTDIVN_FAST(MI_U8, 25,  2);
    TESTDIVN_FAST(MI_U8, 49,  2);

    TESTDIVN_FAST(MI_U8, 0,   2);
    TESTDIVN_FAST(MI_U8, 2,   2);
    TESTDIVN_FAST(MI_U8, 4,   2);
    TESTDIVN_FAST(MI_U8, 6,   2);
    TESTDIVN_FAST(MI_U8, 15,  2);
    TESTDIVN_FAST(MI_U8, 90,  2);
    TESTDIVN_FAST(MI_U8, 255, 2);

    Status ret_comm = Status::OK;
    {
        HVX_Vector     vu8_u_0 = vmemu(src_u);
        HVX_Vector     vu8_u_1 = vmemu(src_u + 128);
        HVX_VectorPair vu8_u   = Q6_W_vcombine_VV(vu8_u_1, vu8_u_0);

        for (MI_S32 j = 0; j < 256; j++)
        {
            MI_U8 dst[256], ref[256];
            MI_U8 divisor_u8 = divisor[j];

            HvxVdivnHelper<MI_U8> vdivn(divisor_u8);
            HVX_VectorPair vu8_result = vdivn(vu8_u);

            vmemu(dst)       = Q6_V_lo_W(vu8_result);
            vmemu(dst + 128) = Q6_V_hi_W(vu8_result);

            for (MI_S32 i = 0; i < 256; i++)
            {
                ref[i] = (0 == divisor_u8) ? 0 : src_u[i] / divisor_u8;
            }

            ret_comm |= CHECK_CMP_VECTOR(ctx, dst, ref, 256);
        }
    }

    if (Status::OK == ret_comm)
    {
        AURA_LOGD(ctx, AURA_TAG, "HvxVdivnHelper<MI_U8> test pass !\n");
    }
    else
    {
        AURA_LOGE(ctx, AURA_TAG, "HvxVdivnHelper<MI_U8> test fail !\n");
    }

    ret |= ret_comm;
    AddTestResult(AURA_GET_TEST_STATUS(ret));
}

NEW_TESTCASE(runtime_hexagon_instructions_divn_test_S8)
{
    Status ret   = Status::OK;
    Context *ctx = UnitTest::GetInstance()->GetContext();

    MI_S8 src_u[256];
    MI_S8 divisor[256];
    for (MI_S32 i = 0; i < 256; i++)
    {
        src_u[i]   = i - 128;
        divisor[i] = i - 128;
    }

    TESTDIVN_FAST(MI_S8, 3,    2);
    TESTDIVN_FAST(MI_S8, 5,    2);
    TESTDIVN_FAST(MI_S8, 7,    2);
    TESTDIVN_FAST(MI_S8, 9,    2);
    TESTDIVN_FAST(MI_S8, 25,   2);
    TESTDIVN_FAST(MI_S8, 49,   2);

    TESTDIVN_FAST(MI_S8, 0,    2);
    TESTDIVN_FAST(MI_S8, 2,    2);
    TESTDIVN_FAST(MI_S8, 4,    2);
    TESTDIVN_FAST(MI_S8, 6,    2);
    TESTDIVN_FAST(MI_S8, 15,   2);
    TESTDIVN_FAST(MI_S8, 59,   2);
    TESTDIVN_FAST(MI_S8, 127,  2);

    TESTDIVN_FAST(MI_S8, -2,   2);
    TESTDIVN_FAST(MI_S8, -3,   2);
    TESTDIVN_FAST(MI_S8, -4,   2);
    TESTDIVN_FAST(MI_S8, -5,   2);
    TESTDIVN_FAST(MI_S8, -13,  2);
    TESTDIVN_FAST(MI_S8, -57,  2);
    TESTDIVN_FAST(MI_S8, -128, 2);

    Status ret_comm = Status::OK;
    {
        HVX_Vector     vs8_u_0 = vmemu(src_u);
        HVX_Vector     vs8_u_1 = vmemu(src_u + 128);
        HVX_VectorPair vs8_u   = Q6_W_vcombine_VV(vs8_u_1, vs8_u_0);

        for (MI_S32 j = 0; j < 256; j++)
        {
            MI_S8 dst[256], ref[256];
            MI_S8 divisor_s8 = divisor[j];

            HvxVdivnHelper<MI_S8> vdivn(divisor_s8);
            HVX_VectorPair vs8_result = vdivn(vs8_u);
            vmemu(dst)       = Q6_V_lo_W(vs8_result);
            vmemu(dst + 128) = Q6_V_hi_W(vs8_result);

            for (MI_S32 i = 0; i < 256; i++)
            {
                ref[i] = (divisor_s8 == 0) ? 0 : src_u[i] / divisor_s8;
            }

            ret_comm |= CHECK_CMP_VECTOR(ctx, dst, ref, 256);
        }
    }

    if (Status::OK == ret_comm)
    {
        AURA_LOGD(ctx, AURA_TAG, "HvxVdivnHelper<MI_S8> test pass !\n");
    }
    else
    {
        AURA_LOGE(ctx, AURA_TAG, "HvxVdivnHelper<MI_S8> test fail !\n");
    }

    ret |= ret_comm;
    AddTestResult(AURA_GET_TEST_STATUS(ret));
}

NEW_TESTCASE(runtime_hexagon_instructions_divn_test_U16)
{
    Status ret      = Status::OK;
    Status ret_comm = Status::OK;
    Context *ctx    = UnitTest::GetInstance()->GetContext();

    MI_U16 *src_u   = static_cast<MI_U16*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, 65536 * sizeof(MI_U16), 0));
    MI_U16 *divisor = static_cast<MI_U16*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, 65536 * sizeof(MI_U16), 0));
    if (MI_NULL == src_u || MI_NULL == divisor)
    {
        goto EXIT;
    }

    for (MI_S32 i = 0; i < 65536; i++)
    {
        src_u[i]   = i;
        divisor[i] = i;
    }

    TESTDIVN_FAST(MI_U16, 3,     1024);
    TESTDIVN_FAST(MI_U16, 5,     1024);
    TESTDIVN_FAST(MI_U16, 7,     1024);
    TESTDIVN_FAST(MI_U16, 9,     1024);
    TESTDIVN_FAST(MI_U16, 25,    1024);
    TESTDIVN_FAST(MI_U16, 49,    1024);

    TESTDIVN_FAST(MI_U16, 0,     1024);
    TESTDIVN_FAST(MI_U16, 2,     1024);
    TESTDIVN_FAST(MI_U16, 4,     1024);
    TESTDIVN_FAST(MI_U16, 5,     1024);
    TESTDIVN_FAST(MI_U16, 6542,  1024);
    TESTDIVN_FAST(MI_U16, 10354, 1024);
    TESTDIVN_FAST(MI_U16, 65535, 1024);

    {
        for (MI_S32 i = 0; i < 1024; i += 2)
        {
            MI_U16 dst[128], ref[128];
            MI_U16 *p_src = src_u + i * 64;

            HVX_Vector vu16_u_0   = vmemu(p_src);
            HVX_Vector vu16_u_1   = vmemu(p_src + 64);
            HVX_VectorPair vu16_u = Q6_W_vcombine_VV(vu16_u_1, vu16_u_0);

            for (MI_S32 j = 0; j < 65536; j++)
            {
                MI_U16 divisor_u16 = divisor[j];
                HvxVdivnHelper<MI_U16> vdivn(divisor_u16);
                HVX_VectorPair v_dst = vdivn(vu16_u);

                vmemu(dst)      = Q6_V_lo_W(v_dst);
                vmemu(dst + 64) = Q6_V_hi_W(v_dst);

                for (MI_S32 i = 0; i < 128; i++)
                {
                    ref[i] = (0 == divisor_u16) ? 0 : p_src[i] / divisor_u16;
                }

                ret_comm |= CHECK_CMP_VECTOR(ctx, dst, ref, 256);
            }
        }
    }

    if (Status::OK == ret_comm)
    {
        AURA_LOGD(ctx, AURA_TAG, "HvxVdivnHelper<MI_U16> test pass !\n");
    }
    else
    {
        AURA_LOGE(ctx, AURA_TAG, "HvxVdivnHelper<MI_U16> test fail !\n");
    }

    ret |= ret_comm;
EXIT:
    AURA_FREE(ctx, src_u);
    AURA_FREE(ctx, divisor);

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}

NEW_TESTCASE(runtime_hexagon_instructions_divn_test_S16)
{
    Status ret      = Status::OK;
    Status ret_comm = Status::OK;
    Context *ctx    = UnitTest::GetInstance()->GetContext();

    MI_S16 *src_u   = static_cast<MI_S16*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, 65536 * sizeof(MI_S16), 0));
    MI_S16 *divisor = static_cast<MI_S16*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, 65536 * sizeof(MI_S16), 0));
    if (MI_NULL == src_u || MI_NULL == divisor)
    {
        goto EXIT;
    }

    for (MI_S32 i = 0; i < 65536; i++)
    {
        src_u[i]   = i - 32768;
        divisor[i] = i - 32768;
    }

    TESTDIVN_FAST(MI_S16, 3,      1024);
    TESTDIVN_FAST(MI_S16, 5,      1024);
    TESTDIVN_FAST(MI_S16, 7,      1024);
    TESTDIVN_FAST(MI_S16, 9,      1024);
    TESTDIVN_FAST(MI_S16, 25,     1024);
    TESTDIVN_FAST(MI_S16, 49,     1024);

    TESTDIVN_FAST(MI_S16, 0,      1024);
    TESTDIVN_FAST(MI_S16, 15,     1024);
    TESTDIVN_FAST(MI_S16, 364,    1024);
    TESTDIVN_FAST(MI_S16, 6542,   1024);
    TESTDIVN_FAST(MI_S16, 10245,  1024);
    TESTDIVN_FAST(MI_S16, 23354,  1024);

    TESTDIVN_FAST(MI_S16, -2,     1024);
    TESTDIVN_FAST(MI_S16, -42,    1024);
    TESTDIVN_FAST(MI_S16, -436,   1024);
    TESTDIVN_FAST(MI_S16, -5432,  1024);
    TESTDIVN_FAST(MI_S16, -15154, 1024);
    TESTDIVN_FAST(MI_S16, -32768, 1024);

    {
        for (MI_S32 i = 0; i < 1024; i += 2)
        {
            MI_S16 dst[128], ref[128];
            MI_S16 *p_src = src_u + i * 64;

            HVX_Vector vs16_u_0   = vmemu(p_src);
            HVX_Vector vs16_u_1   = vmemu(p_src + 64);
            HVX_VectorPair vs16_u = Q6_W_vcombine_VV(vs16_u_1, vs16_u_0);

            for (MI_S32 j = 0; j < 65535; j++)
            {
                MI_S16 divisor_s16 = divisor[j];
                HvxVdivnHelper<MI_S16> vdivn(divisor_s16);
                HVX_VectorPair v_dst = vdivn(vs16_u);

                vmemu(dst)      = Q6_V_lo_W(v_dst);
                vmemu(dst + 64) = Q6_V_hi_W(v_dst);

                for (MI_S32 i = 0; i < 128; i++)
                {
                    ref[i] = (0 == divisor_s16) ? 0 : p_src[i] / divisor_s16;
                }

                ret_comm |= CHECK_CMP_VECTOR(ctx, dst, ref, 256);
            }
        }
    }

    if (Status::OK == ret_comm)
    {
        AURA_LOGD(ctx, AURA_TAG, "HvxVdivnHelper<MI_S16> test pass !\n");
    }
    else
    {
        AURA_LOGE(ctx, AURA_TAG, "HvxVdivnHelper<MI_S16> test fail !\n");
    }

    ret |= ret_comm;
EXIT:
    AURA_FREE(ctx, src_u);
    AURA_FREE(ctx, divisor);

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}

NEW_TESTCASE(runtime_hexagon_instructions_divn_test_U32)
{
    Status ret      = Status::OK;
    Status ret_comm = Status::OK;
    Context *ctx = UnitTest::GetInstance()->GetContext();

    MI_U32 *src_u   = static_cast<MI_U32*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, ((2 * 65536 + 256) * sizeof(MI_U32)), 0));
    MI_U32 *divisor = static_cast<MI_U32*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, ((2 * 65536 + 256) * sizeof(MI_U32)), 0));
    if (MI_NULL == src_u || MI_NULL == divisor)
    {
        goto EXIT;
    }

    for (MI_U32 i = 0; i < 65536; i++)
    {
        src_u[i]   = i;
        divisor[i] = i;
    }

    for (MI_U32 i = 0; i < 1024; i++)
    {
        MI_U32 start_number = 65536 + (1 << 22) * i;
        for (MI_U32 j = 0; j < 64; j++)
        {
            src_u[65536 + i * 64 + j]   = start_number + rand() | ((1 << 21));
            divisor[65536 + i * 64 + j] = start_number + rand() | ((1 << 21));
        }
    }

    for (MI_U32 i = 0; i < 256; i++)
    {
        src_u[65536 * 2 + i]   = UINT32_MAX - 255 + i;
        divisor[65536 * 2 + i] = UINT32_MAX - 255 + i;
    }

    TESTDIVN_FAST(MI_U32, 0,                4104);
    TESTDIVN_FAST(MI_U32, 9,                4104);
    TESTDIVN_FAST(MI_U32, 12,               4104);
    TESTDIVN_FAST(MI_U32, 543,              4104);
    TESTDIVN_FAST(MI_U32, 5251,             4104);
    TESTDIVN_FAST(MI_U32, 43242,            4104);
    TESTDIVN_FAST(MI_U32, (UINT32_MAX - 1), 4104);

    {
        for (MI_U32 i = 0; i < 4104; i += 2)
        {
            MI_U32 dst[64], ref[64];
            MI_U32 *p_src = src_u + 32 * i;

            HVX_Vector vu32_u_0 = vmemu(p_src);
            HVX_Vector vu32_u_1 = vmemu(p_src + 32);
            HVX_VectorPair wu32_u = Q6_W_vcombine_VV(vu32_u_1, vu32_u_0);

            for (MI_U32 j = 0; j < 131328; j++)
            {
                MI_U32 divisor_u32 = divisor[j];
                HvxVdivnHelper<MI_U32> vdiv_n(divisor_u32);
                HVX_VectorPair w_dst = vdiv_n(wu32_u);

                vmemu(dst)      = Q6_V_lo_W(w_dst);
                vmemu(dst + 32) = Q6_V_hi_W(w_dst);

                for (MI_U32 k = 0; k < 64; k++)
                {
                    ref[k] = (0 == divisor_u32) ? 0 : (p_src[k] / divisor_u32);
                }

                ret_comm |= CHECK_CMP_VECTOR(ctx, dst, ref, 256);
            }
        }
    }

    if (Status::OK == ret_comm)
    {
        AURA_LOGD(ctx, AURA_TAG, "HvxVdivnHelper<MI_U32> test pass !\n");
    }
    else
    {
        AURA_LOGE(ctx, AURA_TAG, "HvxVdivnHelper<MI_U32> test fail !\n");
    }

    ret |= ret_comm;
EXIT:
    AURA_FREE(ctx, src_u);
    AURA_FREE(ctx, divisor);

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}

NEW_TESTCASE(runtime_hexagon_instructions_divn_test_S32)
{
    Status ret      = Status::OK;
    Status ret_comm = Status::OK;
    Context *ctx = UnitTest::GetInstance()->GetContext();

    MI_S32 *src_u   = static_cast<MI_S32*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, ((2 * 65536 + 256) * sizeof(MI_S32)), 0));
    MI_S32 *divisor = static_cast<MI_S32*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, ((2 * 65536 + 256) * sizeof(MI_S32)), 0));
    if (MI_NULL == src_u || MI_NULL == divisor)
    {
        goto EXIT;
    }

    for (MI_U32 i = 0; i < 65536; i++)
    {
        src_u[i]   = i - 32768;
        divisor[i] = i - 32768;
    }

    for (MI_U32 i = 0; i < 512; i++)
    {
        MI_S32 start_number_n = -65536 - (1 << 21) * i;
        MI_S32 start_number   =  65536 + (1 << 21) * i;

        for (MI_U32 j = 0; j < 64; j++)
        {
            src_u[65536 + i * 64 + j]           = start_number_n - rand() | ((1 << 21));
            divisor[65536 + i * 64 + j]         = start_number_n - rand() | ((1 << 21));
            src_u[65536 + (i + 512) * 64 + j]   = start_number   + rand() | ((1 << 22));
            divisor[65536 + (i + 512) * 64 + j] = start_number   + rand() | ((1 << 22));
        }
    }

    for (MI_U32 i = 0; i < 128; i++)
    {
        src_u[65536 * 2 + i]           = INT32_MIN + 255 - i;
        src_u[65536 * 2 + (i + 128)]   = INT32_MAX - 255 + i;
        divisor[65536 * 2 + i]         = INT32_MIN + 255 - i;
        divisor[65536 * 2 + (i + 128)] = INT32_MAX - 255 + i;
    }

    TESTDIVN_FAST(MI_S32, 0,               4104);
    TESTDIVN_FAST(MI_S32, 2,               4104);
    TESTDIVN_FAST(MI_S32, 34,              4104);
    TESTDIVN_FAST(MI_S32, 535,             4104);
    TESTDIVN_FAST(MI_S32, 2342,            4104);
    TESTDIVN_FAST(MI_S32, 73523,           4104);
    TESTDIVN_FAST(MI_S32, 544459,          4104);
    TESTDIVN_FAST(MI_S32, (INT32_MAX - 1), 4104);

    TESTDIVN_FAST(MI_S32, -2,              4104);
    TESTDIVN_FAST(MI_S32, -54,             4104);
    TESTDIVN_FAST(MI_S32, -975,            4104);
    TESTDIVN_FAST(MI_S32, -1054,           4104);
    TESTDIVN_FAST(MI_S32, -56402,          4104);
    TESTDIVN_FAST(MI_S32, (INT32_MIN + 1), 4104);

    {
        for (MI_U32 i = 0; i < 4104; i += 2)
        {
            MI_S32 dst[64], ref[64];
            MI_S32 *p_src = src_u + 32 * i;

            HVX_Vector vs32_u_0   = vmemu(p_src);
            HVX_Vector vs32_u_1   = vmemu(p_src + 32);
            HVX_VectorPair ws32_u = Q6_W_vcombine_VV(vs32_u_1, vs32_u_0);

            for (MI_U32 j = 0; j < 131328; j++)
            {
                MI_S32 divisor_s32 = divisor[j];
                HvxVdivnHelper<MI_S32> vdivn(divisor_s32);
                HVX_VectorPair w_dst = vdivn(ws32_u);

                vmemu(dst)      = Q6_V_lo_W(w_dst);
                vmemu(dst + 32) = Q6_V_hi_W(w_dst);

                for (MI_U32 k = 0; k < 64; k++)
                {
                    ref[k] = (0 == divisor_s32) ? 0 : (p_src[k] / divisor_s32);
                }

                ret_comm |= CHECK_CMP_VECTOR(ctx, dst, ref, 256);
            }
        }
    }

    if (Status::OK == ret_comm)
    {
        AURA_LOGD(ctx, AURA_TAG, "HvxVdivnHelper<MI_S32> test pass !\n");
    }
    else
    {
        AURA_LOGE(ctx, AURA_TAG, "HvxVdivnHelper<MI_S32> test fail !\n");
    }

    ret |= ret_comm;

EXIT:
    AURA_FREE(ctx, src_u);
    AURA_FREE(ctx, divisor);

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}