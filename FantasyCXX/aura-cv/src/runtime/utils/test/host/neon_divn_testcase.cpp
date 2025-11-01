#include "aura/tools/unit_test.h"

#if defined(AURA_ENABLE_NEON)
using namespace aura;

static Status NeonCheckDivnEqual(Context *ctx, void *dst, void *ref, MI_S32 size,
                                 const MI_CHAR *file, const MI_CHAR *func, MI_S32 line)
{
    Status ret = Status::OK;

    MI_U8 *dst_u8 = static_cast<MI_U8*>(dst);
    MI_U8 *ref_u8 = static_cast<MI_U8*>(ref);

    for (MI_S32 i = 0; i < size; i++)
    {
        ret |= TestCheckEQ(ctx, dst_u8[i], ref_u8[i], "CheckVectorEqual failed\n", file, func, line);
        if (ret != Status::OK)
        {
            AURA_LOGD(ctx, AURA_TAG, "error %d != %d\n", static_cast<MI_S32>(dst_u8[i]), static_cast<MI_S32>(ref_u8[i]));
            return ret;
        }
    }

    return ret;
}

#define CHECK_CMP_VECTOR(ctx, dst, ref, size) NeonCheckDivnEqual(ctx, dst, ref, size, __FILE__, __FUNCTION__, __LINE__)

#define TESTDIVN_FAST(type, v_type, ntype, v_ntype, divisor, item_number, function)          \
    for (MI_S32 item = 0; item < item_number; item++)                                        \
    {                                                                                        \
        constexpr MI_S32 compute_num = 16 / sizeof(type);                                    \
        type *p_src = src_u + item * compute_num;                                            \
        v_type vq_src = neon::vload1q(p_src);                                                \
        ntype dst[compute_num], ref[compute_num];                                            \
                                                                                             \
        v_ntype vq_result = neon::function<divisor>(vq_src);                                 \
        neon::vstore(dst, vq_result);                                                        \
                                                                                             \
        for (MI_S32 i = 0; i < compute_num; i++)                                             \
        {                                                                                    \
            ref[i] = SaturateCast<ntype>((0 == divisor) ? 0 : p_src[i] / divisor);           \
        }                                                                                    \
                                                                                             \
        MI_U8 size = sizeof(type) == sizeof(ntype) ? 16 : 8;                                 \
        Status ret_cmp = CHECK_CMP_VECTOR(ctx, dst, ref, size);                              \
        if (Status::OK == ret_cmp)                                                           \
        {                                                                                    \
            AURA_LOGD(ctx, AURA_TAG, "divsor is %d item %d :pass !\n", divisor, item);       \
        }                                                                                    \
        else                                                                                 \
        {                                                                                    \
            AURA_LOGE(ctx, AURA_TAG, "divsor is %d item %d :fail !\n", divisor, item);       \
        }                                                                                    \
        ret |= ret_cmp;                                                                      \
    }

NEW_TESTCASE(runtime_neon_instructions_divn_test_U8)
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

    TESTDIVN_FAST(MI_U8, uint8x16_t, MI_U8, uint8x16_t, 3,   16, vdiv_n);
    TESTDIVN_FAST(MI_U8, uint8x16_t, MI_U8, uint8x16_t, 5,   16, vdiv_n);
    TESTDIVN_FAST(MI_U8, uint8x16_t, MI_U8, uint8x16_t, 7,   16, vdiv_n);
    TESTDIVN_FAST(MI_U8, uint8x16_t, MI_U8, uint8x16_t, 9,   16, vdiv_n);
    TESTDIVN_FAST(MI_U8, uint8x16_t, MI_U8, uint8x16_t, 25,  16, vdiv_n);
    TESTDIVN_FAST(MI_U8, uint8x16_t, MI_U8, uint8x16_t, 49,  16, vdiv_n);

    TESTDIVN_FAST(MI_U8, uint8x16_t, MI_U8, uint8x16_t, 0,   16, vdiv_n);
    TESTDIVN_FAST(MI_U8, uint8x16_t, MI_U8, uint8x16_t, 2,   16, vdiv_n);
    TESTDIVN_FAST(MI_U8, uint8x16_t, MI_U8, uint8x16_t, 4,   16, vdiv_n);
    TESTDIVN_FAST(MI_U8, uint8x16_t, MI_U8, uint8x16_t, 6,   16, vdiv_n);
    TESTDIVN_FAST(MI_U8, uint8x16_t, MI_U8, uint8x16_t, 15,  16, vdiv_n);
    TESTDIVN_FAST(MI_U8, uint8x16_t, MI_U8, uint8x16_t, 90,  16, vdiv_n);
    TESTDIVN_FAST(MI_U8, uint8x16_t, MI_U8, uint8x16_t, 255, 16, vdiv_n);

    Status ret_comm = Status::OK;
    {
        for (MI_S32 i = 0; i < 16; i++)
        {
            MI_U8 *p_src = src_u + i * 16;
            uint8x16_t vu8_u = vld1q_u8(p_src);

            for (MI_S32 j = 0; j <= 255; j++)
            {
                MI_U8 dst[16], ref[16];
                MI_U8 divisor_u8 = divisor[j];
                neon::VdivNHelper<MI_U8> vdivn(divisor_u8);

                uint8x16_t vu8_result = vdivn(vu8_u);
                vst1q_u8(dst, vu8_result);

                for (MI_S32 i = 0; i < 16; i++)
                {
                    ref[i] = p_src[i] / divisor_u8;
                }

                ret |= CHECK_CMP_VECTOR(ctx, dst, ref, 16);
            }
        }
    }

    if (Status::OK == ret_comm)
    {
        AURA_LOGD(ctx, AURA_TAG, "VdivNHelper<MI_U8> test pass !\n");
    }
    else
    {
        AURA_LOGE(ctx, AURA_TAG, "VdivNHelper<MI_U8> test fail !\n");
    }

    ret |= ret_comm;
    AddTestResult(AURA_GET_TEST_STATUS(ret));
}

NEW_TESTCASE(runtime_neon_instructions_divn_test_S8)
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

    TESTDIVN_FAST(MI_S8, int8x16_t, MI_S8, int8x16_t, 3,    16, vdiv_n);
    TESTDIVN_FAST(MI_S8, int8x16_t, MI_S8, int8x16_t, 5,    16, vdiv_n);
    TESTDIVN_FAST(MI_S8, int8x16_t, MI_S8, int8x16_t, 7,    16, vdiv_n);
    TESTDIVN_FAST(MI_S8, int8x16_t, MI_S8, int8x16_t, 9,    16, vdiv_n);
    TESTDIVN_FAST(MI_S8, int8x16_t, MI_S8, int8x16_t, 25,   16, vdiv_n);
    TESTDIVN_FAST(MI_S8, int8x16_t, MI_S8, int8x16_t, 49,   16, vdiv_n);

    TESTDIVN_FAST(MI_S8, int8x16_t, MI_S8, int8x16_t, 0,    16, vdiv_n);
    TESTDIVN_FAST(MI_S8, int8x16_t, MI_S8, int8x16_t, 2,    16, vdiv_n);
    TESTDIVN_FAST(MI_S8, int8x16_t, MI_S8, int8x16_t, 4,    16, vdiv_n);
    TESTDIVN_FAST(MI_S8, int8x16_t, MI_S8, int8x16_t, 6,    16, vdiv_n);
    TESTDIVN_FAST(MI_S8, int8x16_t, MI_S8, int8x16_t, 15,   16, vdiv_n);
    TESTDIVN_FAST(MI_S8, int8x16_t, MI_S8, int8x16_t, 59,   16, vdiv_n);
    TESTDIVN_FAST(MI_S8, int8x16_t, MI_S8, int8x16_t, 127,  16, vdiv_n);

    TESTDIVN_FAST(MI_S8, int8x16_t, MI_S8, int8x16_t, -2,   16, vdiv_n);
    TESTDIVN_FAST(MI_S8, int8x16_t, MI_S8, int8x16_t, -3,   16, vdiv_n);
    TESTDIVN_FAST(MI_S8, int8x16_t, MI_S8, int8x16_t, -4,   16, vdiv_n);
    TESTDIVN_FAST(MI_S8, int8x16_t, MI_S8, int8x16_t, -5,   16, vdiv_n);
    TESTDIVN_FAST(MI_S8, int8x16_t, MI_S8, int8x16_t, -13,  16, vdiv_n);
    TESTDIVN_FAST(MI_S8, int8x16_t, MI_S8, int8x16_t, -57,  16, vdiv_n);
    TESTDIVN_FAST(MI_S8, int8x16_t, MI_S8, int8x16_t, -128, 16, vdiv_n);

    Status ret_comm = Status::OK;
    {
        for (MI_S32 i = 0; i < 16; i++)
        {
            MI_S8 *p_src = src_u + i * 16;
            int8x16_t vs8_u = vld1q_s8(p_src);

            for (MI_S32 j = 0; j < 256; j++)
            {
                MI_S8 dst[16], ref[16];
                MI_S8 divisor_s8 = divisor[j];
                neon::VdivNHelper<MI_S8> vdivn(divisor_s8);

                int8x16_t vs8_result = vdivn(vs8_u);
                vst1q_s8(dst, vs8_result);

                for (MI_S32 i = 0; i < 16; i++)
                {
                    ref[i] = p_src[i] / divisor_s8;
                }

                ret |= CHECK_CMP_VECTOR(ctx, dst, ref, 16);
            }
        }
    }

    if (Status::OK == ret_comm)
    {
        AURA_LOGD(ctx, AURA_TAG, "VdivNHelper<MI_S8> test pass !\n");
    }
    else
    {
        AURA_LOGE(ctx, AURA_TAG, "VdivNHelper<MI_S8> test fail !\n");
    }

    ret |= ret_comm;

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}

NEW_TESTCASE(runtime_neon_instructions_divn_test_U16)
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

    TESTDIVN_FAST(MI_U16, uint16x8_t, MI_U8,  uint8x8_t,  3,     8192, vqdivn_n);
    TESTDIVN_FAST(MI_U16, uint16x8_t, MI_U8,  uint8x8_t,  5,     8192, vqdivn_n);
    TESTDIVN_FAST(MI_U16, uint16x8_t, MI_U8,  uint8x8_t,  7,     8192, vqdivn_n);
    TESTDIVN_FAST(MI_U16, uint16x8_t, MI_U8,  uint8x8_t,  9,     8192, vqdivn_n);
    TESTDIVN_FAST(MI_U16, uint16x8_t, MI_U8,  uint8x8_t,  25,    8192, vqdivn_n);
    TESTDIVN_FAST(MI_U16, uint16x8_t, MI_U8,  uint8x8_t,  49,    8192, vqdivn_n);

    TESTDIVN_FAST(MI_U16, uint16x8_t, MI_U8,  uint8x8_t,  0,     8192, vqdivn_n);
    TESTDIVN_FAST(MI_U16, uint16x8_t, MI_U8,  uint8x8_t,  2,     8192, vqdivn_n);
    TESTDIVN_FAST(MI_U16, uint16x8_t, MI_U8,  uint8x8_t,  4,     8192, vqdivn_n);
    TESTDIVN_FAST(MI_U16, uint16x8_t, MI_U8,  uint8x8_t,  5,     8192, vqdivn_n);
    TESTDIVN_FAST(MI_U16, uint16x8_t, MI_U8,  uint8x8_t,  6542,  8192, vqdivn_n);
    TESTDIVN_FAST(MI_U16, uint16x8_t, MI_U8,  uint8x8_t,  10354, 8192, vqdivn_n);
    TESTDIVN_FAST(MI_U16, uint16x8_t, MI_U8,  uint8x8_t,  65535, 8192, vqdivn_n);

    TESTDIVN_FAST(MI_U16, uint16x8_t, MI_U16, uint16x8_t, 3,     8192, vdiv_n);
    TESTDIVN_FAST(MI_U16, uint16x8_t, MI_U16, uint16x8_t, 5,     8192, vdiv_n);
    TESTDIVN_FAST(MI_U16, uint16x8_t, MI_U16, uint16x8_t, 7,     8192, vdiv_n);
    TESTDIVN_FAST(MI_U16, uint16x8_t, MI_U16, uint16x8_t, 9,     8192, vdiv_n);
    TESTDIVN_FAST(MI_U16, uint16x8_t, MI_U16, uint16x8_t, 25,    8192, vdiv_n);
    TESTDIVN_FAST(MI_U16, uint16x8_t, MI_U16, uint16x8_t, 49,    8192, vdiv_n);

    TESTDIVN_FAST(MI_U16, uint16x8_t, MI_U16, uint16x8_t, 0,     8192, vdiv_n);
    TESTDIVN_FAST(MI_U16, uint16x8_t, MI_U16, uint16x8_t, 2,     8192, vdiv_n);
    TESTDIVN_FAST(MI_U16, uint16x8_t, MI_U16, uint16x8_t, 4,     8192, vdiv_n);
    TESTDIVN_FAST(MI_U16, uint16x8_t, MI_U16, uint16x8_t, 5,     8192, vdiv_n);
    TESTDIVN_FAST(MI_U16, uint16x8_t, MI_U16, uint16x8_t, 6542,  8192, vdiv_n);
    TESTDIVN_FAST(MI_U16, uint16x8_t, MI_U16, uint16x8_t, 10354, 8192, vdiv_n);
    TESTDIVN_FAST(MI_U16, uint16x8_t, MI_U16, uint16x8_t, 65535, 8192, vdiv_n);

    for (MI_S32 i = 0; i < 8192; i++)
    {
        MI_U16 dst[8], ref[8];
        MI_U16 *p_src = src_u + i * 8;
        uint16x8_t vu16_u = vld1q_u16(p_src);
        for (MI_S32 j = 0; j <= 65535; j++)
        {
            MI_U16 divisor_u16 = divisor[j];
            neon::VdivNHelper<MI_U16> vdivn(divisor_u16);
            uint16x8_t vu16_result = vdivn(vu16_u);
            vst1q_u16(dst, vu16_result);

            for (MI_S32 i = 0; i < 8; i++)
            {
                ref[i] = p_src[i] / divisor_u16;
            }

            ret_comm |= CHECK_CMP_VECTOR(ctx, dst, ref, 16);
        }
    }

    for (MI_S32 i = 0; i < 8192; i++)
    {
        MI_U8 dst[8], ref[8];
        MI_U16 *p_src = src_u + i * 8;
        uint16x8_t vu16_u = vld1q_u16(p_src);
        for (MI_S32 j = 0; j <= 65535; j++)
        {
            MI_U16 divisor_u16 = divisor[j];
            neon::VqdivnNHelper<MI_U16> vdivn(divisor_u16);
            uint8x8_t vu8_result = vdivn(vu16_u);
            vst1_u8(dst, vu8_result);

            for (MI_S32 i = 0; i < 8; i++)
            {
                ref[i] = SaturateCast<MI_U8>((0 == divisor_u16) ? 0 : p_src[i] / divisor_u16);
            }

            ret_comm |= CHECK_CMP_VECTOR(ctx, dst, ref, 8);
        }
    }

    if (Status::OK == ret_comm)
    {
        AURA_LOGD(ctx, AURA_TAG, "VdivNHelper<MI_U16> test pass !\n");
    }
    else
    {
        AURA_LOGE(ctx, AURA_TAG, "VdivNHelper<MI_U16> test fail !\n");
    }

    ret |= ret_comm;
EXIT:
    AURA_FREE(ctx, src_u);
    AURA_FREE(ctx, divisor);

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}

NEW_TESTCASE(runtime_neon_instructions_divn_test_S16)
{
    Status ret      = Status::OK;
    Status ret_comm = Status::OK;
    Context *ctx = UnitTest::GetInstance()->GetContext();

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

    TESTDIVN_FAST(MI_S16, int16x8_t, MI_S8,  int8x8_t,  3,      8192, vqdivn_n);
    TESTDIVN_FAST(MI_S16, int16x8_t, MI_S8,  int8x8_t,  5,      8192, vqdivn_n);
    TESTDIVN_FAST(MI_S16, int16x8_t, MI_S8,  int8x8_t,  7,      8192, vqdivn_n);
    TESTDIVN_FAST(MI_S16, int16x8_t, MI_S8,  int8x8_t,  9,      8192, vqdivn_n);
    TESTDIVN_FAST(MI_S16, int16x8_t, MI_S8,  int8x8_t,  25,     8192, vqdivn_n);
    TESTDIVN_FAST(MI_S16, int16x8_t, MI_S8,  int8x8_t,  49,     8192, vqdivn_n);

    TESTDIVN_FAST(MI_S16, int16x8_t, MI_S8,  int8x8_t,  0,      8192, vqdivn_n);
    TESTDIVN_FAST(MI_S16, int16x8_t, MI_S8,  int8x8_t,  15,     8192, vqdivn_n);
    TESTDIVN_FAST(MI_S16, int16x8_t, MI_S8,  int8x8_t,  364,    8192, vqdivn_n);
    TESTDIVN_FAST(MI_S16, int16x8_t, MI_S8,  int8x8_t,  6542,   8192, vqdivn_n);
    TESTDIVN_FAST(MI_S16, int16x8_t, MI_S8,  int8x8_t,  10245,  8192, vqdivn_n);
    TESTDIVN_FAST(MI_S16, int16x8_t, MI_S8,  int8x8_t,  23354,  8192, vqdivn_n);

    TESTDIVN_FAST(MI_S16, int16x8_t, MI_S8,  int8x8_t,  -2,     8192, vqdivn_n);
    TESTDIVN_FAST(MI_S16, int16x8_t, MI_S8,  int8x8_t,  -42,    8192, vqdivn_n);
    TESTDIVN_FAST(MI_S16, int16x8_t, MI_S8,  int8x8_t,  -436,   8192, vqdivn_n);
    TESTDIVN_FAST(MI_S16, int16x8_t, MI_S8,  int8x8_t,  -5432,  8192, vqdivn_n);
    TESTDIVN_FAST(MI_S16, int16x8_t, MI_S8,  int8x8_t,  -15154, 8192, vqdivn_n);
    TESTDIVN_FAST(MI_S16, int16x8_t, MI_S8,  int8x8_t,  -32768, 8192, vqdivn_n);

    TESTDIVN_FAST(MI_S16, int16x8_t, MI_S16, int16x8_t, 3,      8192, vdiv_n);
    TESTDIVN_FAST(MI_S16, int16x8_t, MI_S16, int16x8_t, 5,      8192, vdiv_n);
    TESTDIVN_FAST(MI_S16, int16x8_t, MI_S16, int16x8_t, 7,      8192, vdiv_n);
    TESTDIVN_FAST(MI_S16, int16x8_t, MI_S16, int16x8_t, 9,      8192, vdiv_n);
    TESTDIVN_FAST(MI_S16, int16x8_t, MI_S16, int16x8_t, 25,     8192, vdiv_n);
    TESTDIVN_FAST(MI_S16, int16x8_t, MI_S16, int16x8_t, 49,     8192, vdiv_n);

    TESTDIVN_FAST(MI_S16, int16x8_t, MI_S16, int16x8_t, 0,      8192, vdiv_n);
    TESTDIVN_FAST(MI_S16, int16x8_t, MI_S16, int16x8_t, 15,     8192, vdiv_n);
    TESTDIVN_FAST(MI_S16, int16x8_t, MI_S16, int16x8_t, 364,    8192, vdiv_n);
    TESTDIVN_FAST(MI_S16, int16x8_t, MI_S16, int16x8_t, 6542,   8192, vdiv_n);
    TESTDIVN_FAST(MI_S16, int16x8_t, MI_S16, int16x8_t, 10245,  8192, vdiv_n);
    TESTDIVN_FAST(MI_S16, int16x8_t, MI_S16, int16x8_t, 23354,  8192, vdiv_n);

    TESTDIVN_FAST(MI_S16, int16x8_t, MI_S16, int16x8_t, -2,     8192, vdiv_n);
    TESTDIVN_FAST(MI_S16, int16x8_t, MI_S16, int16x8_t, -42,    8192, vdiv_n);
    TESTDIVN_FAST(MI_S16, int16x8_t, MI_S16, int16x8_t, -436,   8192, vdiv_n);
    TESTDIVN_FAST(MI_S16, int16x8_t, MI_S16, int16x8_t, -5432,  8192, vdiv_n);
    TESTDIVN_FAST(MI_S16, int16x8_t, MI_S16, int16x8_t, -15154, 8192, vdiv_n);
    TESTDIVN_FAST(MI_S16, int16x8_t, MI_S16, int16x8_t, -32768, 8192, vdiv_n);

    for (MI_S32 i = 0; i < 8192; i++)
    {
        MI_S16 dst[8], ref[8];
        MI_S16 *p_src = src_u + i * 8;
        int16x8_t vs16_u = vld1q_s16(p_src);

        for (MI_S32 j = 0; j <= 65535; j++)
        {
            MI_S16 divisor_s16 = divisor[j];
            neon::VdivNHelper<MI_S16> vdivn(divisor_s16);
            int16x8_t vs16_result = vdivn(vs16_u);
            vst1q_s16(dst, vs16_result);

            for (MI_S32 i = 0; i < 8; i++)
            {
                ref[i] = p_src[i] / divisor_s16;
            }

            ret_comm |= CHECK_CMP_VECTOR(ctx, dst, ref, 16);
        }
    }

    for (MI_S32 i = 0; i < 8192; i++)
    {
        MI_S8 dst[8], ref[8];
        MI_S16 *p_src = src_u + i * 8;
        int16x8_t vs16_u = vld1q_s16(p_src);

        for (MI_S32 j = 0; j <= 65535; j++)
        {
            MI_S16 divisor_s16 = divisor[j];
            neon::VqdivnNHelper<MI_S16> vdivn(divisor_s16);
            int8x8_t vs8_result = vdivn(vs16_u);
            vst1_s8(dst, vs8_result);

            for (MI_S32 i = 0; i < 8; i++)
            {
                ref[i] = SaturateCast<MI_S8>((0 == divisor_s16) ? 0 : p_src[i] / divisor_s16);
            }


            ret_comm |= CHECK_CMP_VECTOR(ctx, dst, ref, 8);
        }
    }

    if (Status::OK == ret_comm)
    {
        AURA_LOGD(ctx, AURA_TAG, "VdivNHelper<MI_S16> test pass !\n");
    }
    else
    {
        AURA_LOGE(ctx, AURA_TAG, "VdivNHelper<MI_S16> test fail !\n");
    }

    ret |= ret_comm;
EXIT:
    AURA_FREE(ctx, src_u);
    AURA_FREE(ctx, divisor);

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}

NEW_TESTCASE(runtime_neon_instructions_divn_test_U32)
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

    TESTDIVN_FAST(MI_U32, uint32x4_t, MI_U16, uint16x4_t, 3,                32832, vqdivn_n);
    TESTDIVN_FAST(MI_U32, uint32x4_t, MI_U16, uint16x4_t, 5,                32832, vqdivn_n);
    TESTDIVN_FAST(MI_U32, uint32x4_t, MI_U16, uint16x4_t, 7,                32832, vqdivn_n);
    TESTDIVN_FAST(MI_U32, uint32x4_t, MI_U16, uint16x4_t, 9,                32832, vqdivn_n);
    TESTDIVN_FAST(MI_U32, uint32x4_t, MI_U16, uint16x4_t, 25,               32832, vqdivn_n);
    TESTDIVN_FAST(MI_U32, uint32x4_t, MI_U16, uint16x4_t, 49,               32832, vqdivn_n);

    TESTDIVN_FAST(MI_U32, uint32x4_t, MI_U16, uint16x4_t, 0,                32832, vqdivn_n);
    TESTDIVN_FAST(MI_U32, uint32x4_t, MI_U16, uint16x4_t, 12,               32832, vqdivn_n);
    TESTDIVN_FAST(MI_U32, uint32x4_t, MI_U16, uint16x4_t, 543,              32832, vqdivn_n);
    TESTDIVN_FAST(MI_U32, uint32x4_t, MI_U16, uint16x4_t, 5251,             32832, vqdivn_n);
    TESTDIVN_FAST(MI_U32, uint32x4_t, MI_U16, uint16x4_t, 43242,            32832, vqdivn_n);
    TESTDIVN_FAST(MI_U32, uint32x4_t, MI_U16, uint16x4_t, (UINT32_MAX - 1), 32832, vqdivn_n);

    TESTDIVN_FAST(MI_U32, uint32x4_t, MI_U32, uint32x4_t, 3,                32832, vdiv_n);
    TESTDIVN_FAST(MI_U32, uint32x4_t, MI_U32, uint32x4_t, 5,                32832, vdiv_n);
    TESTDIVN_FAST(MI_U32, uint32x4_t, MI_U32, uint32x4_t, 7,                32832, vdiv_n);
    TESTDIVN_FAST(MI_U32, uint32x4_t, MI_U32, uint32x4_t, 9,                32832, vdiv_n);
    TESTDIVN_FAST(MI_U32, uint32x4_t, MI_U32, uint32x4_t, 25,               32832, vdiv_n);
    TESTDIVN_FAST(MI_U32, uint32x4_t, MI_U32, uint32x4_t, 49,               32832, vdiv_n);

    TESTDIVN_FAST(MI_U32, uint32x4_t, MI_U32, uint32x4_t, 0,                32832, vdiv_n);
    TESTDIVN_FAST(MI_U32, uint32x4_t, MI_U32, uint32x4_t, 12,               32832, vdiv_n);
    TESTDIVN_FAST(MI_U32, uint32x4_t, MI_U32, uint32x4_t, 543,              32832, vdiv_n);
    TESTDIVN_FAST(MI_U32, uint32x4_t, MI_U32, uint32x4_t, 5251,             32832, vdiv_n);
    TESTDIVN_FAST(MI_U32, uint32x4_t, MI_U32, uint32x4_t, 43242,            32832, vdiv_n);
    TESTDIVN_FAST(MI_U32, uint32x4_t, MI_U32, uint32x4_t, (UINT32_MAX - 1), 32832, vdiv_n);

    for (MI_U32 i = 0; i < 32832; i++)
    {
        MI_U32 dst[4], ref[4];
        MI_U32 *p_src = src_u + 4 * i;
        uint32x4_t vu32_u = vld1q_u32(p_src);

        for (MI_U32 j = 0; j < 131328; j++)
        {
            MI_U32 divisor_u32 = divisor[j];
            neon::VdivNHelper<MI_U32> vdivn(divisor_u32);
            uint32x4_t vu32_result = vdivn(vu32_u);
            vst1q_u32(dst, vu32_result);

            for (MI_U32 k = 0; k < 4; k++)
            {
                ref[k] = p_src[k] / divisor_u32;
            }

            ret_comm |= CHECK_CMP_VECTOR(ctx, dst, ref, 16);
        }
    }

    for (MI_U32 i = 0; i < 32832; i++)
    {
        MI_U16 dst[4], ref[4];
        MI_U32 *p_src = src_u + 4 * i;
        uint32x4_t vu32_u = vld1q_u32(p_src);

        for (MI_U32 j = 0; j < 131328; j++)
        {
            MI_U32 divisor_u32 = divisor[j];
            neon::VqdivnNHelper<MI_U32> vdivn(divisor_u32);
            uint16x4_t vu16_result = vdivn(vu32_u);
            vst1_u16(dst, vu16_result);

            for (MI_U32 k = 0; k < 4; k++)
            {
                ref[k] = SaturateCast<MI_U16>((0 == divisor_u32) ? 0 : p_src[k] / divisor_u32);
            }

            ret_comm |= CHECK_CMP_VECTOR(ctx, dst, ref, 8);
        }
    }

    if (Status::OK == ret)
    {
        AURA_LOGD(ctx, AURA_TAG, "VdivNHelper<MI_U32> test pass !\n");
    }
    else
    {
        AURA_LOGE(ctx, AURA_TAG, "VdivNHelper<MI_U32> test fail !\n");
    }

    ret |= ret_comm;
EXIT:
    AURA_FREE(ctx, src_u);
    AURA_FREE(ctx, divisor);

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}

NEW_TESTCASE(runtime_neon_instructions_divn_test_S32)
{
    Status ret      = Status::OK;
    Status ret_comm = Status::OK;
    Context *ctx = UnitTest::GetInstance()->GetContext();

    MI_S32 *src_u   = static_cast<MI_S32*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, ((2 * 65536  + 256) * sizeof(MI_S32)), 0));
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

    TESTDIVN_FAST(MI_S32, int32x4_t, MI_S16, int16x4_t, 0,               32832, vqdivn_n);
    TESTDIVN_FAST(MI_S32, int32x4_t, MI_S16, int16x4_t, 2,               32832, vqdivn_n);
    TESTDIVN_FAST(MI_S32, int32x4_t, MI_S16, int16x4_t, 34,              32832, vqdivn_n);
    TESTDIVN_FAST(MI_S32, int32x4_t, MI_S16, int16x4_t, 535,             32832, vqdivn_n);
    TESTDIVN_FAST(MI_S32, int32x4_t, MI_S16, int16x4_t, 2342,            32832, vqdivn_n);
    TESTDIVN_FAST(MI_S32, int32x4_t, MI_S16, int16x4_t, 73523,           32832, vqdivn_n);
    TESTDIVN_FAST(MI_S32, int32x4_t, MI_S16, int16x4_t, 544459,          32832, vqdivn_n);
    TESTDIVN_FAST(MI_S32, int32x4_t, MI_S16, int16x4_t, (INT32_MAX - 1), 32832, vqdivn_n);
    TESTDIVN_FAST(MI_S32, int32x4_t, MI_S16, int16x4_t, -2,              32832, vqdivn_n);
    TESTDIVN_FAST(MI_S32, int32x4_t, MI_S16, int16x4_t, -54,             32832, vqdivn_n);
    TESTDIVN_FAST(MI_S32, int32x4_t, MI_S16, int16x4_t, -975,            32832, vqdivn_n);
    TESTDIVN_FAST(MI_S32, int32x4_t, MI_S16, int16x4_t, -1054,           32832, vqdivn_n);
    TESTDIVN_FAST(MI_S32, int32x4_t, MI_S16, int16x4_t, -56402,          32832, vqdivn_n);
    TESTDIVN_FAST(MI_S32, int32x4_t, MI_S16, int16x4_t, (INT32_MIN + 1), 32832, vqdivn_n);
    TESTDIVN_FAST(MI_S32, int32x4_t, MI_S16, int16x4_t, (INT32_MAX - 6), 8192,  vqdivn_n);
    TESTDIVN_FAST(MI_S32, int32x4_t, MI_S16, int16x4_t, -55662,          8192,  vqdivn_n);

    TESTDIVN_FAST(MI_S32, int32x4_t, MI_S32, int32x4_t, (INT32_MAX - 6), 8192,  vdiv_n);
    TESTDIVN_FAST(MI_S32, int32x4_t, MI_S32, int32x4_t, -55662,          8192,  vdiv_n);

    TESTDIVN_FAST(MI_S32, int32x4_t, MI_S32, int32x4_t, 0,               32832, vdiv_n);
    TESTDIVN_FAST(MI_S32, int32x4_t, MI_S32, int32x4_t, 2,               32832, vdiv_n);
    TESTDIVN_FAST(MI_S32, int32x4_t, MI_S32, int32x4_t, 34,              32832, vdiv_n);
    TESTDIVN_FAST(MI_S32, int32x4_t, MI_S32, int32x4_t, 535,             32832, vdiv_n);
    TESTDIVN_FAST(MI_S32, int32x4_t, MI_S32, int32x4_t, 2342,            32832, vdiv_n);
    TESTDIVN_FAST(MI_S32, int32x4_t, MI_S32, int32x4_t, 73523,           32832, vdiv_n);
    TESTDIVN_FAST(MI_S32, int32x4_t, MI_S32, int32x4_t, 544459,          32832, vdiv_n);
    TESTDIVN_FAST(MI_S32, int32x4_t, MI_S32, int32x4_t, (INT32_MAX - 1), 32832, vdiv_n);

    TESTDIVN_FAST(MI_S32, int32x4_t, MI_S32, int32x4_t, -2,              32832, vdiv_n);
    TESTDIVN_FAST(MI_S32, int32x4_t, MI_S32, int32x4_t, -54,             32832, vdiv_n);
    TESTDIVN_FAST(MI_S32, int32x4_t, MI_S32, int32x4_t, -975,            32832, vdiv_n);
    TESTDIVN_FAST(MI_S32, int32x4_t, MI_S32, int32x4_t, -1054,           32832, vdiv_n);
    TESTDIVN_FAST(MI_S32, int32x4_t, MI_S32, int32x4_t, -56402,          32832, vdiv_n);
    TESTDIVN_FAST(MI_S32, int32x4_t, MI_S32, int32x4_t, (INT32_MIN + 1), 32832, vdiv_n);

    for (MI_U32 i = 0; i < 32832; i++)
    {
        MI_S32 dst[4], ref[4];
        MI_S32 *p_src = src_u + 4 * i;
        int32x4_t vs32_u = vld1q_s32(p_src);

        for (MI_U32 j = 0; j < 131328; j++)
        {
            MI_S32 divisor_s32 = divisor[j];
            neon::VdivNHelper<MI_S32> vdivn(divisor_s32);
            int32x4_t vs32_result = vdivn(vs32_u);
            vst1q_s32(dst, vs32_result);

            for (MI_U32 k = 0; k < 4; k++)
            {
                ref[k] = p_src[k] / divisor_s32;
            }

            ret_comm |= CHECK_CMP_VECTOR(ctx, dst, ref, 16);
        }
    }

    for (MI_U32 i = 0; i < 32832; i++)
    {
        MI_S16 dst[4], ref[4];
        MI_S32 *p_src = src_u + 4 * i;
        int32x4_t vs32_u = vld1q_s32(p_src);

        for (MI_U32 j = 0; j < 131328; j++)
        {
            MI_S32 divisor_s32 = divisor[j];
            neon::VqdivnNHelper<MI_S32> vdivn(divisor_s32);
            int16x4_t vs16_result = vdivn(vs32_u);
            vst1_s16(dst, vs16_result);

            for (MI_U32 k = 0; k < 4; k++)
            {
                ref[k] = SaturateCast<MI_S16>((0 == divisor_s32) ? 0 : p_src[k] / divisor_s32);
            }

            ret_comm |= CHECK_CMP_VECTOR(ctx, dst, ref, 8);
        }
    }

    if (Status::OK == ret_comm)
    {
        AURA_LOGD(ctx, AURA_TAG, "VdivNHelper<MI_S32> test pass !\n");
    }
    else
    {
        AURA_LOGE(ctx, AURA_TAG, "VdivNHelper<MI_S32> test fail !\n");
    }

    ret |= ret_comm;
EXIT:
    AURA_FREE(ctx, src_u);
    AURA_FREE(ctx, divisor);

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}

#  endif