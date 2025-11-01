#include "gemm_impl.hpp"
#include "aura/runtime/logger.h"

namespace aura
{

AURA_INLINE AURA_VOID AddDotNeon8x8(MI_S32 k, const MI_F32 *a, MI_S32 lda, const MI_F32 *b, MI_S32 ldb, MI_F32 *c, MI_S32 ldc)
{
    const MI_F32 *ptr_a0 = a + lda * 0;
    const MI_F32 *ptr_a1 = a + lda * 1;
    const MI_F32 *ptr_a2 = a + lda * 2;
    const MI_F32 *ptr_a3 = a + lda * 3;
    const MI_F32 *ptr_a4 = a + lda * 4;
    const MI_F32 *ptr_a5 = a + lda * 5;
    const MI_F32 *ptr_a6 = a + lda * 6;
    const MI_F32 *ptr_a7 = a + lda * 7;

    float32x4_t vqf32_sum00; neon::vdup(vqf32_sum00, 0);
    float32x4_t vqf32_sum01; neon::vdup(vqf32_sum01, 0);
    float32x4_t vqf32_sum02; neon::vdup(vqf32_sum02, 0);
    float32x4_t vqf32_sum03; neon::vdup(vqf32_sum03, 0);
    float32x4_t vqf32_sum04; neon::vdup(vqf32_sum04, 0);
    float32x4_t vqf32_sum05; neon::vdup(vqf32_sum05, 0);
    float32x4_t vqf32_sum06; neon::vdup(vqf32_sum06, 0);
    float32x4_t vqf32_sum07; neon::vdup(vqf32_sum07, 0);

    float32x4_t vqf32_sum40; neon::vdup(vqf32_sum40, 0);
    float32x4_t vqf32_sum41; neon::vdup(vqf32_sum41, 0);
    float32x4_t vqf32_sum42; neon::vdup(vqf32_sum42, 0);
    float32x4_t vqf32_sum43; neon::vdup(vqf32_sum43, 0);
    float32x4_t vqf32_sum44; neon::vdup(vqf32_sum44, 0);
    float32x4_t vqf32_sum45; neon::vdup(vqf32_sum45, 0);
    float32x4_t vqf32_sum46; neon::vdup(vqf32_sum46, 0);
    float32x4_t vqf32_sum47; neon::vdup(vqf32_sum47, 0);

    float32x4_t vqf32_a0;
    float32x4_t vqf32_a1;
    float32x4_t vqf32_a2;
    float32x4_t vqf32_a3;
    float32x4_t vqf32_a4;
    float32x4_t vqf32_a5;
    float32x4_t vqf32_a6;
    float32x4_t vqf32_a7;

    float32x4_t vqf32_b0;
    float32x4_t vqf32_b1;

    MI_F32 ta0 = 0;
    MI_F32 ta1 = 0;
    MI_F32 ta2 = 0;
    MI_F32 ta3 = 0;
    MI_F32 ta4 = 0;
    MI_F32 ta5 = 0;
    MI_F32 ta6 = 0;
    MI_F32 ta7 = 0;

    MI_S32 k_align4 = (k & (-4));

    MI_S32 p = 0;

    for (; p < k_align4; p += 4)
    {
        neon::vload(ptr_a0 + p, vqf32_a0);
        neon::vload(ptr_a1 + p, vqf32_a1);
        neon::vload(ptr_a2 + p, vqf32_a2);
        neon::vload(ptr_a3 + p, vqf32_a3);
        neon::vload(ptr_a4 + p, vqf32_a4);
        neon::vload(ptr_a5 + p, vqf32_a5);
        neon::vload(ptr_a6 + p, vqf32_a6);
        neon::vload(ptr_a7 + p, vqf32_a7);

        vqf32_b0 = neon::vload1q(b + ldb * (p + 0));
        vqf32_b1 = neon::vload1q(b + ldb * (p + 0) + 4);

        ta0 = neon::vgetlane<0>(vqf32_a0);
        ta1 = neon::vgetlane<0>(vqf32_a1);
        ta2 = neon::vgetlane<0>(vqf32_a2);
        ta3 = neon::vgetlane<0>(vqf32_a3);
        ta4 = neon::vgetlane<0>(vqf32_a4);
        ta5 = neon::vgetlane<0>(vqf32_a5);
        ta6 = neon::vgetlane<0>(vqf32_a6);
        ta7 = neon::vgetlane<0>(vqf32_a7);

        vqf32_sum00 = neon::vmla(vqf32_sum00, vqf32_b0, ta0);
        vqf32_sum01 = neon::vmla(vqf32_sum01, vqf32_b0, ta1);
        vqf32_sum02 = neon::vmla(vqf32_sum02, vqf32_b0, ta2);
        vqf32_sum03 = neon::vmla(vqf32_sum03, vqf32_b0, ta3);
        vqf32_sum04 = neon::vmla(vqf32_sum04, vqf32_b0, ta4);
        vqf32_sum05 = neon::vmla(vqf32_sum05, vqf32_b0, ta5);
        vqf32_sum06 = neon::vmla(vqf32_sum06, vqf32_b0, ta6);
        vqf32_sum07 = neon::vmla(vqf32_sum07, vqf32_b0, ta7);

        vqf32_sum40 = neon::vmla(vqf32_sum40, vqf32_b1, ta0);
        vqf32_sum41 = neon::vmla(vqf32_sum41, vqf32_b1, ta1);
        vqf32_sum42 = neon::vmla(vqf32_sum42, vqf32_b1, ta2);
        vqf32_sum43 = neon::vmla(vqf32_sum43, vqf32_b1, ta3);
        vqf32_sum44 = neon::vmla(vqf32_sum44, vqf32_b1, ta4);
        vqf32_sum45 = neon::vmla(vqf32_sum45, vqf32_b1, ta5);
        vqf32_sum46 = neon::vmla(vqf32_sum46, vqf32_b1, ta6);
        vqf32_sum47 = neon::vmla(vqf32_sum47, vqf32_b1, ta7);

        vqf32_b0 = neon::vload1q(b + ldb * (p + 1));
        vqf32_b1 = neon::vload1q(b + ldb * (p + 1) + 4);

        ta0 = neon::vgetlane<1>(vqf32_a0);
        ta1 = neon::vgetlane<1>(vqf32_a1);
        ta2 = neon::vgetlane<1>(vqf32_a2);
        ta3 = neon::vgetlane<1>(vqf32_a3);
        ta4 = neon::vgetlane<1>(vqf32_a4);
        ta5 = neon::vgetlane<1>(vqf32_a5);
        ta6 = neon::vgetlane<1>(vqf32_a6);
        ta7 = neon::vgetlane<1>(vqf32_a7);

        vqf32_sum00 = neon::vmla(vqf32_sum00, vqf32_b0, ta0);
        vqf32_sum01 = neon::vmla(vqf32_sum01, vqf32_b0, ta1);
        vqf32_sum02 = neon::vmla(vqf32_sum02, vqf32_b0, ta2);
        vqf32_sum03 = neon::vmla(vqf32_sum03, vqf32_b0, ta3);
        vqf32_sum04 = neon::vmla(vqf32_sum04, vqf32_b0, ta4);
        vqf32_sum05 = neon::vmla(vqf32_sum05, vqf32_b0, ta5);
        vqf32_sum06 = neon::vmla(vqf32_sum06, vqf32_b0, ta6);
        vqf32_sum07 = neon::vmla(vqf32_sum07, vqf32_b0, ta7);

        vqf32_sum40 = neon::vmla(vqf32_sum40, vqf32_b1, ta0);
        vqf32_sum41 = neon::vmla(vqf32_sum41, vqf32_b1, ta1);
        vqf32_sum42 = neon::vmla(vqf32_sum42, vqf32_b1, ta2);
        vqf32_sum43 = neon::vmla(vqf32_sum43, vqf32_b1, ta3);
        vqf32_sum44 = neon::vmla(vqf32_sum44, vqf32_b1, ta4);
        vqf32_sum45 = neon::vmla(vqf32_sum45, vqf32_b1, ta5);
        vqf32_sum46 = neon::vmla(vqf32_sum46, vqf32_b1, ta6);
        vqf32_sum47 = neon::vmla(vqf32_sum47, vqf32_b1, ta7);

        vqf32_b0 = neon::vload1q(b + ldb * (p + 2));
        vqf32_b1 = neon::vload1q(b + ldb * (p + 2) + 4);

        ta0 = neon::vgetlane<2>(vqf32_a0);
        ta1 = neon::vgetlane<2>(vqf32_a1);
        ta2 = neon::vgetlane<2>(vqf32_a2);
        ta3 = neon::vgetlane<2>(vqf32_a3);
        ta4 = neon::vgetlane<2>(vqf32_a4);
        ta5 = neon::vgetlane<2>(vqf32_a5);
        ta6 = neon::vgetlane<2>(vqf32_a6);
        ta7 = neon::vgetlane<2>(vqf32_a7);

        vqf32_sum00 = neon::vmla(vqf32_sum00, vqf32_b0, ta0);
        vqf32_sum01 = neon::vmla(vqf32_sum01, vqf32_b0, ta1);
        vqf32_sum02 = neon::vmla(vqf32_sum02, vqf32_b0, ta2);
        vqf32_sum03 = neon::vmla(vqf32_sum03, vqf32_b0, ta3);
        vqf32_sum04 = neon::vmla(vqf32_sum04, vqf32_b0, ta4);
        vqf32_sum05 = neon::vmla(vqf32_sum05, vqf32_b0, ta5);
        vqf32_sum06 = neon::vmla(vqf32_sum06, vqf32_b0, ta6);
        vqf32_sum07 = neon::vmla(vqf32_sum07, vqf32_b0, ta7);

        vqf32_sum40 = neon::vmla(vqf32_sum40, vqf32_b1, ta0);
        vqf32_sum41 = neon::vmla(vqf32_sum41, vqf32_b1, ta1);
        vqf32_sum42 = neon::vmla(vqf32_sum42, vqf32_b1, ta2);
        vqf32_sum43 = neon::vmla(vqf32_sum43, vqf32_b1, ta3);
        vqf32_sum44 = neon::vmla(vqf32_sum44, vqf32_b1, ta4);
        vqf32_sum45 = neon::vmla(vqf32_sum45, vqf32_b1, ta5);
        vqf32_sum46 = neon::vmla(vqf32_sum46, vqf32_b1, ta6);
        vqf32_sum47 = neon::vmla(vqf32_sum47, vqf32_b1, ta7);

        vqf32_b0 = neon::vload1q(b + ldb * (p + 3));
        vqf32_b1 = neon::vload1q(b + ldb * (p + 3) + 4);

        ta0 = neon::vgetlane<3>(vqf32_a0);
        ta1 = neon::vgetlane<3>(vqf32_a1);
        ta2 = neon::vgetlane<3>(vqf32_a2);
        ta3 = neon::vgetlane<3>(vqf32_a3);
        ta4 = neon::vgetlane<3>(vqf32_a4);
        ta5 = neon::vgetlane<3>(vqf32_a5);
        ta6 = neon::vgetlane<3>(vqf32_a6);
        ta7 = neon::vgetlane<3>(vqf32_a7);

        vqf32_sum00 = neon::vmla(vqf32_sum00, vqf32_b0, ta0);
        vqf32_sum01 = neon::vmla(vqf32_sum01, vqf32_b0, ta1);
        vqf32_sum02 = neon::vmla(vqf32_sum02, vqf32_b0, ta2);
        vqf32_sum03 = neon::vmla(vqf32_sum03, vqf32_b0, ta3);
        vqf32_sum04 = neon::vmla(vqf32_sum04, vqf32_b0, ta4);
        vqf32_sum05 = neon::vmla(vqf32_sum05, vqf32_b0, ta5);
        vqf32_sum06 = neon::vmla(vqf32_sum06, vqf32_b0, ta6);
        vqf32_sum07 = neon::vmla(vqf32_sum07, vqf32_b0, ta7);

        vqf32_sum40 = neon::vmla(vqf32_sum40, vqf32_b1, ta0);
        vqf32_sum41 = neon::vmla(vqf32_sum41, vqf32_b1, ta1);
        vqf32_sum42 = neon::vmla(vqf32_sum42, vqf32_b1, ta2);
        vqf32_sum43 = neon::vmla(vqf32_sum43, vqf32_b1, ta3);
        vqf32_sum44 = neon::vmla(vqf32_sum44, vqf32_b1, ta4);
        vqf32_sum45 = neon::vmla(vqf32_sum45, vqf32_b1, ta5);
        vqf32_sum46 = neon::vmla(vqf32_sum46, vqf32_b1, ta6);
        vqf32_sum47 = neon::vmla(vqf32_sum47, vqf32_b1, ta7);
    }

    for (; p < k; p++)
    {
        vqf32_b0 = neon::vload1q(b + ldb * p);
        vqf32_b1 = neon::vload1q(b + ldb * p + 4);

        ta0 = *(ptr_a0 + p);
        ta1 = *(ptr_a1 + p);
        ta2 = *(ptr_a2 + p);
        ta3 = *(ptr_a3 + p);
        ta4 = *(ptr_a4 + p);
        ta5 = *(ptr_a5 + p);
        ta6 = *(ptr_a6 + p);
        ta7 = *(ptr_a7 + p);

        vqf32_sum00 = neon::vmla(vqf32_sum00, vqf32_b0, ta0);
        vqf32_sum01 = neon::vmla(vqf32_sum01, vqf32_b0, ta1);
        vqf32_sum02 = neon::vmla(vqf32_sum02, vqf32_b0, ta2);
        vqf32_sum03 = neon::vmla(vqf32_sum03, vqf32_b0, ta3);
        vqf32_sum04 = neon::vmla(vqf32_sum04, vqf32_b0, ta4);
        vqf32_sum05 = neon::vmla(vqf32_sum05, vqf32_b0, ta5);
        vqf32_sum06 = neon::vmla(vqf32_sum06, vqf32_b0, ta6);
        vqf32_sum07 = neon::vmla(vqf32_sum07, vqf32_b0, ta7);

        vqf32_sum40 = neon::vmla(vqf32_sum40, vqf32_b1, ta0);
        vqf32_sum41 = neon::vmla(vqf32_sum41, vqf32_b1, ta1);
        vqf32_sum42 = neon::vmla(vqf32_sum42, vqf32_b1, ta2);
        vqf32_sum43 = neon::vmla(vqf32_sum43, vqf32_b1, ta3);
        vqf32_sum44 = neon::vmla(vqf32_sum44, vqf32_b1, ta4);
        vqf32_sum45 = neon::vmla(vqf32_sum45, vqf32_b1, ta5);
        vqf32_sum46 = neon::vmla(vqf32_sum46, vqf32_b1, ta6);
        vqf32_sum47 = neon::vmla(vqf32_sum47, vqf32_b1, ta7);
    }

    neon::vstore(c + ldc * 0, vqf32_sum00);
    neon::vstore(c + ldc * 1, vqf32_sum01);
    neon::vstore(c + ldc * 2, vqf32_sum02);
    neon::vstore(c + ldc * 3, vqf32_sum03);
    neon::vstore(c + ldc * 4, vqf32_sum04);
    neon::vstore(c + ldc * 5, vqf32_sum05);
    neon::vstore(c + ldc * 6, vqf32_sum06);
    neon::vstore(c + ldc * 7, vqf32_sum07);

    neon::vstore(c + ldc * 0 + 4, vqf32_sum40);
    neon::vstore(c + ldc * 1 + 4, vqf32_sum41);
    neon::vstore(c + ldc * 2 + 4, vqf32_sum42);
    neon::vstore(c + ldc * 3 + 4, vqf32_sum43);
    neon::vstore(c + ldc * 4 + 4, vqf32_sum44);
    neon::vstore(c + ldc * 5 + 4, vqf32_sum45);
    neon::vstore(c + ldc * 6 + 4, vqf32_sum46);
    neon::vstore(c + ldc * 7 + 4, vqf32_sum47);
}

AURA_INLINE AURA_VOID AddDotNeon8x4(MI_S32 k, const MI_F32 *a, MI_S32 lda, const MI_F32 *b, MI_S32 ldb, MI_F32 *c, MI_S32 ldc)
{
    const MI_F32 *ptr_a0 = a + lda * 0;
    const MI_F32 *ptr_a1 = a + lda * 1;
    const MI_F32 *ptr_a2 = a + lda * 2;
    const MI_F32 *ptr_a3 = a + lda * 3;

    float32x4_t vqf32_sum00; neon::vdup(vqf32_sum00, 0);
    float32x4_t vqf32_sum01; neon::vdup(vqf32_sum01, 0);
    float32x4_t vqf32_sum02; neon::vdup(vqf32_sum02, 0);
    float32x4_t vqf32_sum03; neon::vdup(vqf32_sum03, 0);

    float32x4_t vqf32_sum40; neon::vdup(vqf32_sum40, 0);
    float32x4_t vqf32_sum41; neon::vdup(vqf32_sum41, 0);
    float32x4_t vqf32_sum42; neon::vdup(vqf32_sum42, 0);
    float32x4_t vqf32_sum43; neon::vdup(vqf32_sum43, 0);

    float32x4_t vqf32_a0;
    float32x4_t vqf32_a1;
    float32x4_t vqf32_a2;
    float32x4_t vqf32_a3;

    float32x4_t vqf32_b0;
    float32x4_t vqf32_b1;

    MI_F32 ta0 = 0;
    MI_F32 ta1 = 0;
    MI_F32 ta2 = 0;
    MI_F32 ta3 = 0;

    MI_S32 k_align4 = (k & (-4));

    MI_S32 p = 0;

    for (; p < k_align4; p += 4)
    {
        neon::vload(ptr_a0 + p, vqf32_a0);
        neon::vload(ptr_a1 + p, vqf32_a1);
        neon::vload(ptr_a2 + p, vqf32_a2);
        neon::vload(ptr_a3 + p, vqf32_a3);

        vqf32_b0 = neon::vload1q(b + ldb * (p + 0));
        vqf32_b1 = neon::vload1q(b + ldb * (p + 0) + 4);

        ta0 = neon::vgetlane<0>(vqf32_a0);
        ta1 = neon::vgetlane<0>(vqf32_a1);
        ta2 = neon::vgetlane<0>(vqf32_a2);
        ta3 = neon::vgetlane<0>(vqf32_a3);

        vqf32_sum00 = neon::vmla(vqf32_sum00, vqf32_b0, ta0);
        vqf32_sum01 = neon::vmla(vqf32_sum01, vqf32_b0, ta1);
        vqf32_sum02 = neon::vmla(vqf32_sum02, vqf32_b0, ta2);
        vqf32_sum03 = neon::vmla(vqf32_sum03, vqf32_b0, ta3);
        vqf32_sum40 = neon::vmla(vqf32_sum40, vqf32_b1, ta0);
        vqf32_sum41 = neon::vmla(vqf32_sum41, vqf32_b1, ta1);
        vqf32_sum42 = neon::vmla(vqf32_sum42, vqf32_b1, ta2);
        vqf32_sum43 = neon::vmla(vqf32_sum43, vqf32_b1, ta3);

        vqf32_b0 = neon::vload1q(b + ldb * (p + 1));
        vqf32_b1 = neon::vload1q(b + ldb * (p + 1) + 4);

        ta0 = neon::vgetlane<1>(vqf32_a0);
        ta1 = neon::vgetlane<1>(vqf32_a1);
        ta2 = neon::vgetlane<1>(vqf32_a2);
        ta3 = neon::vgetlane<1>(vqf32_a3);

        vqf32_sum00 = neon::vmla(vqf32_sum00, vqf32_b0, ta0);
        vqf32_sum01 = neon::vmla(vqf32_sum01, vqf32_b0, ta1);
        vqf32_sum02 = neon::vmla(vqf32_sum02, vqf32_b0, ta2);
        vqf32_sum03 = neon::vmla(vqf32_sum03, vqf32_b0, ta3);
        vqf32_sum40 = neon::vmla(vqf32_sum40, vqf32_b1, ta0);
        vqf32_sum41 = neon::vmla(vqf32_sum41, vqf32_b1, ta1);
        vqf32_sum42 = neon::vmla(vqf32_sum42, vqf32_b1, ta2);
        vqf32_sum43 = neon::vmla(vqf32_sum43, vqf32_b1, ta3);

        vqf32_b0 = neon::vload1q(b + ldb * (p + 2));
        vqf32_b1 = neon::vload1q(b + ldb * (p + 2) + 4);

        ta0 = neon::vgetlane<2>(vqf32_a0);
        ta1 = neon::vgetlane<2>(vqf32_a1);
        ta2 = neon::vgetlane<2>(vqf32_a2);
        ta3 = neon::vgetlane<2>(vqf32_a3);

        vqf32_sum00 = neon::vmla(vqf32_sum00, vqf32_b0, ta0);
        vqf32_sum01 = neon::vmla(vqf32_sum01, vqf32_b0, ta1);
        vqf32_sum02 = neon::vmla(vqf32_sum02, vqf32_b0, ta2);
        vqf32_sum03 = neon::vmla(vqf32_sum03, vqf32_b0, ta3);
        vqf32_sum40 = neon::vmla(vqf32_sum40, vqf32_b1, ta0);
        vqf32_sum41 = neon::vmla(vqf32_sum41, vqf32_b1, ta1);
        vqf32_sum42 = neon::vmla(vqf32_sum42, vqf32_b1, ta2);
        vqf32_sum43 = neon::vmla(vqf32_sum43, vqf32_b1, ta3);

        vqf32_b0 = neon::vload1q(b + ldb * (p + 3));
        vqf32_b1 = neon::vload1q(b + ldb * (p + 3) + 4);

        ta0 = neon::vgetlane<3>(vqf32_a0);
        ta1 = neon::vgetlane<3>(vqf32_a1);
        ta2 = neon::vgetlane<3>(vqf32_a2);
        ta3 = neon::vgetlane<3>(vqf32_a3);

        vqf32_sum00 = neon::vmla(vqf32_sum00, vqf32_b0, ta0);
        vqf32_sum01 = neon::vmla(vqf32_sum01, vqf32_b0, ta1);
        vqf32_sum02 = neon::vmla(vqf32_sum02, vqf32_b0, ta2);
        vqf32_sum03 = neon::vmla(vqf32_sum03, vqf32_b0, ta3);
        vqf32_sum40 = neon::vmla(vqf32_sum40, vqf32_b1, ta0);
        vqf32_sum41 = neon::vmla(vqf32_sum41, vqf32_b1, ta1);
        vqf32_sum42 = neon::vmla(vqf32_sum42, vqf32_b1, ta2);
        vqf32_sum43 = neon::vmla(vqf32_sum43, vqf32_b1, ta3);
    }

    for (; p < k; p++)
    {
        vqf32_b0 = neon::vload1q(b + ldb * p);
        vqf32_b1 = neon::vload1q(b + ldb * p + 4);

        ta0 = *(ptr_a0 + p);
        ta1 = *(ptr_a1 + p);
        ta2 = *(ptr_a2 + p);
        ta3 = *(ptr_a3 + p);

        vqf32_sum00 = neon::vmla(vqf32_sum00, vqf32_b0, ta0);
        vqf32_sum01 = neon::vmla(vqf32_sum01, vqf32_b0, ta1);
        vqf32_sum02 = neon::vmla(vqf32_sum02, vqf32_b0, ta2);
        vqf32_sum03 = neon::vmla(vqf32_sum03, vqf32_b0, ta3);

        vqf32_sum40 = neon::vmla(vqf32_sum40, vqf32_b1, ta0);
        vqf32_sum41 = neon::vmla(vqf32_sum41, vqf32_b1, ta1);
        vqf32_sum42 = neon::vmla(vqf32_sum42, vqf32_b1, ta2);
        vqf32_sum43 = neon::vmla(vqf32_sum43, vqf32_b1, ta3);
    }

    neon::vstore(c + ldc * 0, vqf32_sum00);
    neon::vstore(c + ldc * 1, vqf32_sum01);
    neon::vstore(c + ldc * 2, vqf32_sum02);
    neon::vstore(c + ldc * 3, vqf32_sum03);

    neon::vstore(c + ldc * 0 + 4, vqf32_sum40);
    neon::vstore(c + ldc * 1 + 4, vqf32_sum41);
    neon::vstore(c + ldc * 2 + 4, vqf32_sum42);
    neon::vstore(c + ldc * 3 + 4, vqf32_sum43);
}

AURA_INLINE AURA_VOID AddDotNeon4x8(MI_S32 k, const MI_F32 *a, MI_S32 lda, const MI_F32 *b, MI_S32 ldb, MI_F32 *c, MI_S32 ldc)
{
    const MI_F32 *ptr_a0 = a + lda * 0;
    const MI_F32 *ptr_a1 = a + lda * 1;
    const MI_F32 *ptr_a2 = a + lda * 2;
    const MI_F32 *ptr_a3 = a + lda * 3;
    const MI_F32 *ptr_a4 = a + lda * 4;
    const MI_F32 *ptr_a5 = a + lda * 5;
    const MI_F32 *ptr_a6 = a + lda * 6;
    const MI_F32 *ptr_a7 = a + lda * 7;

    float32x4_t vqf32_sum0; neon::vdup(vqf32_sum0, 0);
    float32x4_t vqf32_sum1; neon::vdup(vqf32_sum1, 0);
    float32x4_t vqf32_sum2; neon::vdup(vqf32_sum2, 0);
    float32x4_t vqf32_sum3; neon::vdup(vqf32_sum3, 0);
    float32x4_t vqf32_sum4; neon::vdup(vqf32_sum4, 0);
    float32x4_t vqf32_sum5; neon::vdup(vqf32_sum5, 0);
    float32x4_t vqf32_sum6; neon::vdup(vqf32_sum6, 0);
    float32x4_t vqf32_sum7; neon::vdup(vqf32_sum7, 0);

    float32x4_t vqf32_a0;
    float32x4_t vqf32_a1;
    float32x4_t vqf32_a2;
    float32x4_t vqf32_a3;
    float32x4_t vqf32_a4;
    float32x4_t vqf32_a5;
    float32x4_t vqf32_a6;
    float32x4_t vqf32_a7;

    float32x4_t vqf32_b;

    MI_F32 ta0 = 0;
    MI_F32 ta1 = 0;
    MI_F32 ta2 = 0;
    MI_F32 ta3 = 0;
    MI_F32 ta4 = 0;
    MI_F32 ta5 = 0;
    MI_F32 ta6 = 0;
    MI_F32 ta7 = 0;

    MI_S32 k_align4 = (k & (-4));

    MI_S32 p = 0;

    for (; p < k_align4; p += 4)
    {
        neon::vload(ptr_a0 + p, vqf32_a0);
        neon::vload(ptr_a1 + p, vqf32_a1);
        neon::vload(ptr_a2 + p, vqf32_a2);
        neon::vload(ptr_a3 + p, vqf32_a3);
        neon::vload(ptr_a4 + p, vqf32_a4);
        neon::vload(ptr_a5 + p, vqf32_a5);
        neon::vload(ptr_a6 + p, vqf32_a6);
        neon::vload(ptr_a7 + p, vqf32_a7);

        vqf32_b = neon::vload1q(b + ldb * (p + 0));

        ta0 = neon::vgetlane<0>(vqf32_a0);
        ta1 = neon::vgetlane<0>(vqf32_a1);
        ta2 = neon::vgetlane<0>(vqf32_a2);
        ta3 = neon::vgetlane<0>(vqf32_a3);
        ta4 = neon::vgetlane<0>(vqf32_a4);
        ta5 = neon::vgetlane<0>(vqf32_a5);
        ta6 = neon::vgetlane<0>(vqf32_a6);
        ta7 = neon::vgetlane<0>(vqf32_a7);

        vqf32_sum0 = neon::vmla(vqf32_sum0, vqf32_b, ta0);
        vqf32_sum1 = neon::vmla(vqf32_sum1, vqf32_b, ta1);
        vqf32_sum2 = neon::vmla(vqf32_sum2, vqf32_b, ta2);
        vqf32_sum3 = neon::vmla(vqf32_sum3, vqf32_b, ta3);
        vqf32_sum4 = neon::vmla(vqf32_sum4, vqf32_b, ta4);
        vqf32_sum5 = neon::vmla(vqf32_sum5, vqf32_b, ta5);
        vqf32_sum6 = neon::vmla(vqf32_sum6, vqf32_b, ta6);
        vqf32_sum7 = neon::vmla(vqf32_sum7, vqf32_b, ta7);

        vqf32_b = neon::vload1q(b + ldb * (p + 1));

        ta0 = neon::vgetlane<1>(vqf32_a0);
        ta1 = neon::vgetlane<1>(vqf32_a1);
        ta2 = neon::vgetlane<1>(vqf32_a2);
        ta3 = neon::vgetlane<1>(vqf32_a3);
        ta4 = neon::vgetlane<1>(vqf32_a4);
        ta5 = neon::vgetlane<1>(vqf32_a5);
        ta6 = neon::vgetlane<1>(vqf32_a6);
        ta7 = neon::vgetlane<1>(vqf32_a7);

        vqf32_sum0 = neon::vmla(vqf32_sum0, vqf32_b, ta0);
        vqf32_sum1 = neon::vmla(vqf32_sum1, vqf32_b, ta1);
        vqf32_sum2 = neon::vmla(vqf32_sum2, vqf32_b, ta2);
        vqf32_sum3 = neon::vmla(vqf32_sum3, vqf32_b, ta3);
        vqf32_sum4 = neon::vmla(vqf32_sum4, vqf32_b, ta4);
        vqf32_sum5 = neon::vmla(vqf32_sum5, vqf32_b, ta5);
        vqf32_sum6 = neon::vmla(vqf32_sum6, vqf32_b, ta6);
        vqf32_sum7 = neon::vmla(vqf32_sum7, vqf32_b, ta7);

        vqf32_b = neon::vload1q(b + ldb * (p + 2));

        ta0 = neon::vgetlane<2>(vqf32_a0);
        ta1 = neon::vgetlane<2>(vqf32_a1);
        ta2 = neon::vgetlane<2>(vqf32_a2);
        ta3 = neon::vgetlane<2>(vqf32_a3);
        ta4 = neon::vgetlane<2>(vqf32_a4);
        ta5 = neon::vgetlane<2>(vqf32_a5);
        ta6 = neon::vgetlane<2>(vqf32_a6);
        ta7 = neon::vgetlane<2>(vqf32_a7);

        vqf32_sum0 = neon::vmla(vqf32_sum0, vqf32_b, ta0);
        vqf32_sum1 = neon::vmla(vqf32_sum1, vqf32_b, ta1);
        vqf32_sum2 = neon::vmla(vqf32_sum2, vqf32_b, ta2);
        vqf32_sum3 = neon::vmla(vqf32_sum3, vqf32_b, ta3);
        vqf32_sum4 = neon::vmla(vqf32_sum4, vqf32_b, ta4);
        vqf32_sum5 = neon::vmla(vqf32_sum5, vqf32_b, ta5);
        vqf32_sum6 = neon::vmla(vqf32_sum6, vqf32_b, ta6);
        vqf32_sum7 = neon::vmla(vqf32_sum7, vqf32_b, ta7);

        vqf32_b = neon::vload1q(b + ldb * (p + 3));

        ta0 = neon::vgetlane<3>(vqf32_a0);
        ta1 = neon::vgetlane<3>(vqf32_a1);
        ta2 = neon::vgetlane<3>(vqf32_a2);
        ta3 = neon::vgetlane<3>(vqf32_a3);
        ta4 = neon::vgetlane<3>(vqf32_a4);
        ta5 = neon::vgetlane<3>(vqf32_a5);
        ta6 = neon::vgetlane<3>(vqf32_a6);
        ta7 = neon::vgetlane<3>(vqf32_a7);

        vqf32_sum0 = neon::vmla(vqf32_sum0, vqf32_b, ta0);
        vqf32_sum1 = neon::vmla(vqf32_sum1, vqf32_b, ta1);
        vqf32_sum2 = neon::vmla(vqf32_sum2, vqf32_b, ta2);
        vqf32_sum3 = neon::vmla(vqf32_sum3, vqf32_b, ta3);
        vqf32_sum4 = neon::vmla(vqf32_sum4, vqf32_b, ta4);
        vqf32_sum5 = neon::vmla(vqf32_sum5, vqf32_b, ta5);
        vqf32_sum6 = neon::vmla(vqf32_sum6, vqf32_b, ta6);
        vqf32_sum7 = neon::vmla(vqf32_sum7, vqf32_b, ta7);
    }

    for (; p < k; p++)
    {
        vqf32_b = neon::vload1q(b + ldb * p);

        ta0 = *(ptr_a0 + p);
        ta1 = *(ptr_a1 + p);
        ta2 = *(ptr_a2 + p);
        ta3 = *(ptr_a3 + p);
        ta4 = *(ptr_a4 + p);
        ta5 = *(ptr_a5 + p);
        ta6 = *(ptr_a6 + p);
        ta7 = *(ptr_a7 + p);

        vqf32_sum0 = neon::vmla(vqf32_sum0, vqf32_b, ta0);
        vqf32_sum1 = neon::vmla(vqf32_sum1, vqf32_b, ta1);
        vqf32_sum2 = neon::vmla(vqf32_sum2, vqf32_b, ta2);
        vqf32_sum3 = neon::vmla(vqf32_sum3, vqf32_b, ta3);
        vqf32_sum4 = neon::vmla(vqf32_sum4, vqf32_b, ta4);
        vqf32_sum5 = neon::vmla(vqf32_sum5, vqf32_b, ta5);
        vqf32_sum6 = neon::vmla(vqf32_sum6, vqf32_b, ta6);
        vqf32_sum7 = neon::vmla(vqf32_sum7, vqf32_b, ta7);
    }

    neon::vstore(c + ldc * 0, vqf32_sum0);
    neon::vstore(c + ldc * 1, vqf32_sum1);
    neon::vstore(c + ldc * 2, vqf32_sum2);
    neon::vstore(c + ldc * 3, vqf32_sum3);
    neon::vstore(c + ldc * 4, vqf32_sum4);
    neon::vstore(c + ldc * 5, vqf32_sum5);
    neon::vstore(c + ldc * 6, vqf32_sum6);
    neon::vstore(c + ldc * 7, vqf32_sum7);
}

AURA_INLINE AURA_VOID AddDotNeon4x4(MI_S32 k, const MI_F32 *a, MI_S32 lda, const MI_F32 *b, MI_S32 ldb, MI_F32 *c, MI_S32 ldc)
{
    const MI_F32 *ptr_a0 = a + lda * 0;
    const MI_F32 *ptr_a1 = a + lda * 1;
    const MI_F32 *ptr_a2 = a + lda * 2;
    const MI_F32 *ptr_a3 = a + lda * 3;

    float32x4_t vqf32_sum0; neon::vdup(vqf32_sum0, 0);
    float32x4_t vqf32_sum1; neon::vdup(vqf32_sum1, 0);
    float32x4_t vqf32_sum2; neon::vdup(vqf32_sum2, 0);
    float32x4_t vqf32_sum3; neon::vdup(vqf32_sum3, 0);

    float32x4_t vqf32_a0;
    float32x4_t vqf32_a1;
    float32x4_t vqf32_a2;
    float32x4_t vqf32_a3;

    float32x4_t vqf32_b;

    MI_F32 ta0 = 0;
    MI_F32 ta1 = 0;
    MI_F32 ta2 = 0;
    MI_F32 ta3 = 0;

    MI_S32 k_align4 = (k & (-4));

    MI_S32 p = 0;

    for (; p < k_align4; p += 4)
    {
        vqf32_a0 = neon::vload1q(ptr_a0 + p);
        vqf32_a1 = neon::vload1q(ptr_a1 + p);
        vqf32_a2 = neon::vload1q(ptr_a2 + p);
        vqf32_a3 = neon::vload1q(ptr_a3 + p);

        vqf32_b = neon::vload1q(b + ldb * (p + 0));

        ta0 = neon::vgetlane<0>(vqf32_a0);
        ta1 = neon::vgetlane<0>(vqf32_a1);
        ta2 = neon::vgetlane<0>(vqf32_a2);
        ta3 = neon::vgetlane<0>(vqf32_a3);

        vqf32_sum0 = neon::vmla(vqf32_sum0, vqf32_b, ta0);
        vqf32_sum1 = neon::vmla(vqf32_sum1, vqf32_b, ta1);
        vqf32_sum2 = neon::vmla(vqf32_sum2, vqf32_b, ta2);
        vqf32_sum3 = neon::vmla(vqf32_sum3, vqf32_b, ta3);

        vqf32_b = neon::vload1q(b + ldb * (p + 1));

        ta0 = neon::vgetlane<1>(vqf32_a0);
        ta1 = neon::vgetlane<1>(vqf32_a1);
        ta2 = neon::vgetlane<1>(vqf32_a2);
        ta3 = neon::vgetlane<1>(vqf32_a3);

        vqf32_sum0 = neon::vmla(vqf32_sum0, vqf32_b, ta0);
        vqf32_sum1 = neon::vmla(vqf32_sum1, vqf32_b, ta1);
        vqf32_sum2 = neon::vmla(vqf32_sum2, vqf32_b, ta2);
        vqf32_sum3 = neon::vmla(vqf32_sum3, vqf32_b, ta3);

        vqf32_b = neon::vload1q(b + ldb * (p + 2));

        ta0 = neon::vgetlane<2>(vqf32_a0);
        ta1 = neon::vgetlane<2>(vqf32_a1);
        ta2 = neon::vgetlane<2>(vqf32_a2);
        ta3 = neon::vgetlane<2>(vqf32_a3);

        vqf32_sum0 = neon::vmla(vqf32_sum0, vqf32_b, ta0);
        vqf32_sum1 = neon::vmla(vqf32_sum1, vqf32_b, ta1);
        vqf32_sum2 = neon::vmla(vqf32_sum2, vqf32_b, ta2);
        vqf32_sum3 = neon::vmla(vqf32_sum3, vqf32_b, ta3);

        vqf32_b = neon::vload1q(b + ldb * (p + 3));

        ta0 = neon::vgetlane<3>(vqf32_a0);
        ta1 = neon::vgetlane<3>(vqf32_a1);
        ta2 = neon::vgetlane<3>(vqf32_a2);
        ta3 = neon::vgetlane<3>(vqf32_a3);

        vqf32_sum0 = neon::vmla(vqf32_sum0, vqf32_b, ta0);
        vqf32_sum1 = neon::vmla(vqf32_sum1, vqf32_b, ta1);
        vqf32_sum2 = neon::vmla(vqf32_sum2, vqf32_b, ta2);
        vqf32_sum3 = neon::vmla(vqf32_sum3, vqf32_b, ta3);
    }

    for (; p < k; p++)
    {
        vqf32_b = neon::vload1q(b + ldb * p);

        ta0 = *(ptr_a0 + p);
        ta1 = *(ptr_a1 + p);
        ta2 = *(ptr_a2 + p);
        ta3 = *(ptr_a3 + p);

        vqf32_sum0 = neon::vmla(vqf32_sum0, vqf32_b, ta0);
        vqf32_sum1 = neon::vmla(vqf32_sum1, vqf32_b, ta1);
        vqf32_sum2 = neon::vmla(vqf32_sum2, vqf32_b, ta2);
        vqf32_sum3 = neon::vmla(vqf32_sum3, vqf32_b, ta3);
    }

    neon::vstore(c + ldc * 0, vqf32_sum0);
    neon::vstore(c + ldc * 1, vqf32_sum1);
    neon::vstore(c + ldc * 2, vqf32_sum2);
    neon::vstore(c + ldc * 3, vqf32_sum3);
}

static Status GemmNeonImpl(const Mat &src0, const Mat &src1, Mat &dst, MI_S32 start_row, MI_S32 end_row)
{
    // A(src0): k x m
    // B(src1): n x k
    // C(dst) : n x m
    const MI_S32 m = Min(src0.GetSizes().m_height, end_row);
    const MI_S32 n = src1.GetSizes().m_width;
    const MI_S32 k = src0.GetSizes().m_width;

    const MI_S32 stride_a = src0.GetRowPitch() / sizeof(MI_F32);
    const MI_S32 stride_b = src1.GetRowPitch() / sizeof(MI_F32);
    const MI_S32 stride_c = dst.GetRowPitch() / sizeof(MI_F32);

    MI_S32 x = 0;
    MI_S32 y = Min(0, start_row);

    MI_S32 m_align4 = (m & (-4));
    MI_S32 m_align8 = (m & (-8));
    MI_S32 n_align4 = (n & (-4));
    MI_S32 n_align8 = (n & (-8));

    const MI_F32 *data_a = (MI_F32 *)src0.GetData();
    const MI_F32 *data_b = (MI_F32 *)src1.GetData();
    MI_F32       *data_c = (MI_F32 *)dst.GetData();

    for (; y < m_align8; y += 8)
    {
        const MI_F32 *line_a = data_a + stride_a * y;
        MI_F32       *line_c = data_c + stride_c * y;

        x = 0;
        for (; x < n_align8; x += 8)
        {
            AddDotNeon8x8(k, line_a, stride_a, data_b + x, stride_b, line_c + x, stride_c);
        }
        for (; x < n_align4; x += 4)
        {
            AddDotNeon4x8(k, line_a, stride_a, data_b + x, stride_b, line_c + x, stride_c);
        }

        if (x < n)
        {
            x = n - 4;
            AddDotNeon4x8(k, line_a, stride_a, data_b + x, stride_b, line_c + x, stride_c);
        }
    }

    for (; y < m_align4; y += 4)
    {
        const MI_F32 *line_a = data_a + stride_a * y;
        MI_F32       *line_c = data_c + stride_c * y;

        x = 0;
        for (; x < n_align8; x += 8)
        {
            AddDotNeon8x4(k, line_a, stride_a, data_b + x, stride_b, line_c + x, stride_c);
        }
        for (; x < n_align4; x += 4)
        {
            AddDotNeon4x4(k, line_a, stride_a, data_b + x, stride_b, line_c + x, stride_c);
        }

        if (x < n)
        {
            x = n - 4;
            AddDotNeon4x4(k, line_a, stride_a, data_b + x, stride_b, line_c + x, stride_c);
        }
    }

    if (y < m)
    {
        y = m - 4;

        const MI_F32 *line_a = data_a + stride_a * y;
        MI_F32       *line_c = data_c + stride_c * y;

        x = 0;
        for (; x < n_align8; x += 8)
        {
            AddDotNeon8x4(k, line_a, stride_a, data_b + x, stride_b, line_c + x, stride_c);
        }
        for (; x < n_align4; x += 4)
        {
            AddDotNeon4x4(k, line_a, stride_a, data_b + x, stride_b, line_c + x, stride_c);
        }

        if (x < n)
        {
            x = n - 4;
            AddDotNeon4x4(k, line_a, stride_a, data_b + x, stride_b, line_c + x, stride_c);
        }
    }

    return Status::OK;
}

GemmNeon::GemmNeon(Context *ctx, const OpTarget &target) : GemmImpl(ctx, target)
{}

Status GemmNeon::SetArgs(const Array *src0, const Array *src1, Array *dst)
{
    if (GemmImpl::SetArgs(src0, src1, dst) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GemmImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src0->GetArrayType() != ArrayType::MAT) || (src1->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status GemmNeon::Run()
{
    const Mat *src0 = dynamic_cast<const Mat*>(m_src0);
    const Mat *src1 = dynamic_cast<const Mat*>(m_src1);
    Mat *dst        = dynamic_cast<Mat*>(m_dst);

    if ((NULL == src0) || (NULL == src1) || (NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is null");
        return Status::ERROR;
    }

    Status ret = GemmNeonImpl(*src0, *src1, *dst, 0, dst->GetSizes().m_height);
    AURA_RETURN(m_ctx, ret);
}

} // namespace aura