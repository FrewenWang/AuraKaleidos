#include "pyrdown_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

template <typename d16x4x2_t, typename d32x4x2_t, typename Kt,
          typename d32x4_t = typename neon::QVector<Kt>::VType>
AURA_ALWAYS_INLINE AURA_VOID PyrDown5x5VCore(d16x4x2_t &v2d16_src_p1, d16x4x2_t &v2d16_src_p0, d16x4x2_t &v2d16_src_c,
                                           d16x4x2_t &v2d16_src_n0, d16x4x2_t &v2d16_src_n1, d32x4x2_t &v2q32_result,
                                           const Kt *kernel)
{
    Kt k0 = kernel[0], k1 = kernel[1], k2 = kernel[2];

    d32x4_t vq32_esum_p0n0 = neon::vaddl(v2d16_src_p0.val[0], v2d16_src_n0.val[0]);
    d32x4_t vq32_esum_p1n1 = neon::vaddl(v2d16_src_p1.val[0], v2d16_src_n1.val[0]);
    vq32_esum_p0n0         = neon::vmul(vq32_esum_p0n0, k1);
    vq32_esum_p1n1         = neon::vmul(vq32_esum_p1n1, k0);
    v2q32_result.val[0]    = neon::vadd(vq32_esum_p0n0, vq32_esum_p1n1);
    v2q32_result.val[0]    = neon::vmla(v2q32_result.val[0], neon::vmovl(v2d16_src_c.val[0]), k2);

    d32x4_t vq32_osum_p0n0 = neon::vaddl(v2d16_src_p0.val[1], v2d16_src_n0.val[1]);
    d32x4_t vq32_osum_p1n1 = neon::vaddl(v2d16_src_p1.val[1], v2d16_src_n1.val[1]);
    vq32_osum_p0n0         = neon::vmul(vq32_osum_p0n0, k1);
    vq32_osum_p1n1         = neon::vmul(vq32_osum_p1n1, k0);
    v2q32_result.val[1]    = neon::vadd(vq32_osum_p0n0, vq32_osum_p1n1);
    v2q32_result.val[1]    = neon::vmla(v2q32_result.val[1], neon::vmovl(v2d16_src_c.val[1]), k2);
}

template <typename d32x4x2_t, typename d16x4_t, typename Kt>
AURA_ALWAYS_INLINE AURA_VOID PyrDown5x5HCore(d32x4x2_t &v2q32_sum_x0, d32x4x2_t &v2q32_sum_x1, d32x4x2_t &v2q32_sum_x2,
                                           d16x4_t &vd16_result, const Kt *kernel)
{
    using d32x2_t = typename neon::DVector<Kt>::VType;
    using d32x4_t = typename neon::QVector<Kt>::VType;
    using d64x2_t = typename neon::QVector<typename Promote<Kt>::Type>::VType;

    Kt k0 = kernel[0], k1 = kernel[1], k2 = kernel[2];

    d32x4_t vq32_sum_l1   = neon::vext<3>(v2q32_sum_x0.val[0], v2q32_sum_x1.val[0]);
    d32x4_t vq32_sum_l0   = neon::vext<3>(v2q32_sum_x0.val[1], v2q32_sum_x1.val[1]);
    d32x4_t vq32_sum_r1   = neon::vext<1>(v2q32_sum_x1.val[0], v2q32_sum_x2.val[0]);

    d32x4_t vq32_sum_c    = neon::vadd(vq32_sum_l0, v2q32_sum_x1.val[1]);
    d32x4_t vq32_sum_l1r1 = neon::vadd(vq32_sum_l1, vq32_sum_r1);
    d64x2_t vq64_mul_c_lo = neon::vmull(neon::vgetlow(v2q32_sum_x1.val[0]), k2);
    d64x2_t vq64_mul_c_hi = neon::vmull(neon::vgethigh(v2q32_sum_x1.val[0]), k2);
    vq64_mul_c_lo         = neon::vmlal(vq64_mul_c_lo, neon::vgetlow(vq32_sum_c), k1);
    vq64_mul_c_hi         = neon::vmlal(vq64_mul_c_hi, neon::vgethigh(vq32_sum_c), k1);
    vq64_mul_c_lo         = neon::vmlal(vq64_mul_c_lo, neon::vgetlow(vq32_sum_l1r1), k0);
    vq64_mul_c_hi         = neon::vmlal(vq64_mul_c_hi, neon::vgethigh(vq32_sum_l1r1), k0);

    d32x2_t vd32_sum_lo   = neon::vqrshrn_n<20>(vq64_mul_c_lo);
    d32x2_t vd32_sum_hi   = neon::vqrshrn_n<20>(vq64_mul_c_hi);
    vd16_result           = neon::vqrshrn_n<8>(neon::vcombine(vd32_sum_lo, vd32_sum_hi));

    v2q32_sum_x0 = v2q32_sum_x1;
    v2q32_sum_x1 = v2q32_sum_x2;
}

AURA_ALWAYS_INLINE AURA_VOID PyrDown5x5VCore(uint8x8x2_t &v2du8_src_p1, uint8x8x2_t &v2du8_src_p0, uint8x8x2_t &v2du8_src_c,
                                           uint8x8x2_t &v2du8_src_n0, uint8x8x2_t &v2du8_src_n1, uint16x8x2_t &v2qu16_result, const MI_U16 *kernel)
{
    MI_U16 k0 = kernel[0], k1 = kernel[1], k2 = kernel[2];

    uint16x8_t vqu16_esum_p1n1 = neon::vaddl(v2du8_src_p1.val[0], v2du8_src_n1.val[0]);
    uint16x8_t vqu16_esum_p0n0 = neon::vaddl(v2du8_src_p0.val[0], v2du8_src_n0.val[0]);
    vqu16_esum_p1n1            = neon::vmul(vqu16_esum_p1n1, k0);
    vqu16_esum_p0n0            = neon::vmul(vqu16_esum_p0n0, k1);
    v2qu16_result.val[0]       = neon::vadd(vqu16_esum_p1n1, vqu16_esum_p0n0);
    v2qu16_result.val[0]       = neon::vmla(v2qu16_result.val[0], neon::vmovl(v2du8_src_c.val[0]), k2);

    uint16x8_t vqu16_osum_p1n1 = neon::vaddl(v2du8_src_p1.val[1], v2du8_src_n1.val[1]);
    uint16x8_t vqu16_osum_p0n0 = neon::vaddl(v2du8_src_p0.val[1], v2du8_src_n0.val[1]);
    vqu16_osum_p1n1            = neon::vmul(vqu16_osum_p1n1, k0);
    vqu16_osum_p0n0            = neon::vmul(vqu16_osum_p0n0, k1);
    v2qu16_result.val[1]       = neon::vadd(vqu16_osum_p1n1, vqu16_osum_p0n0);
    v2qu16_result.val[1]       = neon::vmla(v2qu16_result.val[1], neon::vmovl(v2du8_src_c.val[1]), k2);
}

AURA_ALWAYS_INLINE AURA_VOID PyrDown5x5HCore(uint16x8x2_t &v2qu16_sum_x0, uint16x8x2_t &v2qu16_sum_x1, uint16x8x2_t &v2qu16_sum_x2,
                                           uint8x8_t &vdu8_result, const MI_U16 *kernel)
{
    MI_U16 k0 = kernel[0], k1 = kernel[1], k2 = kernel[2];

    uint16x8_t vqu16_sum_l1      = neon::vext<7>(v2qu16_sum_x0.val[0], v2qu16_sum_x1.val[0]);
    uint16x8_t vqu16_sum_l0      = neon::vext<7>(v2qu16_sum_x0.val[1], v2qu16_sum_x1.val[1]);
    uint16x8_t vqu16_sum_r1      = neon::vext<1>(v2qu16_sum_x1.val[0], v2qu16_sum_x2.val[0]);

    uint32x4_t vqu32_sum_lo      = neon::vmull(neon::vgetlow(v2qu16_sum_x1.val[0]), k2);
    uint32x4_t vqu32_sum_hi      = neon::vmull(neon::vgethigh(v2qu16_sum_x1.val[0]), k2);
    uint32x4_t vqu32_sum_c_lo    = neon::vaddl(neon::vgetlow(vqu16_sum_l0), neon::vgetlow(v2qu16_sum_x1.val[1]));
    uint32x4_t vqu32_sum_c_hi    = neon::vaddl(neon::vgethigh(vqu16_sum_l0), neon::vgethigh(v2qu16_sum_x1.val[1]));
    uint32x4_t vqu32_result_lo   = neon::vmla(vqu32_sum_lo, vqu32_sum_c_lo, static_cast<MI_U32>(k1));
    uint32x4_t vqu32_result_hi   = neon::vmla(vqu32_sum_hi, vqu32_sum_c_hi, static_cast<MI_U32>(k1));

    uint32x4_t vqu32_sum_l1r1_lo = neon::vaddl(neon::vgetlow(vqu16_sum_l1), neon::vgetlow(vqu16_sum_r1));
    uint32x4_t vqu32_sum_l1r1_hi = neon::vaddl(neon::vgethigh(vqu16_sum_l1), neon::vgethigh(vqu16_sum_r1));
    vqu32_result_lo              = neon::vmla(vqu32_result_lo, vqu32_sum_l1r1_lo, static_cast<MI_U32>(k0));
    vqu32_result_hi              = neon::vmla(vqu32_result_hi, vqu32_sum_l1r1_hi, static_cast<MI_U32>(k0));

    uint16x4_t vdu16_result_lo   = neon::vqrshrn_n<8>(vqu32_result_lo);
    uint16x4_t vdu16_result_hi   = neon::vqrshrn_n<8>(vqu32_result_hi);
    vdu8_result                  = neon::vqrshrn_n<8>(neon::vcombine(vdu16_result_lo, vdu16_result_hi));

    v2qu16_sum_x0 = v2qu16_sum_x1;
    v2qu16_sum_x1 = v2qu16_sum_x2;
}

template <typename Tp, BorderType BORDER_TYPE, typename Kt>
static AURA_VOID PyrDown5x5Row(const Tp *src_p1, const Tp *src_p0, const Tp *src_c, const Tp *src_n0, const Tp *src_n1,
                             MI_S32 iwidth, Tp *dst, MI_S32 owidth, const Kt *kernel)
{
    using MV2dSt      = typename neon::MDVector<Tp, 2>::MVType; // 8x8x2 or 16x4x2
    using VdDt        = typename neon::DVector<Tp>::VType; // 8x8_t or 16x4_t
    using MV2qSumType = typename neon::MQVector<typename Promote<Tp>::Type, 2>::MVType; // 16x8x2_t or 32x4x2_t
    using PromoteType = typename Promote<Kt>::Type;

    constexpr MI_S32 ELEM_COUNTS = static_cast<MI_S32>(sizeof(MV2dSt) / 2 / sizeof(Tp)); // dst
    constexpr MI_S32 VOFFSET     = ELEM_COUNTS << 1;

    const MI_S32 width_align = (owidth & -ELEM_COUNTS) << 1; // src_width
    const Tp border_value    = 0;

    MV2dSt mv_src_p1[3], mv_src_p0[3], mv_src_c[3], mv_src_n0[3], mv_src_n1[3];
    MV2qSumType mv_sum[3];
    VdDt v_result;

    // left
    {
        neon::vload(src_p1,           mv_src_p1[1]);
        neon::vload(src_p1 + VOFFSET, mv_src_p1[2]);
        neon::vload(src_p0,           mv_src_p0[1]);
        neon::vload(src_p0 + VOFFSET, mv_src_p0[2]);
        neon::vload(src_c,            mv_src_c[1]);
        neon::vload(src_c  + VOFFSET, mv_src_c[2]);
        neon::vload(src_n0,           mv_src_n0[1]);
        neon::vload(src_n0 + VOFFSET, mv_src_n0[2]);
        neon::vload(src_n1,           mv_src_n1[1]);
        neon::vload(src_n1 + VOFFSET, mv_src_n1[2]);

        // the left of even channel
        mv_src_p1[0].val[0] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mv_src_p1[1].val[0], src_p1[0], border_value);
        mv_src_p0[0].val[0] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mv_src_p0[1].val[0], src_p0[0], border_value);
        mv_src_c[0].val[0]  = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mv_src_c[1].val[0],  src_c[0],  border_value);
        mv_src_n0[0].val[0] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mv_src_n0[1].val[0], src_n0[0], border_value);
        mv_src_n1[0].val[0] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mv_src_n1[1].val[0], src_n1[0], border_value);

        // the left of odd channel -->  3 1 | 1 1 3 5
        mv_src_p1[0].val[1] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(neon::vext<ELEM_COUNTS - 1>(mv_src_p1[1].val[1], mv_src_p1[1].val[1]), src_p1[0], border_value);
        mv_src_p0[0].val[1] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(neon::vext<ELEM_COUNTS - 1>(mv_src_p0[1].val[1], mv_src_p0[1].val[1]), src_p0[0], border_value);
        mv_src_c[0].val[1]  = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(neon::vext<ELEM_COUNTS - 1>(mv_src_c[1].val[1],  mv_src_c[1].val[1]),  src_c[0],  border_value);
        mv_src_n0[0].val[1] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(neon::vext<ELEM_COUNTS - 1>(mv_src_n0[1].val[1], mv_src_n0[1].val[1]), src_n0[0], border_value);
        mv_src_n1[0].val[1] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(neon::vext<ELEM_COUNTS - 1>(mv_src_n1[1].val[1], mv_src_n1[1].val[1]), src_n1[0], border_value);

        PyrDown5x5VCore(mv_src_p1[0], mv_src_p0[0], mv_src_c[0],
                        mv_src_n0[0], mv_src_n1[0], mv_sum[0], kernel);
        PyrDown5x5VCore(mv_src_p1[1], mv_src_p0[1], mv_src_c[1],
                        mv_src_n0[1], mv_src_n1[1], mv_sum[1], kernel);
        PyrDown5x5VCore(mv_src_p1[2], mv_src_p0[2], mv_src_c[2],
                        mv_src_n0[2], mv_src_n1[2], mv_sum[2], kernel);

        PyrDown5x5HCore(mv_sum[0], mv_sum[1], mv_sum[2], v_result, kernel);

        neon::vstore(dst, v_result);
    }

    // middle
    for (MI_S32 x = VOFFSET; x < (width_align - VOFFSET); x += VOFFSET)
    {
        MI_S32 dx = x >> 1;
        neon::vload(src_p1 + x + VOFFSET, mv_src_p1[2]);
        neon::vload(src_p0 + x + VOFFSET, mv_src_p0[2]);
        neon::vload(src_c  + x + VOFFSET, mv_src_c[2]);
        neon::vload(src_n0 + x + VOFFSET, mv_src_n0[2]);
        neon::vload(src_n1 + x + VOFFSET, mv_src_n1[2]);

        PyrDown5x5VCore(mv_src_p1[2], mv_src_p0[2], mv_src_c[2],
                        mv_src_n0[2], mv_src_n1[2], mv_sum[2], kernel);
        PyrDown5x5HCore(mv_sum[0], mv_sum[1], mv_sum[2], v_result, kernel);

        neon::vstore(dst + dx, v_result);
    }

    // back
    {
        if (width_align != iwidth)
        {
            MI_S32 x = iwidth - (VOFFSET << 1) - (iwidth & 1);
            MI_S32 dx = x >> 1;

            neon::vload(src_p1 + x - VOFFSET, mv_src_p1[0]);
            neon::vload(src_p1 + x,           mv_src_p1[1]);
            neon::vload(src_p1 + x + VOFFSET, mv_src_p1[2]);
            neon::vload(src_p0 + x - VOFFSET, mv_src_p0[0]);
            neon::vload(src_p0 + x,           mv_src_p0[1]);
            neon::vload(src_p0 + x + VOFFSET, mv_src_p0[2]);
            neon::vload(src_c  + x - VOFFSET, mv_src_c[0]);
            neon::vload(src_c  + x,           mv_src_c[1]);
            neon::vload(src_c  + x + VOFFSET, mv_src_c[2]);
            neon::vload(src_n0 + x - VOFFSET, mv_src_n0[0]);
            neon::vload(src_n0 + x,           mv_src_n0[1]);
            neon::vload(src_n0 + x + VOFFSET, mv_src_n0[2]);
            neon::vload(src_n1 + x - VOFFSET, mv_src_n1[0]);
            neon::vload(src_n1 + x,           mv_src_n1[1]);
            neon::vload(src_n1 + x + VOFFSET, mv_src_n1[2]);

            PyrDown5x5VCore(mv_src_p1[0], mv_src_p0[0], mv_src_c[0],
                            mv_src_n0[0], mv_src_n1[0], mv_sum[0], kernel);
            PyrDown5x5VCore(mv_src_p1[1], mv_src_p0[1], mv_src_c[1],
                            mv_src_n0[1], mv_src_n1[1], mv_sum[1], kernel);
            PyrDown5x5VCore(mv_src_p1[2], mv_src_p0[2], mv_src_c[2],
                            mv_src_n0[2], mv_src_n1[2], mv_sum[2], kernel);
            PyrDown5x5HCore(mv_sum[0], mv_sum[1], mv_sum[2], v_result, kernel);

            neon::vstore(dst + dx, v_result);
        }
    }

    // right -- src_width is even
    {
        if (!(iwidth & 1))
        {
            MI_S32 x    = iwidth - VOFFSET;
            MI_S32 dx   = x >> 1;
            MI_S32 last = iwidth - 1;

            // the right of even channel
            mv_src_p1[2].val[0] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(neon::vext<1>(mv_src_p1[2].val[0], mv_src_p1[2].val[0]), src_p1[last], border_value);
            mv_src_p0[2].val[0] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(neon::vext<1>(mv_src_p0[2].val[0], mv_src_p0[2].val[0]), src_p0[last], border_value);
            mv_src_c[2].val[0]  = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(neon::vext<1>(mv_src_c[2].val[0],  mv_src_c[2].val[0]),  src_c[last],  border_value);
            mv_src_n0[2].val[0] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(neon::vext<1>(mv_src_n0[2].val[0], mv_src_n0[2].val[0]), src_n0[last], border_value);
            mv_src_n1[2].val[0] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(neon::vext<1>(mv_src_n1[2].val[0], mv_src_n1[2].val[0]), src_n1[last], border_value);

            PyrDown5x5VCore(mv_src_p1[2], mv_src_p0[2], mv_src_c[2],
                            mv_src_n0[2], mv_src_n1[2], mv_sum[2], kernel);
            PyrDown5x5HCore(mv_sum[0], mv_sum[1], mv_sum[2], v_result, kernel);

            neon::vstore(dst + dx, v_result);
        }
        // right -- src_widht is odd
        else
        {
            // the last vector
            MI_S32 x    = iwidth - VOFFSET - 1;
            MI_S32 dx   = x >> 1;
            MI_S32 last = iwidth - 1;

            // the right of even channel
            mv_src_p1[2].val[0] = GetBorderVector<BorderType::REPLICATE, BorderArea::RIGHT>(mv_src_p1[2].val[0], src_p1[last], border_value);
            mv_src_p0[2].val[0] = GetBorderVector<BorderType::REPLICATE, BorderArea::RIGHT>(mv_src_p0[2].val[0], src_p0[last], border_value);
            mv_src_c[2].val[0]  = GetBorderVector<BorderType::REPLICATE, BorderArea::RIGHT>(mv_src_c[2].val[0],  src_c[last],  border_value);
            mv_src_n0[2].val[0] = GetBorderVector<BorderType::REPLICATE, BorderArea::RIGHT>(mv_src_n0[2].val[0], src_n0[last], border_value);
            mv_src_n1[2].val[0] = GetBorderVector<BorderType::REPLICATE, BorderArea::RIGHT>(mv_src_n1[2].val[0], src_n1[last], border_value);

            PyrDown5x5VCore(mv_src_p1[2], mv_src_p0[2], mv_src_c[2],
                            mv_src_n0[2], mv_src_n1[2], mv_sum[2], kernel);
            PyrDown5x5HCore(mv_sum[0], mv_sum[1], mv_sum[2], v_result, kernel);

            neon::vstore(dst + dx, v_result);

            // the last pixel
            PromoteType sum_row[5] = {0};
            MI_S32 src_l1_idx = iwidth - 3;
            MI_S32 src_l0_idx = iwidth - 2;
            MI_S32 src_c_idx  = iwidth - 1;
            MI_S32 src_r0_idx = GetBorderIdx<BORDER_TYPE>(iwidth, iwidth);
            MI_S32 src_r1_idx = GetBorderIdx<BORDER_TYPE>(iwidth + 1, iwidth);
            sum_row[0] = (src_p1[src_l1_idx] + src_n1[src_l1_idx]) * kernel[0] +
                         (src_p0[src_l1_idx] + src_n0[src_l1_idx]) * kernel[1] + src_c[src_l1_idx] * kernel[2];
            sum_row[1] = (src_p1[src_l0_idx] + src_n1[src_l0_idx]) * kernel[0] +
                         (src_p0[src_l0_idx] + src_n0[src_l0_idx]) * kernel[1] + src_c[src_l0_idx] * kernel[2];
            sum_row[2] = (src_p1[src_c_idx]  + src_n1[src_c_idx])  * kernel[0] +
                         (src_p0[src_c_idx]  + src_n0[src_c_idx])  * kernel[1] + src_c[src_c_idx]  * kernel[2];
            sum_row[3] = (src_p1[src_r0_idx] + src_n1[src_r0_idx]) * kernel[0] +
                         (src_p0[src_r0_idx] + src_n0[src_r0_idx]) * kernel[1] + src_c[src_r0_idx] * kernel[2];
            sum_row[4] = (src_p1[src_r1_idx] + src_n1[src_r1_idx]) * kernel[0] +
                         (src_p0[src_r1_idx] + src_n0[src_r1_idx]) * kernel[1] + src_c[src_r1_idx] * kernel[2];

            PromoteType sum = (sum_row[0] + sum_row[4]) * kernel[0] + (sum_row[1] + sum_row[3]) * kernel[1] + sum_row[2] * kernel[2];

            dst[(iwidth - 1) >> 1] = ShiftSatCast<PromoteType, Tp, PyrDownTraits<Tp>::Q << 1>(sum);
        }
    }
}

template <typename Tp, BorderType BORDER_TYPE>
static Status PyrDown5x5NeonImpl(Context *ctx, const Mat &src, Mat &dst, const Mat &kmat,
                                 MI_S32 start_row, MI_S32 end_row)
{
    AURA_UNUSED(ctx);

    using Kt = typename PyrDownTraits<Tp>::KernelType;

    MI_S32 iwidth = src.GetSizes().m_width;
    MI_S32 owidth = dst.GetSizes().m_width;

    MI_S32 sy = start_row << 1;
    const Kt *kernel = kmat.Ptr<Kt>(0);

    const Tp *src_p1 = src.Ptr<Tp, BORDER_TYPE>(sy - 2, MI_NULL);
    const Tp *src_p0 = src.Ptr<Tp, BORDER_TYPE>(sy - 1, MI_NULL);
    const Tp *src_c  = src.Ptr<Tp>(sy);
    const Tp *src_n0 = src.Ptr<Tp, BORDER_TYPE>(sy + 1, MI_NULL);
    const Tp *src_n1 = src.Ptr<Tp, BORDER_TYPE>(sy + 2, MI_NULL);

    // cal row
    for (MI_S32 dy = start_row; dy < end_row; dy++)
    {
        sy = dy << 1;

        Tp *dst_row  = dst.Ptr<Tp>(dy);

        PyrDown5x5Row<Tp, BORDER_TYPE, Kt>(src_p1, src_p0, src_c, src_n0, src_n1, iwidth, dst_row, owidth, kernel);

        src_p1 = src_c;
        src_p0 = src_n0;
        src_c  = src_n1;
        src_n0 = src.Ptr<Tp, BORDER_TYPE>(sy + 3, MI_NULL);
        src_n1 = src.Ptr<Tp, BORDER_TYPE>(sy + 4, MI_NULL);
    }

    return Status::OK;
}

template <typename Tp>
static Status PyrDown5x5NeonHelper(Context *ctx, const Mat &src, Mat &dst, const Mat &kmat,
                                    BorderType &border_type, const OpTarget &target)
{
    AURA_UNUSED(target);
    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
        return Status::ERROR;
    }

    MI_S32 height = dst.GetSizes().m_height;

    Status ret = Status::ERROR;

    switch (border_type)
    {
        case BorderType::REPLICATE:
        {
            ret = wp->ParallelFor(0, height, PyrDown5x5NeonImpl<Tp, BorderType::REPLICATE>,
                                  ctx, std::cref(src), std::ref(dst), std::cref(kmat));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "PyrDown5x5NeonImpl<BorderType::REPLICATE, Tp> failed");
            }
            break;
        }

        case BorderType::REFLECT_101:
        {
            ret = wp->ParallelFor(0, height, PyrDown5x5NeonImpl<Tp, BorderType::REFLECT_101>,
                                  ctx, std::cref(src), std::ref(dst), std::cref(kmat));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "PyrDown5x5NeonImpl<BorderType::REFLECT_101, Tp> failed");
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "Unsupport border_type");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

Status PyrDown5x5Neon(Context *ctx, const Mat &src, Mat &dst, const Mat &kmat, BorderType &border_type, const OpTarget &target)
{
    Status ret = Status::ERROR;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = PyrDown5x5NeonHelper<MI_U8>(ctx, src, dst, kmat, border_type, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "PyrDown5x5NeonHelper<MI_U8> failed");
            }
            break;
        }

        case ElemType::U16:
        {
            ret = PyrDown5x5NeonHelper<MI_U16>(ctx, src, dst, kmat, border_type, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "PyrDown5x5NeonHelper<MI_U16> failed");
            }
            break;
        }

        case ElemType::S16:
        {
            ret = PyrDown5x5NeonHelper<MI_S16>(ctx, src, dst, kmat, border_type, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "PyrDown5x5NeonHelper<MI_S16> failed");
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "Unsupported source format");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura