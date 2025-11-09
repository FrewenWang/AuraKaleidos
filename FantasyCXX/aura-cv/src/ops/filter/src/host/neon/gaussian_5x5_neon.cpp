#include "gaussian_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

// using d16x4_t = uint16x4_t , int16x4_t
template <typename d16x4_t, typename Kt, typename d32x4_t = typename neon::WVectorBits<d16x4_t>::VType,
          typename std::enable_if<(std::is_same<d16x4_t, uint16x4_t>::value || std::is_same<d16x4_t, int16x4_t>::value)>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Gaussian5x5VCore(d16x4_t &vd16_src_p1, d16x4_t &vd16_src_p0, d16x4_t &vd16_src_c, d16x4_t &vd16_src_n0,
                                            d16x4_t &vd16_src_n1, d32x4_t &vq32_result, const Kt *kernel)
{
    Kt k0 = kernel[0], k1 = kernel[1], k2 = kernel[2];

    d32x4_t vq32_sum_p0n0 = neon::vaddl(vd16_src_p0, vd16_src_n0);
    d32x4_t vq32_sum_p1n1 = neon::vaddl(vd16_src_p1, vd16_src_n1);
    vq32_result           = neon::vmul(neon::vmovl(vd16_src_c), k2);
    vq32_result           = neon::vmla(vq32_result, vq32_sum_p0n0, k1);
    vq32_result           = neon::vmla(vq32_result, vq32_sum_p1n1, k0);
}

// using d16x4_t = uint16x4_t , int16x4_t
template <typename d16x4_t, typename Kt, typename d32x4_t = typename neon::WVectorBits<d16x4_t>::VType,
          typename std::enable_if<(std::is_same<d16x4_t, uint16x4_t>::value || std::is_same<d16x4_t, int16x4_t>::value)>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Gaussian5x5HCore(d32x4_t &vq32_sum_x0, d32x4_t &vq32_sum_x1, d32x4_t &vq32_sum_x2,
                                            d16x4_t &vd16_result, const Kt *kernel)
{
    using d32x2_t = typename neon::DVector<Kt>::VType;
    using d64x2_t = typename neon::WVectorBits<d32x2_t>::VType;

    Kt k0 = kernel[0], k1 = kernel[1], k2 = kernel[2];

    d32x4_t vq32_sum_l1   = neon::vext<2>(vq32_sum_x0, vq32_sum_x1);
    d32x4_t vq32_sum_r0   = neon::vext<1>(vq32_sum_x1, vq32_sum_x2);
    d32x4_t vq32_sum_l0   = neon::vext<3>(vq32_sum_x0, vq32_sum_x1);
    d32x4_t vq32_sum_r1   = neon::vext<2>(vq32_sum_x1, vq32_sum_x2);
    d32x4_t vq32_sum_l0r0 = neon::vadd(vq32_sum_l0, vq32_sum_r0);
    d32x4_t vq32_sum_l1r1 = neon::vadd(vq32_sum_l1, vq32_sum_r1);
    d64x2_t vq64_sum_lo   = neon::vmull(neon::vgetlow(vq32_sum_x1),  k2);
    d64x2_t vq64_sum_hi   = neon::vmull(neon::vgethigh(vq32_sum_x1), k2);

    vq64_sum_lo           = neon::vmlal(vq64_sum_lo, neon::vgetlow(vq32_sum_l0r0),  k1);
    vq64_sum_hi           = neon::vmlal(vq64_sum_hi, neon::vgethigh(vq32_sum_l0r0), k1);
    vq64_sum_lo           = neon::vmlal(vq64_sum_lo, neon::vgetlow(vq32_sum_l1r1),  k0);
    vq64_sum_hi           = neon::vmlal(vq64_sum_hi, neon::vgethigh(vq32_sum_l1r1), k0);

    d32x2_t vd32_sum_lo   = neon::vqshrn_n<20>(vq64_sum_lo);
    d32x2_t vd32_sum_hi   = neon::vqshrn_n<20>(vq64_sum_hi);
    vd16_result           = neon::vqrshrn_n<8>(neon::vcombine(vd32_sum_lo, vd32_sum_hi));

    vq32_sum_x0 = vq32_sum_x1;
    vq32_sum_x1 = vq32_sum_x2;
}

AURA_ALWAYS_INLINE DT_VOID Gaussian5x5VCore(uint8x8_t &vdu8_src_p1, uint8x8_t &vdu8_src_p0, uint8x8_t &vdu8_src_c,
                                            uint8x8_t &vdu8_src_n0, uint8x8_t &vdu8_src_n1,
                                            uint16x8_t &vqu16_result, const DT_U16 *kernel)
{
    DT_U16 k0 = kernel[0], k1 = kernel[1], k2 = kernel[2];

    uint16x8_t vqu16_sum_p1n1 = neon::vaddl(vdu8_src_p1, vdu8_src_n1);
    uint16x8_t vqu16_sum_p0n0 = neon::vaddl(vdu8_src_p0, vdu8_src_n0);

    vqu16_sum_p1n1            = neon::vmul(vqu16_sum_p1n1, k0);
    vqu16_sum_p0n0            = neon::vmul(vqu16_sum_p0n0, k1);
    vqu16_result              = neon::vadd(vqu16_sum_p1n1, vqu16_sum_p0n0);
    vqu16_result              = neon::vmla(vqu16_result, neon::vmovl(vdu8_src_c), k2);
}

AURA_ALWAYS_INLINE DT_VOID Gaussian5x5HCore(uint16x8_t &vqu16_sum_x0, uint16x8_t &vqu16_sum_x1, uint16x8_t &vqu16_sum_x2,
                                            uint8x8_t &vdu8_result, const DT_U16 *kernel)
{
    DT_U16 k0 = kernel[0], k1 = kernel[1], k2 = kernel[2];

    uint16x8_t vqu16_sum_l1      = neon::vext<6>(vqu16_sum_x0, vqu16_sum_x1);
    uint16x8_t vqu16_sum_l0      = neon::vext<7>(vqu16_sum_x0, vqu16_sum_x1);
    uint16x8_t vqu16_sum_r0      = neon::vext<1>(vqu16_sum_x1, vqu16_sum_x2);
    uint16x8_t vqu16_sum_r1      = neon::vext<2>(vqu16_sum_x1, vqu16_sum_x2);
    uint32x4_t vqu32_sum_c_lo    = neon::vmull(neon::vgetlow(vqu16_sum_x1), k2);
    uint32x4_t vqu32_sum_c_hi    = neon::vmull(neon::vgethigh(vqu16_sum_x1), k2);
    uint32x4_t vpu32_sum_l0r0_lo = neon::vaddl(neon::vgetlow(vqu16_sum_l0), neon::vgetlow(vqu16_sum_r0));
    uint32x4_t vpu32_sum_l0r0_hi = neon::vaddl(neon::vgethigh(vqu16_sum_l0), neon::vgethigh(vqu16_sum_r0));
    uint32x4_t vqu32_result_lo   = neon::vmla(vqu32_sum_c_lo, vpu32_sum_l0r0_lo, static_cast<DT_U32>(k1));
    uint32x4_t vqu32_result_hi   = neon::vmla(vqu32_sum_c_hi, vpu32_sum_l0r0_hi, static_cast<DT_U32>(k1));
    uint32x4_t vqu32_sum_l1r1_lo = neon::vaddl(neon::vgetlow(vqu16_sum_l1), neon::vgetlow(vqu16_sum_r1));
    uint32x4_t vqu32_sum_l1r1_hi = neon::vaddl(neon::vgethigh(vqu16_sum_l1), neon::vgethigh(vqu16_sum_r1));

    vqu32_result_lo              = neon::vmla(vqu32_result_lo, vqu32_sum_l1r1_lo, static_cast<DT_U32>(k0));
    vqu32_result_hi              = neon::vmla(vqu32_result_hi, vqu32_sum_l1r1_hi, static_cast<DT_U32>(k0));

    uint16x4_t vdu16_result_lo   = neon::vqshrn_n<8>(vqu32_result_lo);
    uint16x4_t vdu16_result_hi   = neon::vqshrn_n<8>(vqu32_result_hi);
    vdu8_result                  = neon::vqrshrn_n<8>(neon::vcombine(vdu16_result_lo, vdu16_result_hi));

    vqu16_sum_x0 = vqu16_sum_x1;
    vqu16_sum_x1 = vqu16_sum_x2;
}

#if defined(AURA_ENABLE_NEON_FP16)
AURA_ALWAYS_INLINE DT_VOID Gaussian5x5VCore(float16x4_t &vdf16_src_p1, float16x4_t &vdf16_src_p0, float16x4_t &vdf16_src_c,
                                            float16x4_t &vdf16_src_n0, float16x4_t &vdf16_src_n1,
                                            float32x4_t &vqf32_result, const DT_F32 *kernel)
{
    DT_F32 k0 = kernel[0], k1 = kernel[1], k2 = kernel[2];

    float32x4_t vqf32_sum_p1n1 = neon::vadd(neon::vcvt<DT_F32>(vdf16_src_p1), neon::vcvt<DT_F32>(vdf16_src_n1));
    float32x4_t vqf32_sum_p0n0 = neon::vadd(neon::vcvt<DT_F32>(vdf16_src_p0), neon::vcvt<DT_F32>(vdf16_src_n0));
    vqf32_result               = neon::vadd(neon::vmul(vqf32_sum_p0n0, k1), neon::vmul(vqf32_sum_p1n1, k0));
    vqf32_result               = neon::vmla(vqf32_result, neon::vcvt<DT_F32>(vdf16_src_c), k2);
}

AURA_ALWAYS_INLINE DT_VOID Gaussian5x5HCore(float32x4_t &vqf32_sum_x0, float32x4_t &vqf32_sum_x1, float32x4_t &vqf32_sum_x2,
                                            float16x4_t &vdf16_result, const DT_F32 *kernel)
{
    DT_F32 k0 = kernel[0], k1 = kernel[1], k2 = kernel[2];

    float32x4_t vqf32_sum_l1   = neon::vext<2>(vqf32_sum_x0, vqf32_sum_x1);
    float32x4_t vqf32_sum_l0   = neon::vext<3>(vqf32_sum_x0, vqf32_sum_x1);
    float32x4_t vqf32_sum_r0   = neon::vext<1>(vqf32_sum_x1, vqf32_sum_x2);
    float32x4_t vqf32_sum_r1   = neon::vext<2>(vqf32_sum_x1, vqf32_sum_x2);
    float32x4_t vqf32_sum_l1r1 = neon::vmul(neon::vadd(vqf32_sum_l1, vqf32_sum_r1), k0);
    float32x4_t vqf32_sum_l0r0 = neon::vmul(neon::vadd(vqf32_sum_l0, vqf32_sum_r0), k1);
    float32x4_t vqf32_result   = neon::vadd(vqf32_sum_l1r1, vqf32_sum_l0r0);

    vqf32_result               = neon::vadd(vqf32_result, neon::vmul(vqf32_sum_x1, k2));
    vdf16_result               = neon::vcvt<MI_F16>(vqf32_result);

    vqf32_sum_x0 = vqf32_sum_x1;
    vqf32_sum_x1 = vqf32_sum_x2;
}
#endif // AURA_ENABLE_NEON_FP16

AURA_ALWAYS_INLINE DT_VOID Gaussian5x5VCore(float32x4_t &vqf32_src_p1, float32x4_t &vqf32_src_p0, float32x4_t &vqf32_src_c,
                                            float32x4_t &vqf32_src_n0, float32x4_t &vqf32_src_n1,
                                            float32x4_t &vqf32_result, const DT_F32 *kernel)
{
    DT_F32 k0 = kernel[0], k1 = kernel[1], k2 = kernel[2];

    float32x4_t vqf32_sum_p1n1 = neon::vadd(vqf32_src_p1, vqf32_src_n1);
    float32x4_t vqf32_sum_p0n0 = neon::vadd(vqf32_src_p0, vqf32_src_n0);

    vqf32_sum_p1n1             = neon::vmul(vqf32_sum_p1n1, k0);
    vqf32_sum_p0n0             = neon::vmul(vqf32_sum_p0n0, k1);
    vqf32_result               = neon::vadd(vqf32_sum_p0n0, vqf32_sum_p1n1);
    vqf32_result               = neon::vmla(vqf32_result, vqf32_src_c, k2);
}

AURA_ALWAYS_INLINE DT_VOID Gaussian5x5HCore(float32x4_t &vqf32_sum_x0, float32x4_t &vqf32_sum_x1, float32x4_t &vqf32_sum_x2,
                                            float32x4_t &vqf32_result, const DT_F32 *kernel)
{
    DT_F32 k0 = kernel[0], k1 = kernel[1], k2 = kernel[2];

    float32x4_t vqf32_sum_l1   = neon::vext<2>(vqf32_sum_x0, vqf32_sum_x1);
    float32x4_t vqf32_sum_l0   = neon::vext<3>(vqf32_sum_x0, vqf32_sum_x1);
    float32x4_t vqf32_sum_r0   = neon::vext<1>(vqf32_sum_x1, vqf32_sum_x2);
    float32x4_t vqf32_sum_r1   = neon::vext<2>(vqf32_sum_x1, vqf32_sum_x2);
    float32x4_t vqf32_sum_l0r0 = neon::vadd(vqf32_sum_l0, vqf32_sum_r0);
    float32x4_t vqf32_sum_l1r1 = neon::vadd(vqf32_sum_l1, vqf32_sum_r1);

    vqf32_sum_l0r0             = neon::vmul(vqf32_sum_l0r0, k1);
    vqf32_sum_l1r1             = neon::vmul(vqf32_sum_l1r1, k0);
    vqf32_result               = neon::vadd(vqf32_sum_l0r0, vqf32_sum_l1r1);
    vqf32_result               = neon::vmla(vqf32_result, vqf32_sum_x1, k2);

    vqf32_sum_x0 = vqf32_sum_x1;
    vqf32_sum_x1 = vqf32_sum_x2;
}

template <typename Tp, BorderType BORDER_TYPE, DT_S32 C, typename Kt>
static DT_VOID Gaussian5x5Row(const Tp *src_p1, const Tp *src_p0, const Tp *src_c, const Tp *src_n0, const Tp *src_n1,
                              Tp *dst, const Kt *kernel, const std::vector<Tp> &border_value, DT_S32 width)
{
    using MVType    = typename std::conditional<std::is_same<Tp, DT_F32>::value, typename neon::MQVector<Tp, C>::MVType,
                      typename neon::MDVector<Tp, C>::MVType>::type;
    using MVSumType = typename std::conditional<std::is_same<Tp, DT_F32>::value, MVType,
                      typename neon::MQVector<typename Promote<Tp>::Type, C>::MVType>::type;

    constexpr DT_S32 ELEM_COUNTS = static_cast<DT_S32>(sizeof(MVType) / C / sizeof(Tp));
    constexpr DT_S32 VOFFSET     = ELEM_COUNTS * C;
    const DT_S32 width_align     = (width & -ELEM_COUNTS) * C;

    MVType mv_src_p1[3], mv_src_p0[3], mv_src_c[3], mv_src_n0[3], mv_src_n1[3], mv_result;
    MVSumType mv_sum[3];

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

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            mv_src_p1[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mv_src_p1[1].val[ch], src_p1[ch], border_value[ch]);
            mv_src_p0[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mv_src_p0[1].val[ch], src_p0[ch], border_value[ch]);
            mv_src_c[0].val[ch]  = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mv_src_c[1].val[ch],  src_c[ch],  border_value[ch]);
            mv_src_n0[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mv_src_n0[1].val[ch], src_n0[ch], border_value[ch]);
            mv_src_n1[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mv_src_n1[1].val[ch], src_n1[ch], border_value[ch]);

            Gaussian5x5VCore(mv_src_p1[0].val[ch], mv_src_p0[0].val[ch], mv_src_c[0].val[ch],
                             mv_src_n0[0].val[ch], mv_src_n1[0].val[ch], mv_sum[0].val[ch], kernel);
            Gaussian5x5VCore(mv_src_p1[1].val[ch], mv_src_p0[1].val[ch], mv_src_c[1].val[ch],
                             mv_src_n0[1].val[ch], mv_src_n1[1].val[ch], mv_sum[1].val[ch], kernel);
            Gaussian5x5VCore(mv_src_p1[2].val[ch], mv_src_p0[2].val[ch], mv_src_c[2].val[ch],
                             mv_src_n0[2].val[ch], mv_src_n1[2].val[ch], mv_sum[2].val[ch], kernel);
            Gaussian5x5HCore(mv_sum[0].val[ch], mv_sum[1].val[ch], mv_sum[2].val[ch], mv_result.val[ch], kernel);
        }
        neon::vstore(dst, mv_result);
    }

    // middle
    for (DT_S32 x = VOFFSET; x < (width_align - VOFFSET); x += VOFFSET)
    {
        neon::vload(src_p1 + x + VOFFSET, mv_src_p1[2]);
        neon::vload(src_p0 + x + VOFFSET, mv_src_p0[2]);
        neon::vload(src_c  + x + VOFFSET, mv_src_c[2]);
        neon::vload(src_n0 + x + VOFFSET, mv_src_n0[2]);
        neon::vload(src_n1 + x + VOFFSET, mv_src_n1[2]);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            Gaussian5x5VCore(mv_src_p1[2].val[ch], mv_src_p0[2].val[ch], mv_src_c[2].val[ch],
                             mv_src_n0[2].val[ch], mv_src_n1[2].val[ch], mv_sum[2].val[ch], kernel);
            Gaussian5x5HCore(mv_sum[0].val[ch], mv_sum[1].val[ch], mv_sum[2].val[ch], mv_result.val[ch], kernel);
        }
        neon::vstore(dst + x, mv_result);
    }

    // back
    {
        if (width_align != width * C)
        {
            DT_S32 x = (width - (ELEM_COUNTS << 1)) * C;

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

            for (DT_S32 ch = 0; ch < C; ch++)
            {
                Gaussian5x5VCore(mv_src_p1[0].val[ch], mv_src_p0[0].val[ch], mv_src_c[0].val[ch],
                                 mv_src_n0[0].val[ch], mv_src_n1[0].val[ch], mv_sum[0].val[ch], kernel);
                Gaussian5x5VCore(mv_src_p1[1].val[ch], mv_src_p0[1].val[ch], mv_src_c[1].val[ch],
                                 mv_src_n0[1].val[ch], mv_src_n1[1].val[ch], mv_sum[1].val[ch], kernel);
                Gaussian5x5VCore(mv_src_p1[2].val[ch], mv_src_p0[2].val[ch], mv_src_c[2].val[ch],
                                 mv_src_n0[2].val[ch], mv_src_n1[2].val[ch], mv_sum[2].val[ch], kernel);
                Gaussian5x5HCore(mv_sum[0].val[ch], mv_sum[1].val[ch], mv_sum[2].val[ch], mv_result.val[ch], kernel);
            }
            neon::vstore(dst + x, mv_result);
        }
    }

    // right
    {
        DT_S32 x    = (width - ELEM_COUNTS) * C;
        DT_S32 last = (width - 1) * C;

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            mv_src_p1[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mv_src_p1[2].val[ch], src_p1[last + ch], border_value[ch]);
            mv_src_p0[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mv_src_p0[2].val[ch], src_p0[last + ch], border_value[ch]);
            mv_src_c[2].val[ch]  = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mv_src_c[2].val[ch],  src_c[last + ch],  border_value[ch]);
            mv_src_n0[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mv_src_n0[2].val[ch], src_n0[last + ch], border_value[ch]);
            mv_src_n1[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mv_src_n1[2].val[ch], src_n1[last + ch], border_value[ch]);

            Gaussian5x5VCore(mv_src_p1[2].val[ch], mv_src_p0[2].val[ch], mv_src_c[2].val[ch],
                             mv_src_n0[2].val[ch], mv_src_n1[2].val[ch], mv_sum[2].val[ch], kernel);
            Gaussian5x5HCore(mv_sum[0].val[ch], mv_sum[1].val[ch], mv_sum[2].val[ch], mv_result.val[ch], kernel);
        }
        neon::vstore(dst + x, mv_result);
    }
}

template <typename Tp, BorderType BORDER_TYPE, DT_S32 C>
static Status Gaussian5x5NeonImpl(const Mat &src, Mat &dst, const Mat &kmat, const std::vector<Tp> &border_value, const Tp *border_buffer,
                                  DT_S32 start_row, DT_S32 end_row)
{
    using Kt = typename std::conditional<sizeof(Tp) == 4, Tp, typename Promote<Tp>::Type>::type;

    DT_S32 width  = src.GetSizes().m_width;

    const Kt *kernel = kmat.Ptr<Kt>(0);

    const Tp *src_p1 = src.Ptr<Tp, BORDER_TYPE>(start_row - 2, border_buffer);
    const Tp *src_p0 = src.Ptr<Tp, BORDER_TYPE>(start_row - 1, border_buffer);
    const Tp *src_c  = src.Ptr<Tp>(start_row);
    const Tp *src_n0 = src.Ptr<Tp, BORDER_TYPE>(start_row + 1, border_buffer);
    const Tp *src_n1 = src.Ptr<Tp, BORDER_TYPE>(start_row + 2, border_buffer);

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        Tp *dst_row = dst.Ptr<Tp>(y);
        Gaussian5x5Row<Tp, BORDER_TYPE, C, Kt>(src_p1, src_p0, src_c, src_n0, src_n1, dst_row, kernel, border_value, width);

        src_p1 = src_p0;
        src_p0 = src_c;
        src_c  = src_n0;
        src_n0 = src_n1;
        src_n1 = src.Ptr<Tp, BORDER_TYPE>(y + 3, border_buffer);
    }

    return Status::OK;
}

template<typename Tp, BorderType BORDER_TYPE>
static Status Gaussian5x5NeonHelper(Context *ctx, const Mat &src, Mat &dst, const Mat &kmat,
                                    const std::vector<Tp> &border_value, const Tp *border_buffer, const OpTarget &target)
{
    AURA_UNUSED(target);
    Status ret     = Status::ERROR;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
        return ret;
    }

    DT_S32 height  = dst.GetSizes().m_height;
    DT_S32 channel = dst.GetSizes().m_channel;

    switch (channel)
    {
        case 1:
        {
            ret = wp->ParallelFor(0, height, Gaussian5x5NeonImpl<Tp, BORDER_TYPE, 1>,
                                  std::cref(src), std::ref(dst), std::cref(kmat),
                                  std::cref(border_value), border_buffer);
            break;
        }

        case 2:
        {
            ret = wp->ParallelFor(0, height, Gaussian5x5NeonImpl<Tp, BORDER_TYPE, 2>,
                                  std::cref(src), std::ref(dst), std::cref(kmat),
                                  std::cref(border_value), border_buffer);
            break;
        }

        case 3:
        {
            ret = wp->ParallelFor(0, height, Gaussian5x5NeonImpl<Tp, BORDER_TYPE, 3>,
                                  std::cref(src), std::ref(dst), std::cref(kmat),
                                  std::cref(border_value), border_buffer);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "Unsupported channel");
            break;
        }
    }

    AURA_RETURN(ctx, ret);
}

template <typename Tp>
static Status Gaussian5x5NeonHelper(Context *ctx, const Mat &src, Mat &dst, const Mat &kmat,
                                    BorderType &border_type, const Scalar &border_value, const OpTarget &target)
{
    Status ret = Status::ERROR;

    Tp *border_buffer = DT_NULL;
    std::vector<Tp> vec_border_value = border_value.ToVector<Tp>();

    DT_S32 width   = dst.GetSizes().m_width;
    DT_S32 channel = dst.GetSizes().m_channel;

    switch (border_type)
    {
        case BorderType::CONSTANT:
        {
            border_buffer = CreateBorderBuffer(ctx, width, channel, vec_border_value);
            if (DT_NULL == border_buffer)
            {
                AURA_ADD_ERROR_STRING(ctx, "CreateBorderBuffer failed");
                return Status::ERROR;
            }

            ret = Gaussian5x5NeonHelper<Tp, BorderType::CONSTANT>(ctx, src, dst, kmat, vec_border_value, border_buffer, target);

            AURA_FREE(ctx, border_buffer);
            break;
        }

        case BorderType::REPLICATE:
        {
            ret = Gaussian5x5NeonHelper<Tp, BorderType::REPLICATE>(ctx, src, dst, kmat, vec_border_value, border_buffer, target);
            break;
        }

        case BorderType::REFLECT_101:
        {
            ret = Gaussian5x5NeonHelper<Tp, BorderType::REFLECT_101>(ctx, src, dst, kmat, vec_border_value, border_buffer, target);
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

Status Gaussian5x5Neon(Context *ctx, const Mat &src, Mat &dst, const Mat &kmat,
                       BorderType &border_type, const Scalar &border_value, const OpTarget &target)
{
    Status ret = Status::ERROR;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = Gaussian5x5NeonHelper<DT_U8>(ctx, src, dst, kmat, border_type, border_value, target);
            break;
        }

        case ElemType::U16:
        {
            ret = Gaussian5x5NeonHelper<DT_U16>(ctx, src, dst, kmat, border_type, border_value, target);
            break;
        }

        case ElemType::S16:
        {
            ret = Gaussian5x5NeonHelper<DT_S16>(ctx, src, dst, kmat, border_type, border_value, target);
            break;
        }

#if defined(AURA_ENABLE_NEON_FP16)
        case ElemType::F16:
        {
            ret = Gaussian5x5NeonHelper<MI_F16>(ctx, src, dst, kmat, border_type, border_value, target);
            break;
        }
#endif // AURA_ENABLE_NEON_FP16

        case ElemType::F32:
        {
            ret = Gaussian5x5NeonHelper<DT_F32>(ctx, src, dst, kmat, border_type, border_value, target);
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
