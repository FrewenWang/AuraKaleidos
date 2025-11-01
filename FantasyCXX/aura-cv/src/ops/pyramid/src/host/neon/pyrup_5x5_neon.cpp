#include "pyrup_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

AURA_ALWAYS_INLINE AURA_VOID PyrUp5x5VCore(uint8x8_t &vdu8_src_p, uint8x8_t &vdu8_src_c, uint8x8_t &vdu8_src_n,
                                         uint32x4_t &vqu32_esum_lo, uint32x4_t &vqu32_esum_hi,
                                         uint32x4_t &vqu32_osum_lo, uint32x4_t &vqu32_osum_hi,
                                         const MI_U16 *kernel)
{
    MI_U16 k0 = kernel[0], k1 = kernel[1], k2 = kernel[2];

    uint16x8_t vqu16_src_c  = neon::vmovl(vdu8_src_c);
    uint16x8_t vqu16_sum0   = neon::vaddl(vdu8_src_p, vdu8_src_n);
    uint16x8_t vqu16_sum1   = neon::vaddl(vdu8_src_c, vdu8_src_n);
    uint32x4_t vqu32_sum_lo = neon::vmull(neon::vgetlow(vqu16_sum0), k0);
    uint32x4_t vqu32_sum_hi = neon::vmull(neon::vgethigh(vqu16_sum0), k0);
    vqu32_esum_lo           = neon::vmlal(vqu32_sum_lo, neon::vgetlow(vqu16_src_c), k2);
    vqu32_esum_hi           = neon::vmlal(vqu32_sum_hi, neon::vgethigh(vqu16_src_c), k2);
    vqu32_osum_lo           = neon::vmull(neon::vgetlow(vqu16_sum1), k1);
    vqu32_osum_hi           = neon::vmull(neon::vgethigh(vqu16_sum1), k1);
}

AURA_ALWAYS_INLINE AURA_VOID PyrUp5x5HCore(uint32x4_t &vqu32_sum_cx0_hi,  uint32x4_t &vqu32_sum_n0x0_hi,
                                         uint32x4_t &vqu32_sum_cx1_lo,  uint32x4_t &vqu32_sum_cx1_hi,
                                         uint32x4_t &vqu32_sum_n0x1_lo, uint32x4_t &vqu32_sum_n0x1_hi,
                                         uint32x4_t &vqu32_sum_cx2_lo,  uint32x4_t &vqu32_sum_cx2_hi,
                                         uint32x4_t &vqu32_sum_n0x2_lo, uint32x4_t &vqu32_sum_n0x2_hi,
                                         uint8x8x2_t &v2du8_dst_c0,     uint8x8x2_t &v2du8_dst_c1,
                                         const MI_U16 *kernel)
{
    MI_U16 k0 = kernel[0], k1 = kernel[1], k2 = kernel[2];

    const MI_U32 u32_k0 = static_cast<MI_U32>(k0);
    const MI_U32 u32_k1 = static_cast<MI_U32>(k1);
    const MI_U32 u32_k2 = static_cast<MI_U32>(k2);

    // current line
    uint32x4_t vqu32_sum_cl0, vqu32_sum_cl1, vqu32_sum_cr0, vqu32_sum_cr1;
    uint16x8_t vqu16_result_cc, vqu16_result_cr;

    vqu32_sum_cl0                  = neon::vext<3>(vqu32_sum_cx0_hi, vqu32_sum_cx1_lo);
    vqu32_sum_cl1                  = neon::vext<3>(vqu32_sum_cx1_lo, vqu32_sum_cx1_hi);
    vqu32_sum_cr0                  = neon::vext<1>(vqu32_sum_cx1_lo, vqu32_sum_cx1_hi);
    vqu32_sum_cr1                  = neon::vext<1>(vqu32_sum_cx1_hi, vqu32_sum_cx2_lo);

    uint32x4_t vqu32_result_c_l0r0 = neon::vadd(vqu32_sum_cl0, vqu32_sum_cr0);
    uint32x4_t vqu32_result_c_l1r1 = neon::vadd(vqu32_sum_cl1, vqu32_sum_cr1);
    vqu32_result_c_l0r0            = neon::vmul(vqu32_result_c_l0r0, u32_k0);
    vqu32_result_c_l1r1            = neon::vmul(vqu32_result_c_l1r1, u32_k0);
    vqu32_result_c_l0r0            = neon::vmla(vqu32_result_c_l0r0, vqu32_sum_cx1_lo, u32_k2);
    vqu32_result_c_l1r1            = neon::vmla(vqu32_result_c_l1r1, vqu32_sum_cx1_hi, u32_k2);
    vqu16_result_cc                = neon::vcombine(neon::vqrshrn_n<10>(vqu32_result_c_l0r0), neon::vqrshrn_n<10>(vqu32_result_c_l1r1));

    uint32x4_t vqu32_result_c_cr0  = neon::vadd(vqu32_sum_cx1_lo, vqu32_sum_cr0);
    uint32x4_t vqu32_result_c_cr1  = neon::vadd(vqu32_sum_cx1_hi, vqu32_sum_cr1);
    vqu32_result_c_cr0             = neon::vmul(vqu32_result_c_cr0, u32_k1);
    vqu32_result_c_cr1             = neon::vmul(vqu32_result_c_cr1, u32_k1);
    vqu16_result_cr                = neon::vcombine(neon::vqrshrn_n<10>(vqu32_result_c_cr0), neon::vqrshrn_n<10>(vqu32_result_c_cr1));
    v2du8_dst_c0                   = neon::vzip(neon::vqrshrn_n<8>(vqu16_result_cc), neon::vqrshrn_n<8>(vqu16_result_cr));

    vqu32_sum_cx0_hi = vqu32_sum_cx1_hi;
    vqu32_sum_cx1_lo = vqu32_sum_cx2_lo; vqu32_sum_cx1_hi = vqu32_sum_cx2_hi;

    // next line
    uint32x4_t vqu32_sum_n0l0, vqu32_sum_n0l1, vqu32_sum_n0r0, vqu32_sum_n0r1;
    uint16x8_t vqu16_result_n0c, vqu16_result_n0r;

    vqu32_sum_n0l0                  = neon::vext<3>(vqu32_sum_n0x0_hi, vqu32_sum_n0x1_lo);
    vqu32_sum_n0l1                  = neon::vext<3>(vqu32_sum_n0x1_lo, vqu32_sum_n0x1_hi);
    vqu32_sum_n0r0                  = neon::vext<1>(vqu32_sum_n0x1_lo, vqu32_sum_n0x1_hi);
    vqu32_sum_n0r1                  = neon::vext<1>(vqu32_sum_n0x1_hi, vqu32_sum_n0x2_lo);

    uint32x4_t vqu32_result_n0_l0r0 = neon::vadd(vqu32_sum_n0l0, vqu32_sum_n0r0);
    uint32x4_t vqu32_result_n0_l1r1 = neon::vadd(vqu32_sum_n0l1, vqu32_sum_n0r1);
    vqu32_result_n0_l0r0            = neon::vmul(vqu32_result_n0_l0r0, u32_k0);
    vqu32_result_n0_l1r1            = neon::vmul(vqu32_result_n0_l1r1, u32_k0);
    vqu32_result_n0_l0r0            = neon::vmla(vqu32_result_n0_l0r0, vqu32_sum_n0x1_lo, u32_k2);
    vqu32_result_n0_l1r1            = neon::vmla(vqu32_result_n0_l1r1, vqu32_sum_n0x1_hi, u32_k2);
    vqu16_result_n0c                = neon::vcombine(neon::vqrshrn_n<10>(vqu32_result_n0_l0r0), neon::vqrshrn_n<10>(vqu32_result_n0_l1r1));

    uint32x4_t vqu32_result_n0_cr0  = neon::vadd(vqu32_sum_n0x1_lo, vqu32_sum_n0r0);
    uint32x4_t vqu32_result_n0_cr1  = neon::vadd(vqu32_sum_n0x1_hi, vqu32_sum_n0r1);
    vqu32_result_n0_cr0             = neon::vmul(vqu32_result_n0_cr0, u32_k1);
    vqu32_result_n0_cr1             = neon::vmul(vqu32_result_n0_cr1, u32_k1);
    vqu16_result_n0r                = neon::vcombine(neon::vqrshrn_n<10>(vqu32_result_n0_cr0), neon::vqrshrn_n<10>(vqu32_result_n0_cr1));
    v2du8_dst_c1                    = neon::vzip(neon::vqrshrn_n<8>(vqu16_result_n0c), neon::vqrshrn_n<8>(vqu16_result_n0r));

    vqu32_sum_n0x0_hi = vqu32_sum_n0x1_hi;
    vqu32_sum_n0x1_lo = vqu32_sum_n0x2_lo; vqu32_sum_n0x1_hi  = vqu32_sum_n0x2_hi;
}

template <typename d16x8_t, typename d32x4_t, typename Kt, typename D16 = typename neon::Scalar<d16x8_t>::SType>
AURA_ALWAYS_INLINE AURA_VOID PyrUp5x5VCore(d16x8_t &vd16_p_src,   d16x8_t &vd16_src_c, d16x8_t &vd16_src_n,
                                         d32x4_t &vq32_esum_lo, d32x4_t &vq32_esum_hi, d32x4_t &vq32_osum_lo,
                                         d32x4_t &vq32_osum_hi, const Kt *kernel)
{
    Kt k0 = kernel[0], k1 = kernel[1], k2 = kernel[2];

    const D16 d16_k2 = static_cast<D16>(k2);
    d32x4_t vq32_sum0_lo = neon::vaddl(neon::vgetlow(vd16_p_src), neon::vgetlow(vd16_src_n));
    d32x4_t vq32_sum0_hi = neon::vaddl(neon::vgethigh(vd16_p_src), neon::vgethigh(vd16_src_n));
    d32x4_t vq32_sum1_lo = neon::vaddl(neon::vgetlow(vd16_src_c), neon::vgetlow(vd16_src_n));
    d32x4_t vq32_sum1_hi = neon::vaddl(neon::vgethigh(vd16_src_c), neon::vgethigh(vd16_src_n));

    vq32_esum_lo         = neon::vmul(vq32_sum0_lo, k0);
    vq32_esum_hi         = neon::vmul(vq32_sum0_hi, k0);
    vq32_esum_lo         = neon::vmlal(vq32_esum_lo, neon::vgetlow(vd16_src_c), d16_k2);
    vq32_esum_hi         = neon::vmlal(vq32_esum_hi, neon::vgethigh(vd16_src_c), d16_k2);
    vq32_osum_lo         = neon::vmul(vq32_sum1_lo, k1);
    vq32_osum_hi         = neon::vmul(vq32_sum1_hi, k1);
}

template <typename d32x4_t, typename d16x8x2_t, typename Kt,
          typename d64x2_t = typename neon::QVector<typename Promote<Kt>::Type>::VType>
AURA_ALWAYS_INLINE AURA_VOID PyrUp5x5HCore(d32x4_t &vq32_sum_cx0_hi, d32x4_t &vq32_sum_nx0_hi,
                                         d32x4_t &vq32_sum_cx1_lo, d32x4_t &vq32_sum_cx1_hi,
                                         d32x4_t &vq32_sum_nx1_lo, d32x4_t &vq32_sum_nx1_hi,
                                         d32x4_t &vq32_sum_cx2_lo, d32x4_t &vq32_sum_cx2_hi,
                                         d32x4_t &vq32_sum_nx2_lo, d32x4_t &vq32_sum_nx2_hi,
                                         d16x8x2_t &vq16_dst_c0, d16x8x2_t &vq16_dst_c1, const Kt *kernel)
{
    // current line
    Kt k0 = kernel[0], k1 = kernel[1], k2 = kernel[2];

    d32x4_t vq32_sum_cl0, vq32_sum_cl1, vq32_sum_cr0, vq32_sum_cr1;
    d32x4_t vq32_result_even_c_lo, vq32_result_even_c_hi;
    d32x4_t vq32_result_odd_c_lo,  vq32_result_odd_c_hi;

    vq32_sum_cl0                = neon::vext<3>(vq32_sum_cx0_hi, vq32_sum_cx1_lo);
    vq32_sum_cl1                = neon::vext<3>(vq32_sum_cx1_lo, vq32_sum_cx1_hi);
    vq32_sum_cr0                = neon::vext<1>(vq32_sum_cx1_lo, vq32_sum_cx1_hi);
    vq32_sum_cr1                = neon::vext<1>(vq32_sum_cx1_hi, vq32_sum_cx2_lo);

    d32x4_t vq32_esum_c_l0r0    = neon::vadd(vq32_sum_cl0, vq32_sum_cr0);
    d32x4_t vq32_esum_c_l1r1    = neon::vadd(vq32_sum_cl1, vq32_sum_cr1);

    d64x2_t vq64_esum_c_l0r0_lo = neon::vmull(neon::vgetlow(vq32_esum_c_l0r0), k0);
    d64x2_t vq64_esum_c_l0r0_hi = neon::vmull(neon::vgethigh(vq32_esum_c_l0r0), k0);
    vq64_esum_c_l0r0_lo         = neon::vmlal(vq64_esum_c_l0r0_lo, neon::vgetlow(vq32_sum_cx1_lo), k2);
    vq64_esum_c_l0r0_hi         = neon::vmlal(vq64_esum_c_l0r0_hi, neon::vgethigh(vq32_sum_cx1_lo), k2);
    vq32_result_even_c_lo       = neon::vcombine(neon::vqrshrn_n<10>(vq64_esum_c_l0r0_lo), neon::vqrshrn_n<10>(vq64_esum_c_l0r0_hi));

    d64x2_t vq64_esum_c_l1r1_lo = neon::vmull(neon::vgetlow(vq32_esum_c_l1r1), k0);
    d64x2_t vq64_esum_c_l1r1_hi = neon::vmull(neon::vgethigh(vq32_esum_c_l1r1), k0);
    vq64_esum_c_l1r1_lo         = neon::vmlal(vq64_esum_c_l1r1_lo, neon::vgetlow(vq32_sum_cx1_hi), k2);
    vq64_esum_c_l1r1_hi         = neon::vmlal(vq64_esum_c_l1r1_hi, neon::vgethigh(vq32_sum_cx1_hi), k2);
    vq32_result_even_c_hi       = neon::vcombine(neon::vqrshrn_n<10>(vq64_esum_c_l1r1_lo), neon::vqrshrn_n<10>(vq64_esum_c_l1r1_hi));

    // current line odd
    d32x4_t vq32_osum_cr0       = neon::vadd(vq32_sum_cx1_lo, vq32_sum_cr0);
    d32x4_t vq32_osum_cr1       = neon::vadd(vq32_sum_cx1_hi, vq32_sum_cr1);

    d64x2_t vq64_osum_c_cr0_lo  = neon::vmull(neon::vgetlow(vq32_osum_cr0), k1);
    d64x2_t vq64_osum_c_cr0_hi  = neon::vmull(neon::vgethigh(vq32_osum_cr0), k1);
    vq32_result_odd_c_lo        = neon::vcombine(neon::vqrshrn_n<10>(vq64_osum_c_cr0_lo), neon::vqrshrn_n<10>(vq64_osum_c_cr0_hi));

    d64x2_t vq64_osum_c_cr1_lo  = neon::vmull(neon::vgetlow(vq32_osum_cr1), k1);
    d64x2_t vq64_osum_c_cr1_hi  = neon::vmull(neon::vgethigh(vq32_osum_cr1), k1);
    vq32_result_odd_c_hi        = neon::vcombine(neon::vqrshrn_n<10>(vq64_osum_c_cr1_lo), neon::vqrshrn_n<10>(vq64_osum_c_cr1_hi));

    vq16_dst_c0.val[0]          = neon::vcombine(neon::vqrshrn_n<16>(vq32_result_even_c_lo), neon::vqrshrn_n<16>(vq32_result_even_c_hi));
    vq16_dst_c0.val[1]          = neon::vcombine(neon::vqrshrn_n<16>(vq32_result_odd_c_lo), neon::vqrshrn_n<16>(vq32_result_odd_c_hi));
    vq16_dst_c0                 = neon::vzip(vq16_dst_c0.val[0], vq16_dst_c0.val[1]);

    vq32_sum_cx0_hi = vq32_sum_cx1_hi;
    vq32_sum_cx1_lo = vq32_sum_cx2_lo; vq32_sum_cx1_hi  = vq32_sum_cx2_hi;

    // next line
    d32x4_t vq32_sum_nl0, vq32_sum_nl1, vq32_sum_nr0, vq32_sum_nr1;

    vq32_sum_nl0                = neon::vext<3>(vq32_sum_nx0_hi, vq32_sum_nx1_lo);
    vq32_sum_nl1                = neon::vext<3>(vq32_sum_nx1_lo, vq32_sum_nx1_hi);
    vq32_sum_nr0                = neon::vext<1>(vq32_sum_nx1_lo, vq32_sum_nx1_hi);
    vq32_sum_nr1                = neon::vext<1>(vq32_sum_nx1_hi, vq32_sum_nx2_lo);

    d32x4_t vq32_esum_n_l0r0    = neon::vadd(vq32_sum_nl0, vq32_sum_nr0);
    d32x4_t vq32_esum_n_l1r1    = neon::vadd(vq32_sum_nl1, vq32_sum_nr1);

    d64x2_t vq64_esum_n_l0r0_lo = neon::vmull(neon::vgetlow(vq32_esum_n_l0r0), k0);
    d64x2_t vq64_esum_n_l0r0_hi = neon::vmull(neon::vgethigh(vq32_esum_n_l0r0), k0);
    vq64_esum_n_l0r0_lo         = neon::vmlal(vq64_esum_n_l0r0_lo, neon::vgetlow(vq32_sum_nx1_lo), k2);
    vq64_esum_n_l0r0_hi         = neon::vmlal(vq64_esum_n_l0r0_hi, neon::vgethigh(vq32_sum_nx1_lo), k2);
    vq32_result_even_c_lo       = neon::vcombine(neon::vqrshrn_n<10>(vq64_esum_n_l0r0_lo), neon::vqrshrn_n<10>(vq64_esum_n_l0r0_hi));

    d64x2_t vq64_esum_n_hi_lo   = neon::vmull(neon::vgetlow(vq32_esum_n_l1r1), k0);
    d64x2_t vq64_esum_n_hi_hi   = neon::vmull(neon::vgethigh(vq32_esum_n_l1r1), k0);
    vq64_esum_n_hi_lo           = neon::vmlal(vq64_esum_n_hi_lo, neon::vgetlow(vq32_sum_nx1_hi), k2);
    vq64_esum_n_hi_hi           = neon::vmlal(vq64_esum_n_hi_hi, neon::vgethigh(vq32_sum_nx1_hi), k2);
    vq32_result_even_c_hi       = neon::vcombine(neon::vqrshrn_n<10>(vq64_esum_n_hi_lo), neon::vqrshrn_n<10>(vq64_esum_n_hi_hi));

    // next line odd
    d32x4_t vq32_osum_n_cr0     = neon::vadd(vq32_sum_nx1_lo, vq32_sum_nr0);
    d32x4_t vq32_osum_n_cr1     = neon::vadd(vq32_sum_nx1_hi, vq32_sum_nr1);

    d64x2_t vq64_osum_n_cr0_lo  = neon::vmull(neon::vgetlow(vq32_osum_n_cr0), k1);
    d64x2_t vq64_osum_n_cr0_hi  = neon::vmull(neon::vgethigh(vq32_osum_n_cr0), k1);
    vq32_result_odd_c_lo        = neon::vcombine(neon::vqrshrn_n<10>(vq64_osum_n_cr0_lo), neon::vqrshrn_n<10>(vq64_osum_n_cr0_hi));

    d64x2_t vq64_osum_n_cr1_lo  = neon::vmull(neon::vgetlow(vq32_osum_n_cr1), k1);
    d64x2_t vq64_osum_n_cr1_hi  = neon::vmull(neon::vgethigh(vq32_osum_n_cr1), k1);
    vq32_result_odd_c_hi        = neon::vcombine(neon::vqrshrn_n<10>(vq64_osum_n_cr1_lo), neon::vqrshrn_n<10>(vq64_osum_n_cr1_hi));

    vq16_dst_c1.val[0]          = neon::vcombine(neon::vqrshrn_n<16>(vq32_result_even_c_lo), neon::vqrshrn_n<16>(vq32_result_even_c_hi));
    vq16_dst_c1.val[1]          = neon::vcombine(neon::vqrshrn_n<16>(vq32_result_odd_c_lo), neon::vqrshrn_n<16>(vq32_result_odd_c_hi));
    vq16_dst_c1                 = neon::vzip(vq16_dst_c1.val[0], vq16_dst_c1.val[1]);

    vq32_sum_nx0_hi = vq32_sum_nx1_hi;
    vq32_sum_nx1_lo = vq32_sum_nx2_lo; vq32_sum_nx1_hi  = vq32_sum_nx2_hi;
}

template <typename Tp, BorderType BORDER_TYPE, typename Kt>
static AURA_VOID PyrUp5x5TwoRow(const Tp *src_p, const Tp *src_c, const Tp *src_n, Tp *dst_c0, Tp *dst_c1, MI_S32 width, const Kt *kernel)
{
    using VSt       = typename std::conditional<std::is_same<Tp, MI_U8>::value, typename neon::DVector<Tp>::VType,
                                                typename neon::QVector<Tp>::VType>::type; // 8x8 or 16x8
    using VDt       = typename std::conditional<std::is_same<Tp, MI_U8>::value, typename neon::MDVector<Tp, 2>::MVType,
                                                typename neon::MQVector<Tp, 2>::MVType>::type;
    using VqSumType = typename std::conditional<std::is_same<Tp, MI_U8>::value, typename neon::QVector<MI_U32>::VType,
                                                typename neon::QVector<typename Promote<Tp>::Type>::VType>::type; // 32x4_t


    constexpr MI_S32 ELEM_COUNTS = static_cast<MI_S32>(sizeof(VSt) / sizeof(Tp)); // src
    constexpr MI_S32 VOFFSET     = ELEM_COUNTS;

    const MI_S32 width_align = width & -ELEM_COUNTS; // src_width
    const Tp border_value    = 0;

    VSt v_src_p[3], v_src_c[3], v_src_n[3];
    VqSumType v_esum[6], v_osum[6];
    VDt mv_result[2];

    // left
    {
        neon::vload(src_p,           v_src_p[1]);
        neon::vload(src_p + VOFFSET, v_src_p[2]);
        neon::vload(src_c,           v_src_c[1]);
        neon::vload(src_c + VOFFSET, v_src_c[2]);
        neon::vload(src_n,           v_src_n[1]);
        neon::vload(src_n + VOFFSET, v_src_n[2]);

        v_src_p[0] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(v_src_p[1], src_p[0], border_value);
        v_src_c[0] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(v_src_c[1], src_c[0], border_value);
        v_src_n[0] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(v_src_n[1], src_n[0], border_value);

        PyrUp5x5VCore(v_src_p[0], v_src_c[0], v_src_n[0], v_esum[0], v_esum[1], v_osum[0], v_osum[1], kernel);
        PyrUp5x5VCore(v_src_p[1], v_src_c[1], v_src_n[1], v_esum[2], v_esum[3], v_osum[2], v_osum[3], kernel);
        PyrUp5x5VCore(v_src_p[2], v_src_c[2], v_src_n[2], v_esum[4], v_esum[5], v_osum[4], v_osum[5], kernel);

        PyrUp5x5HCore(v_esum[1], v_osum[1], v_esum[2], v_esum[3], v_osum[2], v_osum[3], v_esum[4], v_esum[5],
                      v_osum[4], v_osum[5], mv_result[0], mv_result[1], kernel);

        neon::vstore(dst_c0, mv_result[0].val[0]); neon::vstore(dst_c0 + 8, mv_result[0].val[1]);
        neon::vstore(dst_c1, mv_result[1].val[0]); neon::vstore(dst_c1 + 8, mv_result[1].val[1]);
    }

    // middle
    for (MI_S32 x = VOFFSET; x < (width_align - VOFFSET); x += VOFFSET)
    {
        MI_S32 dx = x << 1;
        neon::vload(src_p + x + VOFFSET, v_src_p[2]);
        neon::vload(src_c + x + VOFFSET, v_src_c[2]);
        neon::vload(src_n + x + VOFFSET, v_src_n[2]);

        PyrUp5x5VCore(v_src_p[2], v_src_c[2], v_src_n[2], v_esum[4], v_esum[5], v_osum[4], v_osum[5], kernel);

        PyrUp5x5HCore(v_esum[1], v_osum[1], v_esum[2], v_esum[3], v_osum[2], v_osum[3], v_esum[4], v_esum[5],
                      v_osum[4], v_osum[5], mv_result[0], mv_result[1], kernel);

        neon::vstore(dst_c0 + dx, mv_result[0].val[0]); neon::vstore(dst_c0 + dx + 8, mv_result[0].val[1]);
        neon::vstore(dst_c1 + dx, mv_result[1].val[0]); neon::vstore(dst_c1 + dx + 8, mv_result[1].val[1]);
    }

    // back
    {
        if (width_align != width)
        {
            MI_S32 x = width - (VOFFSET << 1);
            MI_S32 dx = x << 1;

            neon::vload(src_p + x - VOFFSET, v_src_p[0]);
            neon::vload(src_p + x,           v_src_p[1]);
            neon::vload(src_p + x + VOFFSET, v_src_p[2]);
            neon::vload(src_c + x - VOFFSET, v_src_c[0]);
            neon::vload(src_c + x,           v_src_c[1]);
            neon::vload(src_c + x + VOFFSET, v_src_c[2]);
            neon::vload(src_n + x - VOFFSET, v_src_n[0]);
            neon::vload(src_n + x,           v_src_n[1]);
            neon::vload(src_n + x + VOFFSET, v_src_n[2]);

            PyrUp5x5VCore(v_src_p[0], v_src_c[0], v_src_n[0], v_esum[0], v_esum[1], v_osum[0], v_osum[1], kernel);
            PyrUp5x5VCore(v_src_p[1], v_src_c[1], v_src_n[1], v_esum[2], v_esum[3], v_osum[2], v_osum[3], kernel);
            PyrUp5x5VCore(v_src_p[2], v_src_c[2], v_src_n[2], v_esum[4], v_esum[5], v_osum[4], v_osum[5], kernel);

            PyrUp5x5HCore(v_esum[1], v_osum[1], v_esum[2], v_esum[3], v_osum[2], v_osum[3], v_esum[4], v_esum[5],
                          v_osum[4], v_osum[5], mv_result[0], mv_result[1], kernel);

            neon::vstore(dst_c0 + dx, mv_result[0].val[0]); neon::vstore(dst_c0 + dx + 8, mv_result[0].val[1]);
            neon::vstore(dst_c1 + dx, mv_result[1].val[0]); neon::vstore(dst_c1 + dx + 8, mv_result[1].val[1]);
        }
    }

    // right
    {
        MI_S32 x    = width - VOFFSET;
        MI_S32 dx   = x << 1;
        MI_S32 last = width - 1;

        v_src_p[2] = GetBorderVector<BorderType::REPLICATE, BorderArea::RIGHT>(v_src_p[2], src_p[last], border_value);
        v_src_c[2] = GetBorderVector<BorderType::REPLICATE, BorderArea::RIGHT>(v_src_c[2], src_c[last], border_value);
        v_src_n[2] = GetBorderVector<BorderType::REPLICATE, BorderArea::RIGHT>(v_src_n[2], src_n[last], border_value);

        PyrUp5x5VCore(v_src_p[2], v_src_c[2], v_src_n[2], v_esum[4], v_esum[5], v_osum[4], v_osum[5], kernel);

        PyrUp5x5HCore(v_esum[1], v_osum[1], v_esum[2], v_esum[3], v_osum[2], v_osum[3], v_esum[4], v_esum[5],
                      v_osum[4], v_osum[5], mv_result[0], mv_result[1], kernel);

        neon::vstore(dst_c0 + dx, mv_result[0].val[0]); neon::vstore(dst_c0 + dx + 8, mv_result[0].val[1]);
        neon::vstore(dst_c1 + dx, mv_result[1].val[0]); neon::vstore(dst_c1 + dx + 8, mv_result[1].val[1]);
    }
}

template <typename Tp, BorderType BORDER_TYPE>
static Status PyrUp5x5NeonImpl(Context *ctx, const Mat &src, Mat &dst, const Mat &kmat,
                               MI_S32 start_row, MI_S32 end_row)
{
    AURA_UNUSED(ctx);

    using Kt = typename PyrUpTraits<Tp>::KernelType;

    MI_S32 width = src.GetSizes().m_width;

    const Tp *src_p = src.Ptr<Tp, BORDER_TYPE>(start_row - 1, MI_NULL);
    const Tp *src_c = src.Ptr<Tp>(start_row);
    const Tp *src_n = src.Ptr<Tp, BorderType::REPLICATE>(start_row + 1, MI_NULL);

    MI_S32 dy = 0;
    const Kt *kernel = kmat.Ptr<Kt>(0);

    // cal row
    for (MI_S32 sy = start_row; sy < end_row; sy++)
    {
        dy = sy << 1;

        Tp *dst_c0 = dst.Ptr<Tp>(dy);
        Tp *dst_c1 = dst.Ptr<Tp>(dy + 1);

        PyrUp5x5TwoRow<Tp, BORDER_TYPE, Kt>(src_p, src_c, src_n, dst_c0, dst_c1, width, kernel);

        src_p = src_c;
        src_c = src_n;
        src_n = src.Ptr<Tp, BorderType::REPLICATE>(sy + 2, MI_NULL);
    }

    return Status::OK;
}

template <typename Tp>
static Status PyrUp5x5NeonHelper(Context *ctx, const Mat &src, Mat &dst, const Mat &kmat,
                                 BorderType &border_type, const OpTarget &target)
{
    AURA_UNUSED(target);
    Status ret = Status::ERROR;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
        return ret;
    }

    MI_S32 height = src.GetSizes().m_height;

    switch (border_type)
    {
        case BorderType::REPLICATE:
        {
            ret = wp->ParallelFor(0, height, PyrUp5x5NeonImpl<Tp, BorderType::REPLICATE>,
                                  ctx, std::cref(src), std::ref(dst), std::cref(kmat));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "PyrUp5x5NeonImpl<BorderType::REPLICATE, Tp> failed");
            }
            break;
        }

        case BorderType::REFLECT_101:
        {
            ret = wp->ParallelFor(0, height, PyrUp5x5NeonImpl<Tp, BorderType::REFLECT_101>,
                                  ctx, std::cref(src), std::ref(dst), std::cref(kmat));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "PyrUp5x5NeonImpl<BorderType::REFLECT_101, Tp> failed");
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

Status PyrUp5x5Neon(Context *ctx, const Mat &src, Mat &dst, const Mat &kmat, BorderType &border_type, const OpTarget &target)
{
    Status ret = Status::ERROR;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = PyrUp5x5NeonHelper<MI_U8>(ctx, src, dst, kmat, border_type, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "PyrUp5x5NeonHelper<MI_U8> failed");
            }
            break;
        }

        case ElemType::U16:
        {
            ret = PyrUp5x5NeonHelper<MI_U16>(ctx, src, dst, kmat, border_type, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "PyrUp5x5NeonHelper<MI_U16> failed");
            }
            break;
        }

        case ElemType::S16:
        {
            ret = PyrUp5x5NeonHelper<MI_S16>(ctx, src, dst, kmat, border_type, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "PyrUp5x5NeonHelper<MI_S16> failed");
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

} //namespace aura