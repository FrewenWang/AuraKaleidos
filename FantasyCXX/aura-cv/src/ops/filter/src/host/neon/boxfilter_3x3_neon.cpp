#include "boxfilter_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

template <typename D8, typename d8x16_t = typename neon::QVector<D8>::VType,
                       typename d16x8_t = typename neon::QVector<typename Promote<D8>::Type>::VType,
                       typename std::enable_if<std::is_same<D8, DT_U8>::value || std::is_same<D8, DT_S8>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE d8x16_t BoxFilter3x3Core(d8x16_t &vq8_src_px0, d8x16_t &vq8_src_px1, d8x16_t &vq8_src_px2,
                                            d8x16_t &vq8_src_cx0, d8x16_t &vq8_src_cx1, d8x16_t &vq8_src_cx2,
                                            d8x16_t &vq8_src_nx0, d8x16_t &vq8_src_nx1, d8x16_t &vq8_src_nx2)
{
    d8x16_t vq8_src_pl0 = neon::vext<15>(vq8_src_px0, vq8_src_px1);
    d8x16_t vq8_src_pr0 = neon::vext<1>(vq8_src_px1, vq8_src_px2);

    d8x16_t vq8_src_cl0 = neon::vext<15>(vq8_src_cx0, vq8_src_cx1);
    d8x16_t vq8_src_cr0 = neon::vext<1>(vq8_src_cx1, vq8_src_cx2);

    d8x16_t vq8_src_nl0 = neon::vext<15>(vq8_src_nx0, vq8_src_nx1);
    d8x16_t vq8_src_nr0 = neon::vext<1>(vq8_src_nx1, vq8_src_nx2);

    d16x8_t vq16_sum_p_lo = neon::vaddw(neon::vaddl(neon::vgetlow(vq8_src_pl0),  neon::vgetlow(vq8_src_px1)),  neon::vgetlow(vq8_src_pr0));
    d16x8_t vq16_sum_p_hi = neon::vaddw(neon::vaddl(neon::vgethigh(vq8_src_pl0), neon::vgethigh(vq8_src_px1)), neon::vgethigh(vq8_src_pr0));

    d16x8_t vq16_sum_c_lo = neon::vaddw(neon::vaddl(neon::vgetlow(vq8_src_cl0),  neon::vgetlow(vq8_src_cx1)),  neon::vgetlow(vq8_src_cr0));
    d16x8_t vq16_sum_c_hi = neon::vaddw(neon::vaddl(neon::vgethigh(vq8_src_cl0), neon::vgethigh(vq8_src_cx1)), neon::vgethigh(vq8_src_cr0));

    d16x8_t vq16_sum_n_lo = neon::vaddw(neon::vaddl(neon::vgetlow(vq8_src_nl0),  neon::vgetlow(vq8_src_nx1)),  neon::vgetlow(vq8_src_nr0));
    d16x8_t vq16_sum_n_hi = neon::vaddw(neon::vaddl(neon::vgethigh(vq8_src_nl0), neon::vgethigh(vq8_src_nx1)), neon::vgethigh(vq8_src_nr0));

    d16x8_t vq16_sum_lo   = neon::vadd(neon::vadd(vq16_sum_p_lo, vq16_sum_c_lo), vq16_sum_n_lo);
    d16x8_t vq16_sum_hi   = neon::vadd(neon::vadd(vq16_sum_p_hi, vq16_sum_c_hi), vq16_sum_n_hi);

    d8x16_t vq8_result = neon::vcombine(neon::vqdivn_n<9>(vq16_sum_lo), neon::vqdivn_n<9>(vq16_sum_hi));

    vq8_src_px0 = vq8_src_px1;
    vq8_src_cx0 = vq8_src_cx1;
    vq8_src_nx0 = vq8_src_nx1;

    vq8_src_px1 = vq8_src_px2;
    vq8_src_cx1 = vq8_src_cx2;
    vq8_src_nx1 = vq8_src_nx2;

    return vq8_result;
}

template <typename D8, typename d8x16_t = typename neon::QVector<D8>::VType,
                       typename d16x8_t = typename neon::QVector<typename Promote<D8>::Type>::VType,
                       typename std::enable_if<std::is_same<D8, DT_U8>::value || std::is_same<D8, DT_S8>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID BoxFilter3x3Core(d8x16_t &vq8_src_p0x0, d8x16_t &vq8_src_p0x1, d8x16_t &vq8_src_p0x2,
                                            d8x16_t &vq8_src_c0x0, d8x16_t &vq8_src_c0x1, d8x16_t &vq8_src_c0x2,
                                            d8x16_t &vq8_src_c1x0, d8x16_t &vq8_src_c1x1, d8x16_t &vq8_src_c1x2,
                                            d8x16_t &vq8_src_n0x0, d8x16_t &vq8_src_n0x1, d8x16_t &vq8_src_n0x2,
                                            d8x16_t &vq8_result0,  d8x16_t &vq8_result1)
{
    d8x16_t vq8_src_p0l0 = neon::vext<15>(vq8_src_p0x0, vq8_src_p0x1);
    d8x16_t vq8_src_p0r0 = neon::vext<1>(vq8_src_p0x1, vq8_src_p0x2);

    d8x16_t vq8_src_c0l0 = neon::vext<15>(vq8_src_c0x0, vq8_src_c0x1);
    d8x16_t vq8_src_c0r0 = neon::vext<1>(vq8_src_c0x1, vq8_src_c0x2);

    d8x16_t vq8_src_c1l0 = neon::vext<15>(vq8_src_c1x0, vq8_src_c1x1);
    d8x16_t vq8_src_c1r0 = neon::vext<1>(vq8_src_c1x1, vq8_src_c1x2);

    d8x16_t vq8_src_n0l0 = neon::vext<15>(vq8_src_n0x0, vq8_src_n0x1);
    d8x16_t vq8_src_n0r0 = neon::vext<1>(vq8_src_n0x1, vq8_src_n0x2);

    d16x8_t vq16_sum_p0_lo  = neon::vaddw(neon::vaddl(neon::vgetlow(vq8_src_p0l0),  neon::vgetlow(vq8_src_p0x1)),  neon::vgetlow(vq8_src_p0r0));
    d16x8_t vq16_sum_p0_hi  = neon::vaddw(neon::vaddl(neon::vgethigh(vq8_src_p0l0), neon::vgethigh(vq8_src_p0x1)), neon::vgethigh(vq8_src_p0r0));
    d16x8_t vq16_sum_c0_lo  = neon::vaddw(neon::vaddl(neon::vgetlow(vq8_src_c0l0),  neon::vgetlow(vq8_src_c0x1)),  neon::vgetlow(vq8_src_c0r0));
    d16x8_t vq16_sum_c0_hi  = neon::vaddw(neon::vaddl(neon::vgethigh(vq8_src_c0l0), neon::vgethigh(vq8_src_c0x1)), neon::vgethigh(vq8_src_c0r0));
    d16x8_t vq16_sum_c1_lo  = neon::vaddw(neon::vaddl(neon::vgetlow(vq8_src_c1l0),  neon::vgetlow(vq8_src_c1x1)),  neon::vgetlow(vq8_src_c1r0));
    d16x8_t vq16_sum_c1_hi  = neon::vaddw(neon::vaddl(neon::vgethigh(vq8_src_c1l0), neon::vgethigh(vq8_src_c1x1)), neon::vgethigh(vq8_src_c1r0));
    d16x8_t vq16_sum_n0_lo  = neon::vaddw(neon::vaddl(neon::vgetlow(vq8_src_n0l0),  neon::vgetlow(vq8_src_n0x1)),  neon::vgetlow(vq8_src_n0r0));
    d16x8_t vq16_sum_n0_hi  = neon::vaddw(neon::vaddl(neon::vgethigh(vq8_src_n0l0), neon::vgethigh(vq8_src_n0x1)), neon::vgethigh(vq8_src_n0r0));

    d16x8_t vq16_sum_c_lo   = neon::vadd(vq16_sum_c0_lo, vq16_sum_c1_lo);
    d16x8_t vq16_sum_c_hi   = neon::vadd(vq16_sum_c0_hi, vq16_sum_c1_hi);

    d16x8_t vq16_sum_cp0_lo = neon::vadd(vq16_sum_c_lo, vq16_sum_p0_lo);
    d16x8_t vq16_sum_cp0_hi = neon::vadd(vq16_sum_c_hi, vq16_sum_p0_hi);
    vq8_result0             = neon::vcombine(neon::vqdivn_n<9>(vq16_sum_cp0_lo), neon::vqdivn_n<9>(vq16_sum_cp0_hi));
    d16x8_t vq16_sum_cn0_lo = neon::vadd(vq16_sum_c_lo, vq16_sum_n0_lo);
    d16x8_t vq16_sum_cn0_hi = neon::vadd(vq16_sum_c_hi, vq16_sum_n0_hi);
    vq8_result1             = neon::vcombine(neon::vqdivn_n<9>(vq16_sum_cn0_lo), neon::vqdivn_n<9>(vq16_sum_cn0_hi));

    vq8_src_p0x0 = vq8_src_p0x1;
    vq8_src_c0x0 = vq8_src_c0x1;
    vq8_src_c1x0 = vq8_src_c1x1;
    vq8_src_n0x0 = vq8_src_n0x1;

    vq8_src_p0x1 = vq8_src_p0x2;
    vq8_src_c0x1 = vq8_src_c0x2;
    vq8_src_c1x1 = vq8_src_c1x2;
    vq8_src_n0x1 = vq8_src_n0x2;
}

template <typename D16, typename d16x8_t = typename neon::QVector<D16>::VType,
                        typename d32x4_t = typename neon::QVector<typename Promote<D16>::Type>::VType,
                        typename std::enable_if<std::is_same<D16, DT_U16>::value || std::is_same<D16, DT_S16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE d16x8_t BoxFilter3x3Core(d16x8_t &vq16_src_px0, d16x8_t &vq16_src_px1, d16x8_t &vq16_src_px2,
                                            d16x8_t &vq16_src_cx0, d16x8_t &vq16_src_cx1, d16x8_t &vq16_src_cx2,
                                            d16x8_t &vq16_src_nx0, d16x8_t &vq16_src_nx1, d16x8_t &vq16_src_nx2)
{
    d16x8_t vq16_src_pl0 = neon::vext<7>(vq16_src_px0, vq16_src_px1);
    d16x8_t vq16_src_pr0 = neon::vext<1>(vq16_src_px1, vq16_src_px2);

    d16x8_t vq16_src_cl0 = neon::vext<7>(vq16_src_cx0, vq16_src_cx1);
    d16x8_t vq16_src_cr0 = neon::vext<1>(vq16_src_cx1, vq16_src_cx2);

    d16x8_t vq16_src_nl0 = neon::vext<7>(vq16_src_nx0, vq16_src_nx1);
    d16x8_t vq16_src_nr0 = neon::vext<1>(vq16_src_nx1, vq16_src_nx2);

    d32x4_t vq32_sum_p_lo = neon::vaddw(neon::vaddl(neon::vgetlow(vq16_src_pl0),  neon::vgetlow(vq16_src_px1)),  neon::vgetlow(vq16_src_pr0));
    d32x4_t vq32_sum_p_hi = neon::vaddw(neon::vaddl(neon::vgethigh(vq16_src_pl0), neon::vgethigh(vq16_src_px1)), neon::vgethigh(vq16_src_pr0));

    d32x4_t vq32_sum_c_lo = neon::vaddw(neon::vaddl(neon::vgetlow(vq16_src_cl0),  neon::vgetlow(vq16_src_cx1)),  neon::vgetlow(vq16_src_cr0));
    d32x4_t vq32_sum_c_hi = neon::vaddw(neon::vaddl(neon::vgethigh(vq16_src_cl0), neon::vgethigh(vq16_src_cx1)), neon::vgethigh(vq16_src_cr0));

    d32x4_t vq32_sum_n_lo = neon::vaddw(neon::vaddl(neon::vgetlow(vq16_src_nl0),  neon::vgetlow(vq16_src_nx1)),  neon::vgetlow(vq16_src_nr0));
    d32x4_t vq32_sum_n_hi = neon::vaddw(neon::vaddl(neon::vgethigh(vq16_src_nl0), neon::vgethigh(vq16_src_nx1)), neon::vgethigh(vq16_src_nr0));

    d32x4_t vq32_sum_lo   = neon::vadd(neon::vadd(vq32_sum_p_lo, vq32_sum_c_lo), vq32_sum_n_lo);
    d32x4_t vq32_sum_hi   = neon::vadd(neon::vadd(vq32_sum_p_hi, vq32_sum_c_hi), vq32_sum_n_hi);

    d16x8_t vq16_result   = neon::vcombine(neon::vqdivn_n<9>(vq32_sum_lo), neon::vqdivn_n<9>(vq32_sum_hi));

    vq16_src_px0 = vq16_src_px1;
    vq16_src_cx0 = vq16_src_cx1;
    vq16_src_nx0 = vq16_src_nx1;

    vq16_src_px1 = vq16_src_px2;
    vq16_src_cx1 = vq16_src_cx2;
    vq16_src_nx1 = vq16_src_nx2;

    return vq16_result;
}

template <typename D16, typename d16x8_t = typename neon::QVector<D16>::VType,
                        typename d32x4_t = typename neon::QVector<typename Promote<D16>::Type>::VType,
                        typename std::enable_if<std::is_same<D16, DT_U16>::value || std::is_same<D16, DT_S16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID BoxFilter3x3Core(d16x8_t &vq16_src_p0x0, d16x8_t &vq16_src_p0x1, d16x8_t &vq16_src_p0x2,
                                            d16x8_t &vq16_src_c0x0, d16x8_t &vq16_src_c0x1, d16x8_t &vq16_src_c0x2,
                                            d16x8_t &vq16_src_c1x0, d16x8_t &vq16_src_c1x1, d16x8_t &vq16_src_c1x2,
                                            d16x8_t &vq16_src_n0x0, d16x8_t &vq16_src_n0x1, d16x8_t &vq16_src_n0x2,
                                            d16x8_t &vq16_result0,  d16x8_t &vq16_result1)
{
    d16x8_t vq16_src_p0l0 = neon::vext<7>(vq16_src_p0x0, vq16_src_p0x1);
    d16x8_t vq16_src_p0r0 = neon::vext<1>(vq16_src_p0x1, vq16_src_p0x2);

    d16x8_t vq16_src_c0l0 = neon::vext<7>(vq16_src_c0x0, vq16_src_c0x1);
    d16x8_t vq16_src_c0r0 = neon::vext<1>(vq16_src_c0x1, vq16_src_c0x2);

    d16x8_t vq16_src_c1l0 = neon::vext<7>(vq16_src_c1x0, vq16_src_c1x1);
    d16x8_t vq16_src_c1r0 = neon::vext<1>(vq16_src_c1x1, vq16_src_c1x2);

    d16x8_t vq16_src_n0l0 = neon::vext<7>(vq16_src_n0x0, vq16_src_n0x1);
    d16x8_t vq16_src_n0r0 = neon::vext<1>(vq16_src_n0x1, vq16_src_n0x2);

    d32x4_t vq32_sum_p0_lo  = neon::vaddw(neon::vaddl(neon::vgetlow(vq16_src_p0l0),  neon::vgetlow(vq16_src_p0x1)),  neon::vgetlow(vq16_src_p0r0));
    d32x4_t vq32_sum_p0_hi  = neon::vaddw(neon::vaddl(neon::vgethigh(vq16_src_p0l0), neon::vgethigh(vq16_src_p0x1)), neon::vgethigh(vq16_src_p0r0));
    d32x4_t vq32_sum_c0_lo  = neon::vaddw(neon::vaddl(neon::vgetlow(vq16_src_c0l0),  neon::vgetlow(vq16_src_c0x1)),  neon::vgetlow(vq16_src_c0r0));
    d32x4_t vq32_sum_c0_hi  = neon::vaddw(neon::vaddl(neon::vgethigh(vq16_src_c0l0), neon::vgethigh(vq16_src_c0x1)), neon::vgethigh(vq16_src_c0r0));
    d32x4_t vq32_sum_c1_lo  = neon::vaddw(neon::vaddl(neon::vgetlow(vq16_src_c1l0),  neon::vgetlow(vq16_src_c1x1)),  neon::vgetlow(vq16_src_c1r0));
    d32x4_t vq32_sum_c1_hi  = neon::vaddw(neon::vaddl(neon::vgethigh(vq16_src_c1l0), neon::vgethigh(vq16_src_c1x1)), neon::vgethigh(vq16_src_c1r0));
    d32x4_t vq32_sum_n0_lo  = neon::vaddw(neon::vaddl(neon::vgetlow(vq16_src_n0l0),  neon::vgetlow(vq16_src_n0x1)),  neon::vgetlow(vq16_src_n0r0));
    d32x4_t vq32_sum_n0_hi  = neon::vaddw(neon::vaddl(neon::vgethigh(vq16_src_n0l0), neon::vgethigh(vq16_src_n0x1)), neon::vgethigh(vq16_src_n0r0));

    d32x4_t vq32_sum_c_lo   = neon::vadd(vq32_sum_c0_lo, vq32_sum_c1_lo);
    d32x4_t vq32_sum_c_hi   = neon::vadd(vq32_sum_c0_hi, vq32_sum_c1_hi);

    d32x4_t vq32_sum_p0c_lo = neon::vadd(vq32_sum_c_lo, vq32_sum_p0_lo);
    d32x4_t vq32_sum_p0c_hi = neon::vadd(vq32_sum_c_hi, vq32_sum_p0_hi);
    vq16_result0            = neon::vcombine(neon::vqdivn_n<9>(vq32_sum_p0c_lo), neon::vqdivn_n<9>(vq32_sum_p0c_hi));
    d32x4_t vq32_sum_cn0_lo = neon::vadd(vq32_sum_c_lo, vq32_sum_n0_lo);
    d32x4_t vq32_sum_cn0_hi = neon::vadd(vq32_sum_c_hi, vq32_sum_n0_hi);
    vq16_result1            = neon::vcombine(neon::vqdivn_n<9>(vq32_sum_cn0_lo), neon::vqdivn_n<9>(vq32_sum_cn0_hi));

    vq16_src_p0x0 = vq16_src_p0x1;
    vq16_src_c0x0 = vq16_src_c0x1;
    vq16_src_c1x0 = vq16_src_c1x1;
    vq16_src_n0x0 = vq16_src_n0x1;

    vq16_src_p0x1 = vq16_src_p0x2;
    vq16_src_c0x1 = vq16_src_c0x2;
    vq16_src_c1x1 = vq16_src_c1x2;
    vq16_src_n0x1 = vq16_src_n0x2;
}

#if defined(AURA_ENABLE_NEON_FP16)
template <typename Tp, typename std::enable_if<std::is_same<Tp, MI_F16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE float16x8_t BoxFilter3x3Core(float16x8_t &vqf16_src_px0, float16x8_t &vqf16_src_px1, float16x8_t &vqf16_src_px2,
                                                float16x8_t &vqf16_src_cx0, float16x8_t &vqf16_src_cx1, float16x8_t &vqf16_src_cx2,
                                                float16x8_t &vqf16_src_nx0, float16x8_t &vqf16_src_nx1, float16x8_t &vqf16_src_nx2)
{
    float16x8_t vqf16_src_pl0  = neon::vext<7>(vqf16_src_px0, vqf16_src_px1);
    float16x8_t vqf16_src_pr0  = neon::vext<1>(vqf16_src_px1, vqf16_src_px2);

    float16x8_t vqf16_src_cl0  = neon::vext<7>(vqf16_src_cx0, vqf16_src_cx1);
    float16x8_t vqf16_src_cr0  = neon::vext<1>(vqf16_src_cx1, vqf16_src_cx2);

    float16x8_t vqf16_src_nl0  = neon::vext<7>(vqf16_src_nx0, vqf16_src_nx1);
    float16x8_t vqf16_src_nr0  = neon::vext<1>(vqf16_src_nx1, vqf16_src_nx2);

    float32x4_t vqf32_sum_p_lo = neon::vadd(neon::vadd(neon::vcvt<DT_F32>(neon::vgetlow(vqf16_src_pl0)),
                                                       neon::vcvt<DT_F32>(neon::vgetlow(vqf16_src_pr0))),
                                                       neon::vcvt<DT_F32>(neon::vgetlow(vqf16_src_px1)));
    float32x4_t vqf32_sum_c_lo = neon::vadd(neon::vadd(neon::vcvt<DT_F32>(neon::vgetlow(vqf16_src_cl0)),
                                                       neon::vcvt<DT_F32>(neon::vgetlow(vqf16_src_cr0))),
                                                       neon::vcvt<DT_F32>(neon::vgetlow(vqf16_src_cx1)));
    float32x4_t vqf32_sum_n_lo = neon::vadd(neon::vadd(neon::vcvt<DT_F32>(neon::vgetlow(vqf16_src_nl0)),
                                                       neon::vcvt<DT_F32>(neon::vgetlow(vqf16_src_nr0))),
                                                       neon::vcvt<DT_F32>(neon::vgetlow(vqf16_src_nx1)));

    float32x4_t vqf32_sum_p_hi = neon::vadd(neon::vadd(neon::vcvt<DT_F32>(neon::vgethigh(vqf16_src_pl0)),
                                                       neon::vcvt<DT_F32>(neon::vgethigh(vqf16_src_pr0))),
                                                       neon::vcvt<DT_F32>(neon::vgethigh(vqf16_src_px1)));
    float32x4_t vqf32_sum_c_hi = neon::vadd(neon::vadd(neon::vcvt<DT_F32>(neon::vgethigh(vqf16_src_cl0)),
                                                       neon::vcvt<DT_F32>(neon::vgethigh(vqf16_src_cr0))),
                                                       neon::vcvt<DT_F32>(neon::vgethigh(vqf16_src_cx1)));
    float32x4_t vqf32_sum_n_hi = neon::vadd(neon::vadd(neon::vcvt<DT_F32>(neon::vgethigh(vqf16_src_nl0)),
                                                       neon::vcvt<DT_F32>(neon::vgethigh(vqf16_src_nr0))),
                                                       neon::vcvt<DT_F32>(neon::vgethigh(vqf16_src_nx1)));

    float32x4_t vqf32_sum_lo    = neon::vadd(neon::vadd(vqf32_sum_p_lo, vqf32_sum_c_lo), vqf32_sum_n_lo);
    float32x4_t vqf32_result_lo = neon::vmul(vqf32_sum_lo, static_cast<DT_F32>(1.0 / 9));

    float32x4_t vqf32_sum_hi    = neon::vadd(neon::vadd(vqf32_sum_p_hi, vqf32_sum_c_hi), vqf32_sum_n_hi);
    float32x4_t vqf32_result_hi = neon::vmul(vqf32_sum_hi, static_cast<DT_F32>(1.0 / 9));

    float16x8_t vqf16_result    = neon::vcombine(neon::vcvt<MI_F16>(vqf32_result_lo), neon::vcvt<MI_F16>(vqf32_result_hi));

    vqf16_src_px0 = vqf16_src_px1;
    vqf16_src_cx0 = vqf16_src_cx1;
    vqf16_src_nx0 = vqf16_src_nx1;

    vqf16_src_px1 = vqf16_src_px2;
    vqf16_src_cx1 = vqf16_src_cx2;
    vqf16_src_nx1 = vqf16_src_nx2;

    return vqf16_result;
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, MI_F16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID BoxFilter3x3Core(float16x8_t &vqf16_src_p0x0, float16x8_t &vqf16_src_p0x1, float16x8_t &vqf16_src_p0x2,
                                            float16x8_t &vqf16_src_c0x0, float16x8_t &vqf16_src_c0x1, float16x8_t &vqf16_src_c0x2,
                                            float16x8_t &vqf16_src_c1x0, float16x8_t &vqf16_src_c1x1, float16x8_t &vqf16_src_c1x2,
                                            float16x8_t &vqf16_src_n0x0, float16x8_t &vqf16_src_n0x1, float16x8_t &vqf16_src_n0x2,
                                            float16x8_t &vqf16_result0,  float16x8_t &vqf16_result1)
{
    float16x8_t vqf16_src_p0l0  = neon::vext<7>(vqf16_src_p0x0, vqf16_src_p0x1);
    float16x8_t vqf16_src_p0r0  = neon::vext<1>(vqf16_src_p0x1, vqf16_src_p0x2);

    float16x8_t vqf16_src_c0l0  = neon::vext<7>(vqf16_src_c0x0, vqf16_src_c0x1);
    float16x8_t vqf16_src_c0r0  = neon::vext<1>(vqf16_src_c0x1, vqf16_src_c0x2);

    float16x8_t vqf16_src_c1l0  = neon::vext<7>(vqf16_src_c1x0, vqf16_src_c1x1);
    float16x8_t vqf16_src_c1r0  = neon::vext<1>(vqf16_src_c1x1, vqf16_src_c1x2);

    float16x8_t vqf16_src_n0l0  = neon::vext<7>(vqf16_src_n0x0, vqf16_src_n0x1);
    float16x8_t vqf16_src_n0r0  = neon::vext<1>(vqf16_src_n0x1, vqf16_src_n0x2);

    float32x4_t vqf32_sum_p0_lo = neon::vadd(neon::vadd(neon::vcvt<DT_F32>(neon::vgetlow(vqf16_src_p0l0)),
                                                        neon::vcvt<DT_F32>(neon::vgetlow(vqf16_src_p0r0))),
                                                        neon::vcvt<DT_F32>(neon::vgetlow(vqf16_src_p0x1)));
    float32x4_t vqf32_sum_c0_lo = neon::vadd(neon::vadd(neon::vcvt<DT_F32>(neon::vgetlow(vqf16_src_c0l0)),
                                                        neon::vcvt<DT_F32>(neon::vgetlow(vqf16_src_c0r0))),
                                                        neon::vcvt<DT_F32>(neon::vgetlow(vqf16_src_c0x1)));
    float32x4_t vqf32_sum_c1_lo = neon::vadd(neon::vadd(neon::vcvt<DT_F32>(neon::vgetlow(vqf16_src_c1l0)),
                                                        neon::vcvt<DT_F32>(neon::vgetlow(vqf16_src_c1r0))),
                                                        neon::vcvt<DT_F32>(neon::vgetlow(vqf16_src_c1x1)));
    float32x4_t vqf32_sum_n0_lo = neon::vadd(neon::vadd(neon::vcvt<DT_F32>(neon::vgetlow(vqf16_src_n0l0)),
                                                        neon::vcvt<DT_F32>(neon::vgetlow(vqf16_src_n0r0))),
                                                        neon::vcvt<DT_F32>(neon::vgetlow(vqf16_src_n0x1)));

    float32x4_t vqf32_sum_p0_hi = neon::vadd(neon::vadd(neon::vcvt<DT_F32>(neon::vgethigh(vqf16_src_p0l0)),
                                                        neon::vcvt<DT_F32>(neon::vgethigh(vqf16_src_p0r0))),
                                                        neon::vcvt<DT_F32>(neon::vgethigh(vqf16_src_p0x1)));
    float32x4_t vqf32_sum_c0_hi = neon::vadd(neon::vadd(neon::vcvt<DT_F32>(neon::vgethigh(vqf16_src_c0l0)),
                                                        neon::vcvt<DT_F32>(neon::vgethigh(vqf16_src_c0r0))),
                                                        neon::vcvt<DT_F32>(neon::vgethigh(vqf16_src_c0x1)));
    float32x4_t vqf32_sum_c1_hi = neon::vadd(neon::vadd(neon::vcvt<DT_F32>(neon::vgethigh(vqf16_src_c1l0)),
                                                        neon::vcvt<DT_F32>(neon::vgethigh(vqf16_src_c1r0))),
                                                        neon::vcvt<DT_F32>(neon::vgethigh(vqf16_src_c1x1)));
    float32x4_t vqf32_sum_n0_hi = neon::vadd(neon::vadd(neon::vcvt<DT_F32>(neon::vgethigh(vqf16_src_n0l0)),
                                                        neon::vcvt<DT_F32>(neon::vgethigh(vqf16_src_n0r0))),
                                                        neon::vcvt<DT_F32>(neon::vgethigh(vqf16_src_n0x1)));

    float32x4_t vqf32_sum_c_lo   = neon::vadd(vqf32_sum_c0_lo,  vqf32_sum_c1_lo);
    float32x4_t vqf32_sum_p0c_lo = neon::vadd(vqf32_sum_c_lo,   vqf32_sum_p0_lo);
    float32x4_t vqf32_result0_lo = neon::vmul(vqf32_sum_p0c_lo, static_cast<DT_F32>(1.0 / 9));
    float32x4_t vqf32_sum_cn0_lo = neon::vadd(vqf32_sum_c_lo,   vqf32_sum_n0_lo);
    float32x4_t vqf32_result1_lo = neon::vmul(vqf32_sum_cn0_lo, static_cast<DT_F32>(1.0 / 9));

    float32x4_t vqf32_sum_c_hi   = neon::vadd(vqf32_sum_c0_hi,  vqf32_sum_c1_hi);
    float32x4_t vqf32_sum_p0c_hi = neon::vadd(vqf32_sum_c_hi,   vqf32_sum_p0_hi);
    float32x4_t vqf32_result0_hi = neon::vmul(vqf32_sum_p0c_hi, static_cast<DT_F32>(1.0 / 9));
    float32x4_t vqf32_sum_cn0_hi = neon::vadd(vqf32_sum_c_hi,   vqf32_sum_n0_hi);
    float32x4_t vqf32_result1_hi = neon::vmul(vqf32_sum_cn0_hi, static_cast<DT_F32>(1.0 / 9));

    vqf16_result0 = neon::vcombine(neon::vcvt<MI_F16>(vqf32_result0_lo), neon::vcvt<MI_F16>(vqf32_result0_hi));
    vqf16_result1 = neon::vcombine(neon::vcvt<MI_F16>(vqf32_result1_lo), neon::vcvt<MI_F16>(vqf32_result1_hi));

    vqf16_src_p0x0 = vqf16_src_p0x1;
    vqf16_src_c0x0 = vqf16_src_c0x1;
    vqf16_src_c1x0 = vqf16_src_c1x1;
    vqf16_src_n0x0 = vqf16_src_n0x1;

    vqf16_src_p0x1 = vqf16_src_p0x2;
    vqf16_src_c0x1 = vqf16_src_c0x2;
    vqf16_src_c1x1 = vqf16_src_c1x2;
    vqf16_src_n0x1 = vqf16_src_n0x2;
}
#endif // AURA_ENABLE_NEON_FP16

template <typename Tp, typename std::enable_if<std::is_same<Tp, DT_F32>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE float32x4_t BoxFilter3x3Core(float32x4_t &vqf32_src_px0, float32x4_t &vqf32_src_px1, float32x4_t &vqf32_src_px2,
                                                float32x4_t &vqf32_src_cx0, float32x4_t &vqf32_src_cx1, float32x4_t &vqf32_src_cx2,
                                                float32x4_t &vqf32_src_nx0, float32x4_t &vqf32_src_nx1, float32x4_t &vqf32_src_nx2)
{
    float32x4_t vqf32_src_pl0 = neon::vext<3>(vqf32_src_px0, vqf32_src_px1);
    float32x4_t vqf32_src_pr0 = neon::vext<1>(vqf32_src_px1, vqf32_src_px2);

    float32x4_t vqf32_src_cl0 = neon::vext<3>(vqf32_src_cx0, vqf32_src_cx1);
    float32x4_t vqf32_src_cr0 = neon::vext<1>(vqf32_src_cx1, vqf32_src_cx2);

    float32x4_t vqf32_src_nl0 = neon::vext<3>(vqf32_src_nx0, vqf32_src_nx1);
    float32x4_t vqf32_src_nr0 = neon::vext<1>(vqf32_src_nx1, vqf32_src_nx2);

    float32x4_t vqf32_sum_p   = neon::vadd(neon::vadd(vqf32_src_pl0, vqf32_src_pr0), vqf32_src_px1);
    float32x4_t vqf32_sum_c   = neon::vadd(neon::vadd(vqf32_src_cl0, vqf32_src_cr0), vqf32_src_cx1);
    float32x4_t vqf32_sum_n   = neon::vadd(neon::vadd(vqf32_src_nl0, vqf32_src_nr0), vqf32_src_nx1);

    float32x4_t vqf32_sum     = neon::vadd(neon::vadd(vqf32_sum_p, vqf32_sum_c), vqf32_sum_n);
    float32x4_t vqf32_result  = neon::vmul(vqf32_sum, static_cast<DT_F32>(1.0 / 9));

    vqf32_src_px0 = vqf32_src_px1;
    vqf32_src_cx0 = vqf32_src_cx1;
    vqf32_src_nx0 = vqf32_src_nx1;

    vqf32_src_px1 = vqf32_src_px2;
    vqf32_src_cx1 = vqf32_src_cx2;
    vqf32_src_nx1 = vqf32_src_nx2;

    return vqf32_result;
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, DT_F32>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID BoxFilter3x3Core(float32x4_t &vqf32_src_p0x0, float32x4_t &vqf32_src_p0x1, float32x4_t &vqf32_src_p0x2,
                                            float32x4_t &vqf32_src_c0x0, float32x4_t &vqf32_src_c0x1, float32x4_t &vqf32_src_c0x2,
                                            float32x4_t &vqf32_src_c1x0, float32x4_t &vqf32_src_c1x1, float32x4_t &vqf32_src_c1x2,
                                            float32x4_t &vqf32_src_n0x0, float32x4_t &vqf32_src_n0x1, float32x4_t &vqf32_src_n0x2,
                                            float32x4_t &vqf32_result0,  float32x4_t &vqf32_result1)
{
    float32x4_t vqf32_src_p0l0 = neon::vext<3>(vqf32_src_p0x0, vqf32_src_p0x1);
    float32x4_t vqf32_src_p0r0 = neon::vext<1>(vqf32_src_p0x1, vqf32_src_p0x2);

    float32x4_t vqf32_src_c0l0 = neon::vext<3>(vqf32_src_c0x0, vqf32_src_c0x1);
    float32x4_t vqf32_src_c0r0 = neon::vext<1>(vqf32_src_c0x1, vqf32_src_c0x2);

    float32x4_t vqf32_src_c1l0 = neon::vext<3>(vqf32_src_c1x0, vqf32_src_c1x1);
    float32x4_t vqf32_src_c1r0 = neon::vext<1>(vqf32_src_c1x1, vqf32_src_c1x2);

    float32x4_t vqf32_src_n0l0 = neon::vext<3>(vqf32_src_n0x0, vqf32_src_n0x1);
    float32x4_t vqf32_src_n0r0 = neon::vext<1>(vqf32_src_n0x1, vqf32_src_n0x2);

    float32x4_t vqf32_sum_p0   = neon::vadd(neon::vadd(vqf32_src_p0l0, vqf32_src_p0r0), vqf32_src_p0x1);
    float32x4_t vqf32_sum_c0   = neon::vadd(neon::vadd(vqf32_src_c0l0, vqf32_src_c0r0), vqf32_src_c0x1);
    float32x4_t vqf32_sum_c1   = neon::vadd(neon::vadd(vqf32_src_c1l0, vqf32_src_c1r0), vqf32_src_c1x1);
    float32x4_t vqf32_sum_n0   = neon::vadd(neon::vadd(vqf32_src_n0l0, vqf32_src_n0r0), vqf32_src_n0x1);

    float32x4_t vqf32_sum_c    = neon::vadd(vqf32_sum_c0,  vqf32_sum_c1);
    float32x4_t vqf32_sum_p0c  = neon::vadd(vqf32_sum_c,   vqf32_sum_p0);
    vqf32_result0              = neon::vmul(vqf32_sum_p0c, static_cast<DT_F32>(1.0 / 9));
    float32x4_t vqf32_sum_cn0  = neon::vadd(vqf32_sum_c,   vqf32_sum_n0);
    vqf32_result1              = neon::vmul(vqf32_sum_cn0, static_cast<DT_F32>(1.0 / 9));

    vqf32_src_p0x0 = vqf32_src_p0x1;
    vqf32_src_c0x0 = vqf32_src_c0x1;
    vqf32_src_c1x0 = vqf32_src_c1x1;
    vqf32_src_n0x0 = vqf32_src_n0x1;

    vqf32_src_p0x1 = vqf32_src_p0x2;
    vqf32_src_c0x1 = vqf32_src_c0x2;
    vqf32_src_c1x1 = vqf32_src_c1x2;
    vqf32_src_n0x1 = vqf32_src_n0x2;
}

template <typename Tp, BorderType BORDER_TYPE, DT_S32 C>
static DT_VOID BoxFilter3x3OneRow(const Tp *src_p, const Tp *src_c, const Tp *src_n, Tp *dst_c,
                                  DT_S32 width, const std::vector<Tp> &border_value)
{
    using MVqType = typename neon::MQVector<Tp, C>::MVType;

    constexpr DT_S32 ELEM_COUNTS = 16 / sizeof(Tp);
    constexpr DT_S32 VOFFSET     = ELEM_COUNTS * C;
    const DT_S32 width_align     = (width & -ELEM_COUNTS) * C;

    MVqType mvq_src_p[3], mvq_src_c[3], mvq_src_n[3];
    MVqType mvq_result;

    // left border
    {
        neon::vload(src_p,           mvq_src_p[1]);
        neon::vload(src_c,           mvq_src_c[1]);
        neon::vload(src_n,           mvq_src_n[1]);
        neon::vload(src_p + VOFFSET, mvq_src_p[2]);
        neon::vload(src_c + VOFFSET, mvq_src_c[2]);
        neon::vload(src_n + VOFFSET, mvq_src_n[2]);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            mvq_src_p[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mvq_src_p[1].val[ch], src_p[ch], border_value[ch]);
            mvq_src_c[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mvq_src_c[1].val[ch], src_c[ch], border_value[ch]);
            mvq_src_n[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mvq_src_n[1].val[ch], src_n[ch], border_value[ch]);

            mvq_result.val[ch] = BoxFilter3x3Core<Tp>(mvq_src_p[0].val[ch], mvq_src_p[1].val[ch], mvq_src_p[2].val[ch],
                                                      mvq_src_c[0].val[ch], mvq_src_c[1].val[ch], mvq_src_c[2].val[ch],
                                                      mvq_src_n[0].val[ch], mvq_src_n[1].val[ch], mvq_src_n[2].val[ch]);
        }

        neon::vstore(dst_c, mvq_result);
    }

    // middle
    {
        for (DT_S32 x = VOFFSET; x < width_align - VOFFSET; x += VOFFSET)
        {
            neon::vload(src_p + x + VOFFSET, mvq_src_p[2]);
            neon::vload(src_c + x + VOFFSET, mvq_src_c[2]);
            neon::vload(src_n + x + VOFFSET, mvq_src_n[2]);

            for (DT_S32 ch = 0; ch < C; ch++)
            {
                mvq_result.val[ch] = BoxFilter3x3Core<Tp>(mvq_src_p[0].val[ch], mvq_src_p[1].val[ch], mvq_src_p[2].val[ch],
                                                          mvq_src_c[0].val[ch], mvq_src_c[1].val[ch], mvq_src_c[2].val[ch],
                                                          mvq_src_n[0].val[ch], mvq_src_n[1].val[ch], mvq_src_n[2].val[ch]);
            }

            neon::vstore(dst_c + x, mvq_result);
        }
    }

    // back
    {
        if (width_align != width * C)
        {
            DT_S32 x = (width - (ELEM_COUNTS << 1)) * C;
            neon::vload(src_p + x - VOFFSET, mvq_src_p[0]);
            neon::vload(src_c + x - VOFFSET, mvq_src_c[0]);
            neon::vload(src_n + x - VOFFSET, mvq_src_n[0]);
            neon::vload(src_p + x,           mvq_src_p[1]);
            neon::vload(src_c + x,           mvq_src_c[1]);
            neon::vload(src_n + x,           mvq_src_n[1]);
            neon::vload(src_p + x + VOFFSET, mvq_src_p[2]);
            neon::vload(src_c + x + VOFFSET, mvq_src_c[2]);
            neon::vload(src_n + x + VOFFSET, mvq_src_n[2]);

            for (DT_S32 ch = 0; ch < C; ch++)
            {
                mvq_result.val[ch] = BoxFilter3x3Core<Tp>(mvq_src_p[0].val[ch], mvq_src_p[1].val[ch], mvq_src_p[2].val[ch],
                                                          mvq_src_c[0].val[ch], mvq_src_c[1].val[ch], mvq_src_c[2].val[ch],
                                                          mvq_src_n[0].val[ch], mvq_src_n[1].val[ch], mvq_src_n[2].val[ch]);
            }

            neon::vstore(dst_c + x, mvq_result);
        }
    }

    // right border
    {
        DT_S32 x    = (width - ELEM_COUNTS) * C;
        DT_S32 last = (width - 1) * C;

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            mvq_src_p[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mvq_src_p[1].val[ch], src_p[last + ch], border_value[ch]);
            mvq_src_c[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mvq_src_c[1].val[ch], src_c[last + ch], border_value[ch]);
            mvq_src_n[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mvq_src_n[1].val[ch], src_n[last + ch], border_value[ch]);

            mvq_result.val[ch] = BoxFilter3x3Core<Tp>(mvq_src_p[0].val[ch], mvq_src_p[1].val[ch], mvq_src_p[2].val[ch],
                                                      mvq_src_c[0].val[ch], mvq_src_c[1].val[ch], mvq_src_c[2].val[ch],
                                                      mvq_src_n[0].val[ch], mvq_src_n[1].val[ch], mvq_src_n[2].val[ch]);
        }

        neon::vstore(dst_c + x, mvq_result);
    }
}

template <typename Tp, BorderType BORDER_TYPE, DT_S32 C>
static DT_VOID BoxFilter3x3TwoRow(const Tp *src_p0, const Tp *src_c0, const Tp *src_c1,
                                  const Tp *src_n0, Tp *dst_c0, Tp *dst_c1,
                                  DT_S32 width, const std::vector<Tp> &border_value)
{
    using MVqType = typename neon::MQVector<Tp, C>::MVType;

    constexpr DT_S32 ELEM_COUNTS = 16 / sizeof(Tp);
    constexpr DT_S32 VOFFSET     = ELEM_COUNTS * C;
    const DT_S32 width_align     = (width & -ELEM_COUNTS) * C;

    MVqType mvq_src_p0[3], mvq_src_c0[3], mvq_src_c1[3], mvq_src_n0[3];
    MVqType mvq_result0, mvq_result1;

    // left border
    {
        neon::vload(src_p0,           mvq_src_p0[1]);
        neon::vload(src_c0,           mvq_src_c0[1]);
        neon::vload(src_c1,           mvq_src_c1[1]);
        neon::vload(src_n0,           mvq_src_n0[1]);
        neon::vload(src_p0 + VOFFSET, mvq_src_p0[2]);
        neon::vload(src_c0 + VOFFSET, mvq_src_c0[2]);
        neon::vload(src_c1 + VOFFSET, mvq_src_c1[2]);
        neon::vload(src_n0 + VOFFSET, mvq_src_n0[2]);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            mvq_src_p0[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mvq_src_p0[1].val[ch], src_p0[ch], border_value[ch]);
            mvq_src_c0[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mvq_src_c0[1].val[ch], src_c0[ch], border_value[ch]);
            mvq_src_c1[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mvq_src_c1[1].val[ch], src_c1[ch], border_value[ch]);
            mvq_src_n0[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mvq_src_n0[1].val[ch], src_n0[ch], border_value[ch]);

            BoxFilter3x3Core<Tp>(mvq_src_p0[0].val[ch], mvq_src_p0[1].val[ch], mvq_src_p0[2].val[ch],
                                 mvq_src_c0[0].val[ch], mvq_src_c0[1].val[ch], mvq_src_c0[2].val[ch],
                                 mvq_src_c1[0].val[ch], mvq_src_c1[1].val[ch], mvq_src_c1[2].val[ch],
                                 mvq_src_n0[0].val[ch], mvq_src_n0[1].val[ch], mvq_src_n0[2].val[ch],
                                 mvq_result0.val[ch],   mvq_result1.val[ch]);
        }

        neon::vstore(dst_c0, mvq_result0);
        neon::vstore(dst_c1, mvq_result1);
    }

    // middle
    {
        for (DT_S32 x = VOFFSET; x < width_align - VOFFSET; x += VOFFSET)
        {
            neon::vload(src_p0 + x + VOFFSET, mvq_src_p0[2]);
            neon::vload(src_c0 + x + VOFFSET, mvq_src_c0[2]);
            neon::vload(src_c1 + x + VOFFSET, mvq_src_c1[2]);
            neon::vload(src_n0 + x + VOFFSET, mvq_src_n0[2]);

            for (DT_S32 ch = 0; ch < C; ch++)
            {
                BoxFilter3x3Core<Tp>(mvq_src_p0[0].val[ch], mvq_src_p0[1].val[ch], mvq_src_p0[2].val[ch],
                                     mvq_src_c0[0].val[ch], mvq_src_c0[1].val[ch], mvq_src_c0[2].val[ch],
                                     mvq_src_c1[0].val[ch], mvq_src_c1[1].val[ch], mvq_src_c1[2].val[ch],
                                     mvq_src_n0[0].val[ch], mvq_src_n0[1].val[ch], mvq_src_n0[2].val[ch],
                                     mvq_result0.val[ch],   mvq_result1.val[ch]);
            }

            neon::vstore(dst_c0 + x, mvq_result0);
            neon::vstore(dst_c1 + x, mvq_result1);
        }
    }

    // back
    {
        if (width_align != width * C)
        {
            DT_S32 x = (width - (ELEM_COUNTS << 1)) * C;
            neon::vload(src_p0 + x - VOFFSET, mvq_src_p0[0]);
            neon::vload(src_c0 + x - VOFFSET, mvq_src_c0[0]);
            neon::vload(src_c1 + x - VOFFSET, mvq_src_c1[0]);
            neon::vload(src_n0 + x - VOFFSET, mvq_src_n0[0]);
            neon::vload(src_p0 + x,           mvq_src_p0[1]);
            neon::vload(src_c0 + x,           mvq_src_c0[1]);
            neon::vload(src_c1 + x,           mvq_src_c1[1]);
            neon::vload(src_n0 + x,           mvq_src_n0[1]);
            neon::vload(src_p0 + x + VOFFSET, mvq_src_p0[2]);
            neon::vload(src_c0 + x + VOFFSET, mvq_src_c0[2]);
            neon::vload(src_c1 + x + VOFFSET, mvq_src_c1[2]);
            neon::vload(src_n0 + x + VOFFSET, mvq_src_n0[2]);

            for (DT_S32 ch = 0; ch < C; ch++)
            {
                BoxFilter3x3Core<Tp>(mvq_src_p0[0].val[ch], mvq_src_p0[1].val[ch], mvq_src_p0[2].val[ch],
                                     mvq_src_c0[0].val[ch], mvq_src_c0[1].val[ch], mvq_src_c0[2].val[ch],
                                     mvq_src_c1[0].val[ch], mvq_src_c1[1].val[ch], mvq_src_c1[2].val[ch],
                                     mvq_src_n0[0].val[ch], mvq_src_n0[1].val[ch], mvq_src_n0[2].val[ch],
                                     mvq_result0.val[ch],   mvq_result1.val[ch]);
            }

            neon::vstore(dst_c0 + x, mvq_result0);
            neon::vstore(dst_c1 + x, mvq_result1);
        }
    }

    // right border
    {
        DT_S32 x    = (width - ELEM_COUNTS) * C;
        DT_S32 last = (width - 1) * C;

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            mvq_src_p0[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mvq_src_p0[1].val[ch], src_p0[last + ch], border_value[ch]);
            mvq_src_c0[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mvq_src_c0[1].val[ch], src_c0[last + ch], border_value[ch]);
            mvq_src_c1[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mvq_src_c1[1].val[ch], src_c1[last + ch], border_value[ch]);
            mvq_src_n0[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mvq_src_n0[1].val[ch], src_n0[last + ch], border_value[ch]);

            BoxFilter3x3Core<Tp>(mvq_src_p0[0].val[ch], mvq_src_p0[1].val[ch], mvq_src_p0[2].val[ch],
                                 mvq_src_c0[0].val[ch], mvq_src_c0[1].val[ch], mvq_src_c0[2].val[ch],
                                 mvq_src_c1[0].val[ch], mvq_src_c1[1].val[ch], mvq_src_c1[2].val[ch],
                                 mvq_src_n0[0].val[ch], mvq_src_n0[1].val[ch], mvq_src_n0[2].val[ch],
                                 mvq_result0.val[ch],   mvq_result1.val[ch]);
        }

        neon::vstore(dst_c0 + x, mvq_result0);
        neon::vstore(dst_c1 + x, mvq_result1);
    }
}

template <typename Tp, BorderType BORDER_TYPE, DT_S32 C>
static Status BoxFilter3x3NeonImpl(const Mat &src, Mat &dst, const std::vector<Tp> &border_value,
                                   const Tp *border_buffer, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 width = dst.GetSizes().m_width;

    DT_S32 y = start_row;

    const Tp *src_p  = src.Ptr<Tp, BORDER_TYPE>(y - 1, border_buffer);
    const Tp *src_c0 = src.Ptr<Tp>(y);
    const Tp *src_c1 = src.Ptr<Tp, BORDER_TYPE>(y + 1);
    const Tp *src_n  = src.Ptr<Tp, BORDER_TYPE>(y + 2, border_buffer);

    DT_S32 h_align2 = (end_row - start_row) & (-2);
    for (; y < start_row + h_align2; y += 2)
    {
        Tp *dst_c0 = dst.Ptr<Tp>(y);
        Tp *dst_c1 = dst.Ptr<Tp>(y + 1);
        BoxFilter3x3TwoRow<Tp, BORDER_TYPE, C>(src_p, src_c0, src_c1, src_n, dst_c0, dst_c1, width, border_value);

        src_p  = src_c1;
        src_c0 = src_n;
        src_c1 = src.Ptr<Tp, BORDER_TYPE>(y + 3, border_buffer);
        src_n  = src.Ptr<Tp, BORDER_TYPE>(y + 4, border_buffer);
    }

    src_n = src.Ptr<Tp, BORDER_TYPE>(y + 1, border_buffer);
    if (y < end_row)
    {
        Tp *dst_c = dst.Ptr<Tp>(y);
        BoxFilter3x3OneRow<Tp, BORDER_TYPE, C>(src_p, src_c0, src_n, dst_c, width, border_value);
    }

    return Status::OK;
}

template <typename Tp, BorderType BORDER_TYPE>
static Status BoxFilter3x3NeonHelper(Context *ctx, const Mat &src, Mat &dst,
                                     const std::vector<Tp> &border_value,
                                     const Tp *border_buffer, const OpTarget &target)
{
    AURA_UNUSED(target);
    Status ret = Status::ERROR;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return ret;
    }

    DT_S32 height  = dst.GetSizes().m_height;
    DT_S32 channel = dst.GetSizes().m_channel;

    switch(channel)
    {
        case 1:
        {
            ret = wp->ParallelFor(0, height, BoxFilter3x3NeonImpl<Tp, BORDER_TYPE, 1>, std::cref(src), std::ref(dst),
                                  std::cref(border_value), border_buffer);
            break;
        }

        case 2:
        {
            ret = wp->ParallelFor(0, height, BoxFilter3x3NeonImpl<Tp, BORDER_TYPE, 2>, std::cref(src), std::ref(dst),
                                  std::cref(border_value), border_buffer);
            break;
        }

        case 3:
        {
            ret = wp->ParallelFor(0, height, BoxFilter3x3NeonImpl<Tp, BORDER_TYPE, 3>, std::cref(src), std::ref(dst),
                                  std::cref(border_value), border_buffer);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "Unsupported channel");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

template <typename Tp>
static Status BoxFilter3x3NeonHelper(Context *ctx, const Mat &src, Mat &dst, BorderType border_type,
                                     const Scalar &border_value, const OpTarget &target)
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

            ret = BoxFilter3x3NeonHelper<Tp, BorderType::CONSTANT>(ctx, src, dst, vec_border_value, border_buffer, target);

            AURA_FREE(ctx, border_buffer);

            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "BoxFilter3x3NeonHelper<Tp, BorderType::CONSTANT> failed");
            }
            break;
        }

        case BorderType::REPLICATE:
        {
            ret = BoxFilter3x3NeonHelper<Tp, BorderType::REPLICATE>(ctx, src, dst, vec_border_value, border_buffer, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "BoxFilter3x3NeonHelper<Tp, BorderType::REPLICATE> failed");
            }
            break;
        }

        case BorderType::REFLECT_101:
        {
            ret = BoxFilter3x3NeonHelper<Tp, BorderType::REFLECT_101>(ctx, src, dst, vec_border_value, border_buffer, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "BoxFilter3x3NeonHelper<Tp, BorderType::REFLECT_101> failed");
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "Unsupported border_type.");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

Status BoxFilter3x3Neon(Context *ctx, const Mat &src, Mat &dst, BorderType border_type,
                        const Scalar &border_value, const OpTarget &target)
{
    Status ret = Status::ERROR;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = BoxFilter3x3NeonHelper<DT_U8>(ctx, src, dst, border_type, border_value, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "BoxFilter3x3NeonHelper<DT_U8> failed");
            }
            break;
        }

        case ElemType::S8:
        {
            ret = BoxFilter3x3NeonHelper<DT_S8>(ctx, src, dst, border_type, border_value, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "BoxFilter3x3NeonHelper<DT_S8> failed");
            }
            break;
        }

        case ElemType::U16:
        {
            ret = BoxFilter3x3NeonHelper<DT_U16>(ctx, src, dst, border_type, border_value, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "BoxFilter3x3NeonHelper<DT_U16> failed");
            }
            break;
        }

        case ElemType::S16:
        {
            ret = BoxFilter3x3NeonHelper<DT_S16>(ctx, src, dst, border_type, border_value, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "BoxFilter3x3NeonHelper<DT_S16> failed");
            }
            break;
        }

#if defined(AURA_ENABLE_NEON_FP16)
        case ElemType::F16:
        {
            ret = BoxFilter3x3NeonHelper<MI_F16>(ctx, src, dst, border_type, border_value, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "BoxFilter3x3NeonHelper<MI_F16> failed");
            }
            break;
        }
#endif // AURA_ENABLE_NEON_FP16

        case ElemType::F32:
        {
            ret = BoxFilter3x3NeonHelper<DT_F32>(ctx, src, dst, border_type, border_value, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "BoxFilter3x3NeonHelper<DT_F32> failed");
            }
            break;
        }

        default :
        {
            AURA_ADD_ERROR_STRING(ctx, "Unsupported source format");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura