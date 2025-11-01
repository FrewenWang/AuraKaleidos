#include "boxfilter_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/thread_buffer.h"

namespace aura
{

template <typename D8, typename d8x16_t = typename neon::QVector<D8>::VType,
                       typename d16x8_t = typename neon::QVector<typename Promote<D8>::Type>::VType,
                       typename std::enable_if<std::is_same<D8, MI_U8>::value || std::is_same<D8, MI_S8>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE d8x16_t BoxFilter5x5SumCore(d8x16_t &vq8_src_p1x0, d8x16_t &vq8_src_p1x1, d8x16_t &vq8_src_p1x2,
                                               d8x16_t &vq8_src_p0x0, d8x16_t &vq8_src_p0x1, d8x16_t &vq8_src_p0x2,
                                               d8x16_t &vq8_src_cx0,  d8x16_t &vq8_src_cx1,  d8x16_t &vq8_src_cx2,
                                               d8x16_t &vq8_src_n0x0, d8x16_t &vq8_src_n0x1, d8x16_t &vq8_src_n0x2,
                                               d8x16_t &vq8_src_n1x0, d8x16_t &vq8_src_n1x1, d8x16_t &vq8_src_n1x2,
                                               d16x8_t &vq16_sum_lo,  d16x8_t &vq16_sum_hi)
{
    d16x8_t vq16_sum_x0_hi = neon::vaddw(neon::vadd(neon::vaddl(neon::vgethigh(vq8_src_p1x0), neon::vgethigh(vq8_src_p0x0)),
                                                    neon::vaddl(neon::vgethigh(vq8_src_cx0),  neon::vgethigh(vq8_src_n0x0))),
                                                    neon::vgethigh(vq8_src_n1x0));
    d16x8_t vq16_sum_x1_lo = neon::vaddw(neon::vadd(neon::vaddl(neon::vgetlow(vq8_src_p1x1),  neon::vgetlow(vq8_src_p0x1)),
                                                    neon::vaddl(neon::vgetlow(vq8_src_cx1),   neon::vgetlow(vq8_src_n0x1))),
                                                    neon::vgetlow(vq8_src_n1x1));
    d16x8_t vq16_sum_x1_hi = neon::vaddw(neon::vadd(neon::vaddl(neon::vgethigh(vq8_src_p1x1), neon::vgethigh(vq8_src_p0x1)),
                                                    neon::vaddl(neon::vgethigh(vq8_src_cx1),  neon::vgethigh(vq8_src_n0x1))),
                                                    neon::vgethigh(vq8_src_n1x1));
    d16x8_t vq16_sum_x2_lo = neon::vaddw(neon::vadd(neon::vaddl(neon::vgetlow(vq8_src_p1x2),  neon::vgetlow(vq8_src_p0x2)),
                                                    neon::vaddl(neon::vgetlow(vq8_src_cx2),   neon::vgetlow(vq8_src_n0x2))),
                                                    neon::vgetlow(vq8_src_n1x2));

    d16x8_t vq16_sum_l1_lo = neon::vext<6>(vq16_sum_x0_hi, vq16_sum_x1_lo);
    d16x8_t vq16_sum_l0_lo = neon::vext<7>(vq16_sum_x0_hi, vq16_sum_x1_lo);
    d16x8_t vq16_sum_r0_lo = neon::vext<1>(vq16_sum_x1_lo, vq16_sum_x1_hi);
    d16x8_t vq16_sum_r1_lo = neon::vext<2>(vq16_sum_x1_lo, vq16_sum_x1_hi);

    d16x8_t vq16_sum_l1_hi = neon::vext<6>(vq16_sum_x1_lo, vq16_sum_x1_hi);
    d16x8_t vq16_sum_l0_hi = neon::vext<7>(vq16_sum_x1_lo, vq16_sum_x1_hi);
    d16x8_t vq16_sum_r0_hi = neon::vext<1>(vq16_sum_x1_hi, vq16_sum_x2_lo);
    d16x8_t vq16_sum_r1_hi = neon::vext<2>(vq16_sum_x1_hi, vq16_sum_x2_lo);

    vq16_sum_lo = neon::vadd(neon::vadd(neon::vadd(vq16_sum_l1_lo, vq16_sum_l0_lo),
                                        neon::vadd(vq16_sum_x1_lo, vq16_sum_r0_lo)),
                                        vq16_sum_r1_lo);
    vq16_sum_hi = neon::vadd(neon::vadd(neon::vadd(vq16_sum_l1_hi, vq16_sum_l0_hi),
                                        neon::vadd(vq16_sum_x1_hi, vq16_sum_r0_hi)),
                                        vq16_sum_r1_hi);

    vq8_src_p1x0 = vq8_src_p1x1;
    vq8_src_p0x0 = vq8_src_p0x1;
    vq8_src_cx0  = vq8_src_cx1;
    vq8_src_n0x0 = vq8_src_n0x1;
    vq8_src_n1x0 = vq8_src_n1x1;

    vq8_src_p1x1 = vq8_src_p1x2;
    vq8_src_p0x1 = vq8_src_p0x2;
    vq8_src_cx1  = vq8_src_cx2;
    vq8_src_n0x1 = vq8_src_n0x2;
    vq8_src_n1x1 = vq8_src_n1x2;

    return neon::vcombine(neon::vqdivn_n<25>(vq16_sum_lo), neon::vqdivn_n<25>(vq16_sum_hi));
}

template <typename D8, typename d8x16_t = typename neon::QVector<D8>::VType,
                       typename d16x8_t = typename neon::QVector<typename Promote<D8>::Type>::VType,
                       typename std::enable_if<std::is_same<D8, MI_U8>::value || std::is_same<D8, MI_S8>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE d8x16_t BoxFilter5x5SlideCore(d8x16_t &vq8_src_p2x0, d8x16_t &vq8_src_p2x1, d8x16_t &vq8_src_p2x2,
                                                 d8x16_t &vq8_src_n1x0, d8x16_t &vq8_src_n1x1, d8x16_t &vq8_src_n1x2,
                                                 d16x8_t &vq16_sum_lo,  d16x8_t &vq16_sum_hi)
{
    d8x16_t vq8_src_p2l1   = neon::vext<14>(vq8_src_p2x0, vq8_src_p2x1);
    d8x16_t vq8_src_p2l0   = neon::vext<15>(vq8_src_p2x0, vq8_src_p2x1);
    d8x16_t vq8_src_p2r0   = neon::vext<1>(vq8_src_p2x1, vq8_src_p2x2);
    d8x16_t vq8_src_p2r1   = neon::vext<2>(vq8_src_p2x1, vq8_src_p2x2);

    d8x16_t vq8_src_n1l1   = neon::vext<14>(vq8_src_n1x0, vq8_src_n1x1);
    d8x16_t vq8_src_n1l0   = neon::vext<15>(vq8_src_n1x0, vq8_src_n1x1);
    d8x16_t vq8_src_n1r0   = neon::vext<1>(vq8_src_n1x1, vq8_src_n1x2);
    d8x16_t vq8_src_n1r1   = neon::vext<2>(vq8_src_n1x1, vq8_src_n1x2);

    d16x8_t vq16_sum_p2_lo = neon::vaddw(neon::vadd(neon::vaddl(neon::vgetlow(vq8_src_p2l1),  neon::vgetlow(vq8_src_p2l0)),
                                                    neon::vaddl(neon::vgetlow(vq8_src_p2x1),  neon::vgetlow(vq8_src_p2r0))),
                                                    neon::vgetlow(vq8_src_p2r1));
    d16x8_t vq16_sum_p2_hi = neon::vaddw(neon::vadd(neon::vaddl(neon::vgethigh(vq8_src_p2l1), neon::vgethigh(vq8_src_p2l0)),
                                                    neon::vaddl(neon::vgethigh(vq8_src_p2x1), neon::vgethigh(vq8_src_p2r0))),
                                                    neon::vgethigh(vq8_src_p2r1));
    d16x8_t vq16_sum_n1_lo = neon::vaddw(neon::vadd(neon::vaddl(neon::vgetlow(vq8_src_n1l1),  neon::vgetlow(vq8_src_n1l0)),
                                                    neon::vaddl(neon::vgetlow(vq8_src_n1x1),  neon::vgetlow(vq8_src_n1r0))),
                                                    neon::vgetlow(vq8_src_n1r1));
    d16x8_t vq16_sum_n1_hi = neon::vaddw(neon::vadd(neon::vaddl(neon::vgethigh(vq8_src_n1l1), neon::vgethigh(vq8_src_n1l0)),
                                                    neon::vaddl(neon::vgethigh(vq8_src_n1x1), neon::vgethigh(vq8_src_n1r0))),
                                                    neon::vgethigh(vq8_src_n1r1));

    vq16_sum_lo = neon::vsub(neon::vadd(vq16_sum_lo, vq16_sum_n1_lo), vq16_sum_p2_lo);
    vq16_sum_hi = neon::vsub(neon::vadd(vq16_sum_hi, vq16_sum_n1_hi), vq16_sum_p2_hi);

    vq8_src_p2x0 = vq8_src_p2x1;
    vq8_src_n1x0 = vq8_src_n1x1;

    vq8_src_p2x1 = vq8_src_p2x2;
    vq8_src_n1x1 = vq8_src_n1x2;

    return neon::vcombine(neon::vqdivn_n<25>(vq16_sum_lo), neon::vqdivn_n<25>(vq16_sum_hi));
}

template <typename D16, typename d16x8_t = typename neon::QVector<D16>::VType,
                        typename d32x4_t = typename neon::QVector<typename Promote<D16>::Type>::VType,
                        typename std::enable_if<std::is_same<D16, MI_U16>::value || std::is_same<D16, MI_S16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE d16x8_t BoxFilter5x5SumCore(d16x8_t &vq16_src_p1x0, d16x8_t &vq16_src_p1x1, d16x8_t &vq16_src_p1x2,
                                               d16x8_t &vq16_src_p0x0, d16x8_t &vq16_src_p0x1, d16x8_t &vq16_src_p0x2,
                                               d16x8_t &vq16_src_cx0,  d16x8_t &vq16_src_cx1,  d16x8_t &vq16_src_cx2,
                                               d16x8_t &vq16_src_n0x0, d16x8_t &vq16_src_n0x1, d16x8_t &vq16_src_n0x2,
                                               d16x8_t &vq16_src_n1x0, d16x8_t &vq16_src_n1x1, d16x8_t &vq16_src_n1x2,
                                               d32x4_t &vq32_sum_lo,   d32x4_t &vq32_sum_hi)
{
    d32x4_t vq32_sum_x0_hi = neon::vaddw(neon::vadd(neon::vaddl(neon::vgethigh(vq16_src_p1x0), neon::vgethigh(vq16_src_p0x0)),
                                                    neon::vaddl(neon::vgethigh(vq16_src_cx0),  neon::vgethigh(vq16_src_n0x0))),
                                                    neon::vgethigh(vq16_src_n1x0));
    d32x4_t vq32_sum_x1_lo = neon::vaddw(neon::vadd(neon::vaddl(neon::vgetlow(vq16_src_p1x1),  neon::vgetlow(vq16_src_p0x1)),
                                                    neon::vaddl(neon::vgetlow(vq16_src_cx1),   neon::vgetlow(vq16_src_n0x1))),
                                                    neon::vgetlow(vq16_src_n1x1));
    d32x4_t vq32_sum_x1_hi = neon::vaddw(neon::vadd(neon::vaddl(neon::vgethigh(vq16_src_p1x1), neon::vgethigh(vq16_src_p0x1)),
                                                    neon::vaddl(neon::vgethigh(vq16_src_cx1),  neon::vgethigh(vq16_src_n0x1))),
                                                    neon::vgethigh(vq16_src_n1x1));
    d32x4_t vq32_sum_x2_lo = neon::vaddw(neon::vadd(neon::vaddl(neon::vgetlow(vq16_src_p1x2),  neon::vgetlow(vq16_src_p0x2)),
                                                    neon::vaddl(neon::vgetlow(vq16_src_cx2),   neon::vgetlow(vq16_src_n0x2))),
                                                    neon::vgetlow(vq16_src_n1x2));

    d32x4_t vq32_sum_l1_lo = neon::vext<2>(vq32_sum_x0_hi, vq32_sum_x1_lo);
    d32x4_t vq32_sum_l0_lo = neon::vext<3>(vq32_sum_x0_hi, vq32_sum_x1_lo);
    d32x4_t vq32_sum_r0_lo = neon::vext<1>(vq32_sum_x1_lo, vq32_sum_x1_hi);
    d32x4_t vq32_sum_r1_lo = neon::vext<2>(vq32_sum_x1_lo, vq32_sum_x1_hi);

    d32x4_t vq32_sum_l1_hi = neon::vext<2>(vq32_sum_x1_lo, vq32_sum_x1_hi);
    d32x4_t vq32_sum_l0_hi = neon::vext<3>(vq32_sum_x1_lo, vq32_sum_x1_hi);
    d32x4_t vq32_sum_r0_hi = neon::vext<1>(vq32_sum_x1_hi, vq32_sum_x2_lo);
    d32x4_t vq32_sum_r1_hi = neon::vext<2>(vq32_sum_x1_hi, vq32_sum_x2_lo);

    vq32_sum_lo = neon::vadd(neon::vadd(neon::vadd(vq32_sum_l1_lo, vq32_sum_l0_lo),
                                        neon::vadd(vq32_sum_x1_lo, vq32_sum_r0_lo)),
                                        vq32_sum_r1_lo);
    vq32_sum_hi = neon::vadd(neon::vadd(neon::vadd(vq32_sum_l1_hi, vq32_sum_l0_hi),
                                        neon::vadd(vq32_sum_x1_hi, vq32_sum_r0_hi)),
                                        vq32_sum_r1_hi);

    vq16_src_p1x0 = vq16_src_p1x1;
    vq16_src_p0x0 = vq16_src_p0x1;
    vq16_src_cx0  = vq16_src_cx1;
    vq16_src_n0x0 = vq16_src_n0x1;
    vq16_src_n1x0 = vq16_src_n1x1;

    vq16_src_p1x1 = vq16_src_p1x2;
    vq16_src_p0x1 = vq16_src_p0x2;
    vq16_src_cx1  = vq16_src_cx2;
    vq16_src_n0x1 = vq16_src_n0x2;
    vq16_src_n1x1 = vq16_src_n1x2;

    return neon::vcombine(neon::vqdivn_n<25>(vq32_sum_lo), neon::vqdivn_n<25>(vq32_sum_hi));
}

template <typename D16, typename d16x8_t = typename neon::QVector<D16>::VType,
                        typename d32x4_t = typename neon::QVector<typename Promote<D16>::Type>::VType,
                        typename std::enable_if<std::is_same<D16, MI_U16>::value || std::is_same<D16, MI_S16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE d16x8_t BoxFilter5x5SlideCore(d16x8_t &vq16_src_p2x0, d16x8_t &vq16_src_p2x1, d16x8_t &vq16_src_p2x2,
                                                 d16x8_t &vq16_src_n1x0, d16x8_t &vq16_src_n1x1, d16x8_t &vq16_src_n1x2,
                                                 d32x4_t &vq32_sum_lo,   d32x4_t &vq32_sum_hi)
{
    d16x8_t vq16_src_p2l1  = neon::vext<6>(vq16_src_p2x0, vq16_src_p2x1);
    d16x8_t vq16_src_p2l0  = neon::vext<7>(vq16_src_p2x0, vq16_src_p2x1);
    d16x8_t vq16_src_p2r0  = neon::vext<1>(vq16_src_p2x1, vq16_src_p2x2);
    d16x8_t vq16_src_p2r1  = neon::vext<2>(vq16_src_p2x1, vq16_src_p2x2);

    d16x8_t vq16_src_n1l1  = neon::vext<6>(vq16_src_n1x0, vq16_src_n1x1);
    d16x8_t vq16_src_n1l0  = neon::vext<7>(vq16_src_n1x0, vq16_src_n1x1);
    d16x8_t vq16_src_n1r0  = neon::vext<1>(vq16_src_n1x1, vq16_src_n1x2);
    d16x8_t vq16_src_n1r1  = neon::vext<2>(vq16_src_n1x1, vq16_src_n1x2);

    d32x4_t vq32_sum_p2_lo = neon::vaddw(neon::vadd(neon::vaddl(neon::vgetlow(vq16_src_p2l1),  neon::vgetlow(vq16_src_p2l0)),
                                                    neon::vaddl(neon::vgetlow(vq16_src_p2x1),  neon::vgetlow(vq16_src_p2r0))),
                                                    neon::vgetlow(vq16_src_p2r1));
    d32x4_t vq32_sum_p2_hi = neon::vaddw(neon::vadd(neon::vaddl(neon::vgethigh(vq16_src_p2l1), neon::vgethigh(vq16_src_p2l0)),
                                                    neon::vaddl(neon::vgethigh(vq16_src_p2x1), neon::vgethigh(vq16_src_p2r0))),
                                                    neon::vgethigh(vq16_src_p2r1));
    d32x4_t vq32_sum_n1_lo = neon::vaddw(neon::vadd(neon::vaddl(neon::vgetlow(vq16_src_n1l1),  neon::vgetlow(vq16_src_n1l0)),
                                                    neon::vaddl(neon::vgetlow(vq16_src_n1x1),  neon::vgetlow(vq16_src_n1r0))),
                                                    neon::vgetlow(vq16_src_n1r1));
    d32x4_t vq32_sum_n1_hi = neon::vaddw(neon::vadd(neon::vaddl(neon::vgethigh(vq16_src_n1l1), neon::vgethigh(vq16_src_n1l0)),
                                                    neon::vaddl(neon::vgethigh(vq16_src_n1x1), neon::vgethigh(vq16_src_n1r0))),
                                                    neon::vgethigh(vq16_src_n1r1));

    vq32_sum_lo = neon::vsub(neon::vadd(vq32_sum_lo, vq32_sum_n1_lo), vq32_sum_p2_lo);
    vq32_sum_hi = neon::vsub(neon::vadd(vq32_sum_hi, vq32_sum_n1_hi), vq32_sum_p2_hi);

    vq16_src_p2x0 = vq16_src_p2x1;
    vq16_src_n1x0 = vq16_src_n1x1;

    vq16_src_p2x1 = vq16_src_p2x2;
    vq16_src_n1x1 = vq16_src_n1x2;

    return neon::vcombine(neon::vqdivn_n<25>(vq32_sum_lo), neon::vqdivn_n<25>(vq32_sum_hi));
}

#if defined(AURA_ENABLE_NEON_FP16)
template <typename Tp, typename std::enable_if<std::is_same<Tp, MI_F16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE float16x8_t BoxFilter5x5SumCore(float16x8_t &vqf16_src_p1x0, float16x8_t &vqf16_src_p1x1, float16x8_t &vqf16_src_p1x2,
                                                   float16x8_t &vqf16_src_p0x0, float16x8_t &vqf16_src_p0x1, float16x8_t &vqf16_src_p0x2,
                                                   float16x8_t &vqf16_src_cx0,  float16x8_t &vqf16_src_cx1,  float16x8_t &vqf16_src_cx2,
                                                   float16x8_t &vqf16_src_n0x0, float16x8_t &vqf16_src_n0x1, float16x8_t &vqf16_src_n0x2,
                                                   float16x8_t &vqf16_src_n1x0, float16x8_t &vqf16_src_n1x1, float16x8_t &vqf16_src_n1x2,
                                                   float32x4_t &vqf32_sum_lo,   float32x4_t &vqf32_sum_hi)
{
    float32x4_t vqf32_sum_x0_hi = neon::vadd(neon::vadd(neon::vadd(neon::vcvt<MI_F32>(neon::vgethigh(vqf16_src_p1x0)),
                                                                   neon::vcvt<MI_F32>(neon::vgethigh(vqf16_src_p0x0))),
                                                        neon::vadd(neon::vcvt<MI_F32>(neon::vgethigh(vqf16_src_cx0)),
                                                                   neon::vcvt<MI_F32>(neon::vgethigh(vqf16_src_n0x0)))),
                                                                   neon::vcvt<MI_F32>(neon::vgethigh(vqf16_src_n1x0)));
    float32x4_t vqf32_sum_x1_lo = neon::vadd(neon::vadd(neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(vqf16_src_p1x1)),
                                                                   neon::vcvt<MI_F32>(neon::vgetlow(vqf16_src_p0x1))),
                                                        neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(vqf16_src_cx1)),
                                                                   neon::vcvt<MI_F32>(neon::vgetlow(vqf16_src_n0x1)))),
                                                                   neon::vcvt<MI_F32>(neon::vgetlow(vqf16_src_n1x1)));
    float32x4_t vqf32_sum_x1_hi = neon::vadd(neon::vadd(neon::vadd(neon::vcvt<MI_F32>(neon::vgethigh(vqf16_src_p1x1)),
                                                                   neon::vcvt<MI_F32>(neon::vgethigh(vqf16_src_p0x1))),
                                                        neon::vadd(neon::vcvt<MI_F32>(neon::vgethigh(vqf16_src_cx1)),
                                                                   neon::vcvt<MI_F32>(neon::vgethigh(vqf16_src_n0x1)))),
                                                                   neon::vcvt<MI_F32>(neon::vgethigh(vqf16_src_n1x1)));
    float32x4_t vqf32_sum_x2_lo = neon::vadd(neon::vadd(neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(vqf16_src_p1x2)),
                                                                   neon::vcvt<MI_F32>(neon::vgetlow(vqf16_src_p0x2))),
                                                        neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(vqf16_src_cx2)),
                                                                   neon::vcvt<MI_F32>(neon::vgetlow(vqf16_src_n0x2)))),
                                                                   neon::vcvt<MI_F32>(neon::vgetlow(vqf16_src_n1x2)));

    float32x4_t vqf32_sum_l1_lo = neon::vext<2>(vqf32_sum_x0_hi, vqf32_sum_x1_lo);
    float32x4_t vqf32_sum_l0_lo = neon::vext<3>(vqf32_sum_x0_hi, vqf32_sum_x1_lo);
    float32x4_t vqf32_sum_r0_lo = neon::vext<1>(vqf32_sum_x1_lo, vqf32_sum_x1_hi);
    float32x4_t vqf32_sum_r1_lo = neon::vext<2>(vqf32_sum_x1_lo, vqf32_sum_x1_hi);

    float32x4_t vqf32_sum_l1_hi = neon::vext<2>(vqf32_sum_x1_lo, vqf32_sum_x1_hi);
    float32x4_t vqf32_sum_l0_hi = neon::vext<3>(vqf32_sum_x1_lo, vqf32_sum_x1_hi);
    float32x4_t vqf32_sum_r0_hi = neon::vext<1>(vqf32_sum_x1_hi, vqf32_sum_x2_lo);
    float32x4_t vqf32_sum_r1_hi = neon::vext<2>(vqf32_sum_x1_hi, vqf32_sum_x2_lo);

    vqf32_sum_lo = neon::vadd(neon::vadd(neon::vadd(vqf32_sum_l1_lo, vqf32_sum_l0_lo),
                                         neon::vadd(vqf32_sum_x1_lo, vqf32_sum_r0_lo)),
                                         vqf32_sum_r1_lo);
    vqf32_sum_hi = neon::vadd(neon::vadd(neon::vadd(vqf32_sum_l1_hi, vqf32_sum_l0_hi),
                                         neon::vadd(vqf32_sum_x1_hi, vqf32_sum_r0_hi)),
                                         vqf32_sum_r1_hi);

    float32x4_t vqf32_result_lo = neon::vmul(vqf32_sum_lo, static_cast<MI_F32>(1.0 / 25));
    float32x4_t vqf32_result_hi = neon::vmul(vqf32_sum_hi, static_cast<MI_F32>(1.0 / 25));

    float16x8_t vqf16_result = neon::vcombine(neon::vcvt<MI_F16>(vqf32_result_lo), neon::vcvt<MI_F16>(vqf32_result_hi));

    vqf16_src_p1x0 = vqf16_src_p1x1;
    vqf16_src_p0x0 = vqf16_src_p0x1;
    vqf16_src_cx0  = vqf16_src_cx1;
    vqf16_src_n0x0 = vqf16_src_n0x1;
    vqf16_src_n1x0 = vqf16_src_n1x1;

    vqf16_src_p1x1 = vqf16_src_p1x2;
    vqf16_src_p0x1 = vqf16_src_p0x2;
    vqf16_src_cx1  = vqf16_src_cx2;
    vqf16_src_n0x1 = vqf16_src_n0x2;
    vqf16_src_n1x1 = vqf16_src_n1x2;

    return vqf16_result;
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, MI_F16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE float16x8_t BoxFilter5x5SlideCore(float16x8_t &vqf16_src_p2x0, float16x8_t &vqf16_src_p2x1, float16x8_t &vqf16_src_p2x2,
                                                     float16x8_t &vqf16_src_n1x0, float16x8_t &vqf16_src_n1x1, float16x8_t &vqf16_src_n1x2,
                                                     float32x4_t &vqf32_sum_lo,   float32x4_t &vqf32_sum_hi)
{
    float16x8_t vqf16_src_p2l1 = neon::vext<6>(vqf16_src_p2x0, vqf16_src_p2x1);
    float16x8_t vqf16_src_p2l0 = neon::vext<7>(vqf16_src_p2x0, vqf16_src_p2x1);
    float16x8_t vqf16_src_p2r0 = neon::vext<1>(vqf16_src_p2x1, vqf16_src_p2x2);
    float16x8_t vqf16_src_p2r1 = neon::vext<2>(vqf16_src_p2x1, vqf16_src_p2x2);

    float16x8_t vqf16_src_n1l1 = neon::vext<6>(vqf16_src_n1x0, vqf16_src_n1x1);
    float16x8_t vqf16_src_n1l0 = neon::vext<7>(vqf16_src_n1x0, vqf16_src_n1x1);
    float16x8_t vqf16_src_n1r0 = neon::vext<1>(vqf16_src_n1x1, vqf16_src_n1x2);
    float16x8_t vqf16_src_n1r1 = neon::vext<2>(vqf16_src_n1x1, vqf16_src_n1x2);

    float32x4_t vqf32_sum_p2_lo = neon::vadd(neon::vadd(neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(vqf16_src_p2l1)),
                                                                   neon::vcvt<MI_F32>(neon::vgetlow(vqf16_src_p2l0))),
                                                        neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(vqf16_src_p2x1)),
                                                                   neon::vcvt<MI_F32>(neon::vgetlow(vqf16_src_p2r0)))),
                                                                   neon::vcvt<MI_F32>(neon::vgetlow(vqf16_src_p2r1)));
    float32x4_t vqf32_sum_p2_hi = neon::vadd(neon::vadd(neon::vadd(neon::vcvt<MI_F32>(neon::vgethigh(vqf16_src_p2l1)),
                                                                   neon::vcvt<MI_F32>(neon::vgethigh(vqf16_src_p2l0))),
                                                        neon::vadd(neon::vcvt<MI_F32>(neon::vgethigh(vqf16_src_p2x1)),
                                                                   neon::vcvt<MI_F32>(neon::vgethigh(vqf16_src_p2r0)))),
                                                                   neon::vcvt<MI_F32>(neon::vgethigh(vqf16_src_p2r1)));
    float32x4_t vqf32_sum_n1_lo = neon::vadd(neon::vadd(neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(vqf16_src_n1l1)),
                                                                   neon::vcvt<MI_F32>(neon::vgetlow(vqf16_src_n1l0))),
                                                        neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(vqf16_src_n1x1)),
                                                                   neon::vcvt<MI_F32>(neon::vgetlow(vqf16_src_n1r0)))),
                                                                   neon::vcvt<MI_F32>(neon::vgetlow(vqf16_src_n1r1)));
    float32x4_t vqf32_sum_n1_hi = neon::vadd(neon::vadd(neon::vadd(neon::vcvt<MI_F32>(neon::vgethigh(vqf16_src_n1l1)),
                                                                   neon::vcvt<MI_F32>(neon::vgethigh(vqf16_src_n1l0))),
                                                        neon::vadd(neon::vcvt<MI_F32>(neon::vgethigh(vqf16_src_n1x1)),
                                                                   neon::vcvt<MI_F32>(neon::vgethigh(vqf16_src_n1r0)))),
                                                                   neon::vcvt<MI_F32>(neon::vgethigh(vqf16_src_n1r1)));

    vqf32_sum_lo = neon::vsub(neon::vadd(vqf32_sum_lo, vqf32_sum_n1_lo), vqf32_sum_p2_lo);
    vqf32_sum_hi = neon::vsub(neon::vadd(vqf32_sum_hi, vqf32_sum_n1_hi), vqf32_sum_p2_hi);

    float32x4_t vqf32_result_lo = neon::vmul(vqf32_sum_lo, static_cast<MI_F32>(1.0 / 25));
    float32x4_t vqf32_result_hi = neon::vmul(vqf32_sum_hi, static_cast<MI_F32>(1.0 / 25));
    float16x8_t vqf16_result    = neon::vcombine(neon::vcvt<MI_F16>(vqf32_result_lo), neon::vcvt<MI_F16>(vqf32_result_hi));

    vqf16_src_p2x0 = vqf16_src_p2x1;
    vqf16_src_n1x0 = vqf16_src_n1x1;

    vqf16_src_p2x1 = vqf16_src_p2x2;
    vqf16_src_n1x1 = vqf16_src_n1x2;

    return vqf16_result;
}
#endif // AURA_ENABLE_NEON_FP16

template <typename Tp, typename std::enable_if<std::is_same<Tp, MI_F32>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE float32x4_t BoxFilter5x5SumCore(float32x4_t &vqf32_src_p1x0, float32x4_t &vqf32_src_p1x1, float32x4_t &vqf32_src_p1x2,
                                                   float32x4_t &vqf32_src_p0x0, float32x4_t &vqf32_src_p0x1, float32x4_t &vqf32_src_p0x2,
                                                   float32x4_t &vqf32_src_cx0,  float32x4_t &vqf32_src_cx1,  float32x4_t &vqf32_src_cx2,
                                                   float32x4_t &vqf32_src_n0x0, float32x4_t &vqf32_src_n0x1, float32x4_t &vqf32_src_n0x2,
                                                   float32x4_t &vqf32_src_n1x0, float32x4_t &vqf32_src_n1x1, float32x4_t &vqf32_src_n1x2,
                                                   float32x4_t &vqf32_sum,      float32x4_t &vqf32_sum_unused)
{
    AURA_UNUSED(vqf32_sum_unused);

    float32x4_t vqf32_sum_x0 = neon::vadd(neon::vadd(neon::vadd(vqf32_src_p1x0, vqf32_src_p0x0), neon::vadd(vqf32_src_cx0, vqf32_src_n0x0)), vqf32_src_n1x0);
    float32x4_t vqf32_sum_x1 = neon::vadd(neon::vadd(neon::vadd(vqf32_src_p1x1, vqf32_src_p0x1), neon::vadd(vqf32_src_cx1, vqf32_src_n0x1)), vqf32_src_n1x1);
    float32x4_t vqf32_sum_x2 = neon::vadd(neon::vadd(neon::vadd(vqf32_src_p1x2, vqf32_src_p0x2), neon::vadd(vqf32_src_cx2, vqf32_src_n0x2)), vqf32_src_n1x2);

    float32x4_t vqf32_sum_l1 = neon::vext<2>(vqf32_sum_x0, vqf32_sum_x1);
    float32x4_t vqf32_sum_l0 = neon::vext<3>(vqf32_sum_x0, vqf32_sum_x1);
    float32x4_t vqf32_sum_r0 = neon::vext<1>(vqf32_sum_x1, vqf32_sum_x2);
    float32x4_t vqf32_sum_r1 = neon::vext<2>(vqf32_sum_x1, vqf32_sum_x2);

    vqf32_sum = neon::vadd(neon::vadd(neon::vadd(vqf32_sum_l1, vqf32_sum_l0), neon::vadd(vqf32_sum_x1, vqf32_sum_r0)), vqf32_sum_r1);

    vqf32_src_p1x0 = vqf32_src_p1x1;
    vqf32_src_p0x0 = vqf32_src_p0x1;
    vqf32_src_cx0  = vqf32_src_cx1;
    vqf32_src_n0x0 = vqf32_src_n0x1;
    vqf32_src_n1x0 = vqf32_src_n1x1;

    vqf32_src_p1x1 = vqf32_src_p1x2;
    vqf32_src_p0x1 = vqf32_src_p0x2;
    vqf32_src_cx1  = vqf32_src_cx2;
    vqf32_src_n0x1 = vqf32_src_n0x2;
    vqf32_src_n1x1 = vqf32_src_n1x2;

    return neon::vmul(vqf32_sum, static_cast<MI_F32>(1.0 / 25));
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, MI_F32>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE float32x4_t BoxFilter5x5SlideCore(float32x4_t &vqf32_src_p2x0, float32x4_t &vqf32_src_p2x1, float32x4_t &vqf32_src_p2x2,
                                                     float32x4_t &vqf32_src_n1x0, float32x4_t &vqf32_src_n1x1, float32x4_t &vqf32_src_n1x2,
                                                     float32x4_t &vqf32_sum,      float32x4_t &vqf32_sum_unused)
{
    AURA_UNUSED(vqf32_sum_unused);

    float32x4_t vqf32_src_p2l1 = neon::vext<2>(vqf32_src_p2x0, vqf32_src_p2x1);
    float32x4_t vqf32_src_p2l0 = neon::vext<3>(vqf32_src_p2x0, vqf32_src_p2x1);
    float32x4_t vqf32_src_p2r0 = neon::vext<1>(vqf32_src_p2x1, vqf32_src_p2x2);
    float32x4_t vqf32_src_p2r1 = neon::vext<2>(vqf32_src_p2x1, vqf32_src_p2x2);

    float32x4_t vqf32_src_n1l1 = neon::vext<2>(vqf32_src_n1x0, vqf32_src_n1x1);
    float32x4_t vqf32_src_n1l0 = neon::vext<3>(vqf32_src_n1x0, vqf32_src_n1x1);
    float32x4_t vqf32_src_n1r0 = neon::vext<1>(vqf32_src_n1x1, vqf32_src_n1x2);
    float32x4_t vqf32_src_n1r1 = neon::vext<2>(vqf32_src_n1x1, vqf32_src_n1x2);

    float32x4_t vqf32_sum_p2   = neon::vadd(neon::vadd(neon::vadd(vqf32_src_p2l1, vqf32_src_p2l0), neon::vadd(vqf32_src_p2x1, vqf32_src_p2r0)), vqf32_src_p2r1);
    float32x4_t vqf32_sum_n1   = neon::vadd(neon::vadd(neon::vadd(vqf32_src_n1l1, vqf32_src_n1l0), neon::vadd(vqf32_src_n1x1, vqf32_src_n1r0)), vqf32_src_n1r1);
    vqf32_sum                  = neon::vsub(neon::vadd(vqf32_sum, vqf32_sum_n1), vqf32_sum_p2);

    float32x4_t vqf32_result   = neon::vmul(vqf32_sum, static_cast<MI_F32>(1.0 / 25));

    vqf32_src_p2x0 = vqf32_src_p2x1;
    vqf32_src_n1x0 = vqf32_src_n1x1;

    vqf32_src_p2x1 = vqf32_src_p2x2;
    vqf32_src_n1x1 = vqf32_src_n1x2;

    return vqf32_result;
}

template <typename Tp, BorderType BORDER_TYPE, MI_S32 C, typename SumType = typename Promote<Tp>::Type,
          MI_BOOL ISF32 = std::is_same<Tp, MI_F32>::value>
static AURA_VOID BoxFilter5x5Row(const Tp *src_p2, const Tp *src_n1, Tp *dst_c, MI_S32 width, SumType *sum_buffer,
                               const std::vector<Tp> &border_value)
{
    using MVqType    = typename neon::MQVector<Tp, C>::MVType;
    using MVqSumType = typename neon::MQVector<SumType, C>::MVType;

    constexpr MI_S32 ELEM_COUNTS = 16 / sizeof(Tp);
    constexpr MI_S32 VOFFSET     = ELEM_COUNTS * C;
    const MI_S32 width_align     = (width & -ELEM_COUNTS) * C;

    MVqType mvq_src_p2[3], mvq_src_n1[3];
    MVqType mvq_result;
    MVqSumType mvq_sum_lo, mvq_sum_hi, mvq_sum_back_lo, mvq_sum_back_hi;
    neon::vload(sum_buffer + (width - (ELEM_COUNTS << 1)) * C, mvq_sum_back_lo);
    if (!ISF32)
    {
        neon::vload(sum_buffer + (width - (ELEM_COUNTS << 1) + (ELEM_COUNTS >> 1)) * C, mvq_sum_back_hi);
    }
    else
    {
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            neon::vdup(mvq_sum_back_hi.val[ch], 0);
        }
    }

    // left border
    {
        neon::vload(src_p2,           mvq_src_p2[1]);
        neon::vload(src_n1,           mvq_src_n1[1]);
        neon::vload(src_p2 + VOFFSET, mvq_src_p2[2]);
        neon::vload(src_n1 + VOFFSET, mvq_src_n1[2]);
        neon::vload(sum_buffer,       mvq_sum_lo);
        if (!ISF32)
        {
            neon::vload(sum_buffer + (ELEM_COUNTS >> 1) * C, mvq_sum_hi);
        }

        for (MI_S32 ch = 0; ch < C; ch++)
        {
            mvq_src_p2[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mvq_src_p2[1].val[ch], src_p2[ch], border_value[ch]);
            mvq_src_n1[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mvq_src_n1[1].val[ch], src_n1[ch], border_value[ch]);

            mvq_result.val[ch] = BoxFilter5x5SlideCore<Tp>(mvq_src_p2[0].val[ch], mvq_src_p2[1].val[ch], mvq_src_p2[2].val[ch],
                                                           mvq_src_n1[0].val[ch], mvq_src_n1[1].val[ch], mvq_src_n1[2].val[ch],
                                                           mvq_sum_lo.val[ch],    mvq_sum_hi.val[ch]);
        }

        neon::vstore(dst_c,      mvq_result);
        neon::vstore(sum_buffer, mvq_sum_lo);
        if (!ISF32)
        {
            neon::vstore(sum_buffer + (ELEM_COUNTS >> 1) * C, mvq_sum_hi);
        }
    }

    // middle
    {
        for (MI_S32 x = VOFFSET; x < width_align - VOFFSET; x += VOFFSET)
        {
            neon::vload(src_p2 + x + VOFFSET, mvq_src_p2[2]);
            neon::vload(src_n1 + x + VOFFSET, mvq_src_n1[2]);
            neon::vload(sum_buffer + x,       mvq_sum_lo);
            if (!ISF32)
            {
                neon::vload(sum_buffer + x + (ELEM_COUNTS >> 1) * C, mvq_sum_hi);
            }

            for (MI_S32 ch = 0; ch < C; ch++)
            {
                mvq_result.val[ch] = BoxFilter5x5SlideCore<Tp>(mvq_src_p2[0].val[ch], mvq_src_p2[1].val[ch], mvq_src_p2[2].val[ch],
                                                               mvq_src_n1[0].val[ch], mvq_src_n1[1].val[ch], mvq_src_n1[2].val[ch],
                                                               mvq_sum_lo.val[ch],    mvq_sum_hi.val[ch]);
            }

            neon::vstore(dst_c + x, mvq_result);
            neon::vstore(sum_buffer + x, mvq_sum_lo);
            if (!ISF32)
            {
                neon::vstore(sum_buffer + x + (ELEM_COUNTS >> 1) * C, mvq_sum_hi);
            }
        }
    }

    // back
    {
        if (width_align != width  * C)
        {
            MI_S32 x = (width - (ELEM_COUNTS << 1)) * C;
            neon::vload(src_p2 + x - VOFFSET, mvq_src_p2[0]);
            neon::vload(src_n1 + x - VOFFSET, mvq_src_n1[0]);
            neon::vload(src_p2 + x,           mvq_src_p2[1]);
            neon::vload(src_n1 + x,           mvq_src_n1[1]);
            neon::vload(src_p2 + x + VOFFSET, mvq_src_p2[2]);
            neon::vload(src_n1 + x + VOFFSET, mvq_src_n1[2]);
            mvq_sum_lo = mvq_sum_back_lo;
            mvq_sum_hi = mvq_sum_back_hi;

            for (MI_S32 ch = 0; ch < C; ch++)
            {
                mvq_result.val[ch] = BoxFilter5x5SlideCore<Tp>(mvq_src_p2[0].val[ch], mvq_src_p2[1].val[ch], mvq_src_p2[2].val[ch],
                                                               mvq_src_n1[0].val[ch], mvq_src_n1[1].val[ch], mvq_src_n1[2].val[ch],
                                                               mvq_sum_lo.val[ch],    mvq_sum_hi.val[ch]);
            }

            neon::vstore(dst_c + x, mvq_result);
            neon::vstore(sum_buffer + x, mvq_sum_lo);
            if (!ISF32)
            {
                neon::vstore(sum_buffer + x + (ELEM_COUNTS >> 1) * C, mvq_sum_hi);
            }
        }
    }

    // right border
    {
        MI_S32 x    = (width - ELEM_COUNTS) * C;
        MI_S32 last = (width - 1) * C;
        neon::vload(sum_buffer + x, mvq_sum_lo);
        if (!ISF32)
        {
            neon::vload(sum_buffer + x + (ELEM_COUNTS >> 1) * C, mvq_sum_hi);
        }

        for (MI_S32 ch = 0; ch < C; ch++)
        {
            mvq_src_p2[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mvq_src_p2[1].val[ch], src_p2[last + ch], border_value[ch]);
            mvq_src_n1[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mvq_src_n1[1].val[ch], src_n1[last + ch], border_value[ch]);

            mvq_result.val[ch] = BoxFilter5x5SlideCore<Tp>(mvq_src_p2[0].val[ch], mvq_src_p2[1].val[ch], mvq_src_p2[2].val[ch],
                                                           mvq_src_n1[0].val[ch], mvq_src_n1[1].val[ch], mvq_src_n1[2].val[ch],
                                                           mvq_sum_lo.val[ch],    mvq_sum_hi.val[ch]);
        }
        neon::vstore(dst_c      + x, mvq_result);
        neon::vstore(sum_buffer + x, mvq_sum_lo);
        if (!ISF32)
        {
            neon::vstore(sum_buffer + x + (ELEM_COUNTS >> 1) * C, mvq_sum_hi);
        }
    }
}

template <typename Tp, BorderType BORDER_TYPE, MI_S32 C, typename SumType = typename Promote<Tp>::Type,
          MI_BOOL ISF32 = std::is_same<Tp, MI_F32>::value>
static AURA_VOID BoxFilter5x5Row(const Tp *src_p1, const Tp *src_p0, const Tp *src_c,
                               const Tp *src_n0, const Tp *src_n1, Tp *dst_c, MI_S32 width,
                               SumType *sum_buffer, const std::vector<Tp> &border_value)
{
    using MVqType    = typename neon::MQVector<Tp, C>::MVType;
    using MVqSumType = typename neon::MQVector<SumType, C>::MVType;

    constexpr MI_S32 ELEM_COUNTS = 16 / sizeof(Tp);
    constexpr MI_S32 VOFFSET     = ELEM_COUNTS * C;
    const MI_S32 width_align     = (width & -ELEM_COUNTS) * C;

    MVqType mvq_src_p1[3], mvq_src_p0[3], mvq_src_c[3], mvq_src_n0[3], mvq_src_n1[3];
    MVqType mvq_result;
    MVqSumType mvq_sum_lo, mvq_sum_hi;

    // left border
    {
        neon::vload(src_p1,           mvq_src_p1[1]);
        neon::vload(src_p0,           mvq_src_p0[1]);
        neon::vload(src_c,            mvq_src_c[1]);
        neon::vload(src_n0,           mvq_src_n0[1]);
        neon::vload(src_n1,           mvq_src_n1[1]);
        neon::vload(src_p1 + VOFFSET, mvq_src_p1[2]);
        neon::vload(src_p0 + VOFFSET, mvq_src_p0[2]);
        neon::vload(src_c  + VOFFSET, mvq_src_c[2]);
        neon::vload(src_n0 + VOFFSET, mvq_src_n0[2]);
        neon::vload(src_n1 + VOFFSET, mvq_src_n1[2]);

        for (MI_S32 ch = 0; ch < C; ch++)
        {
            mvq_src_p1[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mvq_src_p1[1].val[ch], src_p1[ch], border_value[ch]);
            mvq_src_p0[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mvq_src_p0[1].val[ch], src_p0[ch], border_value[ch]);
            mvq_src_c[0].val[ch]  = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mvq_src_c[1].val[ch],  src_c[ch],  border_value[ch]);
            mvq_src_n0[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mvq_src_n0[1].val[ch], src_n0[ch], border_value[ch]);
            mvq_src_n1[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mvq_src_n1[1].val[ch], src_n1[ch], border_value[ch]);

            mvq_result.val[ch] = BoxFilter5x5SumCore<Tp>(mvq_src_p1[0].val[ch], mvq_src_p1[1].val[ch], mvq_src_p1[2].val[ch],
                                                         mvq_src_p0[0].val[ch], mvq_src_p0[1].val[ch], mvq_src_p0[2].val[ch],
                                                         mvq_src_c[0].val[ch],  mvq_src_c[1].val[ch],  mvq_src_c[2].val[ch],
                                                         mvq_src_n0[0].val[ch], mvq_src_n0[1].val[ch], mvq_src_n0[2].val[ch],
                                                         mvq_src_n1[0].val[ch], mvq_src_n1[1].val[ch], mvq_src_n1[2].val[ch],
                                                         mvq_sum_lo.val[ch],    mvq_sum_hi.val[ch]);
        }

        neon::vstore(dst_c,      mvq_result);
        neon::vstore(sum_buffer, mvq_sum_lo);
        if (!ISF32)
        {
            neon::vstore(sum_buffer + (ELEM_COUNTS >> 1) * C, mvq_sum_hi);
        }
    }

    // middle
    {
        for (MI_S32 x = VOFFSET; x < width_align - VOFFSET; x += VOFFSET)
        {
            neon::vload(src_p1 + x + VOFFSET, mvq_src_p1[2]);
            neon::vload(src_p0 + x + VOFFSET, mvq_src_p0[2]);
            neon::vload(src_c  + x + VOFFSET, mvq_src_c[2]);
            neon::vload(src_n0 + x + VOFFSET, mvq_src_n0[2]);
            neon::vload(src_n1 + x + VOFFSET, mvq_src_n1[2]);

            for (MI_S32 ch = 0; ch < C; ch++)
            {
                mvq_result.val[ch] = BoxFilter5x5SumCore<Tp>(mvq_src_p1[0].val[ch], mvq_src_p1[1].val[ch], mvq_src_p1[2].val[ch],
                                                             mvq_src_p0[0].val[ch], mvq_src_p0[1].val[ch], mvq_src_p0[2].val[ch],
                                                             mvq_src_c[0].val[ch],  mvq_src_c[1].val[ch],  mvq_src_c[2].val[ch],
                                                             mvq_src_n0[0].val[ch], mvq_src_n0[1].val[ch], mvq_src_n0[2].val[ch],
                                                             mvq_src_n1[0].val[ch], mvq_src_n1[1].val[ch], mvq_src_n1[2].val[ch],
                                                             mvq_sum_lo.val[ch],    mvq_sum_hi.val[ch]);
            }

            neon::vstore(dst_c      + x, mvq_result);
            neon::vstore(sum_buffer + x, mvq_sum_lo);
            if (!ISF32)
            {
                neon::vstore(sum_buffer + x + (ELEM_COUNTS >> 1) * C, mvq_sum_hi);
            }
        }
    }

    // back
    {
        if (width_align != width  * C)
        {
            MI_S32 x = (width - (ELEM_COUNTS << 1)) * C;
            neon::vload(src_p1 + x - VOFFSET, mvq_src_p1[0]);
            neon::vload(src_p0 + x - VOFFSET, mvq_src_p0[0]);
            neon::vload(src_c  + x - VOFFSET, mvq_src_c[0]);
            neon::vload(src_n0 + x - VOFFSET, mvq_src_n0[0]);
            neon::vload(src_n1 + x - VOFFSET, mvq_src_n1[0]);
            neon::vload(src_p1 + x,           mvq_src_p1[1]);
            neon::vload(src_p0 + x,           mvq_src_p0[1]);
            neon::vload(src_c  + x,           mvq_src_c[1]);
            neon::vload(src_n0 + x,           mvq_src_n0[1]);
            neon::vload(src_n1 + x,           mvq_src_n1[1]);
            neon::vload(src_p1 + x + VOFFSET, mvq_src_p1[2]);
            neon::vload(src_p0 + x + VOFFSET, mvq_src_p0[2]);
            neon::vload(src_c  + x + VOFFSET, mvq_src_c[2]);
            neon::vload(src_n0 + x + VOFFSET, mvq_src_n0[2]);
            neon::vload(src_n1 + x + VOFFSET, mvq_src_n1[2]);

            for (MI_S32 ch = 0; ch < C; ch++)
            {
                mvq_result.val[ch] = BoxFilter5x5SumCore<Tp>(mvq_src_p1[0].val[ch], mvq_src_p1[1].val[ch], mvq_src_p1[2].val[ch],
                                                             mvq_src_p0[0].val[ch], mvq_src_p0[1].val[ch], mvq_src_p0[2].val[ch],
                                                             mvq_src_c[0].val[ch],  mvq_src_c[1].val[ch],  mvq_src_c[2].val[ch],
                                                             mvq_src_n0[0].val[ch], mvq_src_n0[1].val[ch], mvq_src_n0[2].val[ch],
                                                             mvq_src_n1[0].val[ch], mvq_src_n1[1].val[ch], mvq_src_n1[2].val[ch],
                                                             mvq_sum_lo.val[ch],    mvq_sum_hi.val[ch]);
            }

            neon::vstore(dst_c      + x, mvq_result);
            neon::vstore(sum_buffer + x, mvq_sum_lo);
            if (!ISF32)
            {
                neon::vstore(sum_buffer + x + (ELEM_COUNTS >> 1) * C, mvq_sum_hi);
            }
        }
    }

    // right border
    {
        MI_S32 x    = (width - ELEM_COUNTS) * C;
        MI_S32 last = (width - 1) * C;

        for (MI_S32 ch = 0; ch < C; ch++)
        {
            mvq_src_p1[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mvq_src_p1[1].val[ch], src_p1[last + ch], border_value[ch]);
            mvq_src_p0[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mvq_src_p0[1].val[ch], src_p0[last + ch], border_value[ch]);
            mvq_src_c[2].val[ch]  = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mvq_src_c[1].val[ch],  src_c[last + ch],  border_value[ch]);
            mvq_src_n0[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mvq_src_n0[1].val[ch], src_n0[last + ch], border_value[ch]);
            mvq_src_n1[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mvq_src_n1[1].val[ch], src_n1[last + ch], border_value[ch]);

            mvq_result.val[ch] = BoxFilter5x5SumCore<Tp>(mvq_src_p1[0].val[ch], mvq_src_p1[1].val[ch], mvq_src_p1[2].val[ch],
                                                         mvq_src_p0[0].val[ch], mvq_src_p0[1].val[ch], mvq_src_p0[2].val[ch],
                                                         mvq_src_c[0].val[ch],  mvq_src_c[1].val[ch],  mvq_src_c[2].val[ch],
                                                         mvq_src_n0[0].val[ch], mvq_src_n0[1].val[ch], mvq_src_n0[2].val[ch],
                                                         mvq_src_n1[0].val[ch], mvq_src_n1[1].val[ch], mvq_src_n1[2].val[ch],
                                                         mvq_sum_lo.val[ch],    mvq_sum_hi.val[ch]);
        }

        neon::vstore(dst_c      + x, mvq_result);
        neon::vstore(sum_buffer + x, mvq_sum_lo);
        if (!ISF32)
        {
            neon::vstore(sum_buffer + x + (ELEM_COUNTS >> 1) * C, mvq_sum_hi);
        }
    }
}

template <typename Tp, BorderType BORDER_TYPE, MI_S32 C>
static Status BoxFilter5x5NeonImpl(Context *ctx, const Mat &src, Mat &dst, const std::vector<Tp> &border_value,
                                   const Tp *border_buffer, ThreadBuffer &thread_buffer, MI_S32 start_row, MI_S32 end_row)
{
    const MI_S32 ksize = 5;
    const MI_S32 ksh = ksize >> 1;

    MI_S32 width  = dst.GetSizes().m_width;
    MI_S32 height = dst.GetSizes().m_height;

    using SumType = typename Promote<Tp>::Type;
    SumType *sum_buffer = thread_buffer.GetThreadData<SumType>();
    if (!sum_buffer)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    MI_S32 y = start_row;

    // top
    {
        const Tp *src_p1 = src.Ptr<Tp, BORDER_TYPE>(y - 2, border_buffer);
        const Tp *src_p0 = src.Ptr<Tp, BORDER_TYPE>(y - 1, border_buffer);
        const Tp *src_c  = src.Ptr<Tp>(y);
        const Tp *src_n0 = src.Ptr<Tp, BORDER_TYPE>(y + 1, border_buffer);
        const Tp *src_n1 = src.Ptr<Tp, BORDER_TYPE>(y + 2, border_buffer);
        Tp       *dst_c  = dst.Ptr<Tp>(y);
        BoxFilter5x5Row<Tp, BORDER_TYPE, C>(src_p1, src_p0, src_c, src_n0, src_n1, dst_c, width, sum_buffer, border_value);
        y++;
    }

    // middle
    for (; y < Min<MI_S32>(end_row, height - ksh); y++)
    {
        const Tp *src_p2 = src.Ptr<Tp, BORDER_TYPE>(y - 3, border_buffer);
        const Tp *src_n1 = src.Ptr<Tp>(y + 2);
        Tp       *dst_c  = dst.Ptr<Tp>(y);
        BoxFilter5x5Row<Tp, BORDER_TYPE, C>(src_p2, src_n1, dst_c, width, sum_buffer, border_value);
    }

    // bottom
    for (; y < end_row; y++)
    {
        const Tp *src_p2 = src.Ptr<Tp>(y - 3);
        const Tp *src_n1 = src.Ptr<Tp, BORDER_TYPE>(y + 2, border_buffer);
        Tp       *dst_c  = dst.Ptr<Tp>(y);
        BoxFilter5x5Row<Tp, BORDER_TYPE, C>(src_p2, src_n1, dst_c, width, sum_buffer, border_value);
    }

    return Status::OK;
}

template <typename Tp, BorderType BORDER_TYPE>
static Status BoxFilter5x5NeonHelper(Context *ctx, const Mat &src, Mat &dst,
                                     const std::vector<Tp> &border_value,
                                     const Tp *border_buffer, const OpTarget &target)
{
    AURA_UNUSED(target);
    Status ret = Status::ERROR;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return ret;
    }

    MI_S32 height  = dst.GetSizes().m_height;
    MI_S32 width   = dst.GetSizes().m_width;
    MI_S32 channel = dst.GetSizes().m_channel;

    using SumType = typename Promote<Tp>::Type;

    ThreadBuffer thread_buffer(ctx, sizeof(SumType) * width * channel);

    switch(channel)
    {
        case 1:
        {
            ret = wp->ParallelFor(0, height, BoxFilter5x5NeonImpl<Tp, BORDER_TYPE, 1>,
                                  ctx, std::cref(src), std::ref(dst), std::cref(border_value),
                                  border_buffer, std::ref(thread_buffer));
            break;
        }

        case 2:
        {
            ret = wp->ParallelFor(0, height, BoxFilter5x5NeonImpl<Tp, BORDER_TYPE, 2>,
                                  ctx, std::cref(src), std::ref(dst), std::cref(border_value),
                                  border_buffer, std::ref(thread_buffer));
            break;
        }

        case 3:
        {
            ret = wp->ParallelFor(0, height, BoxFilter5x5NeonImpl<Tp, BORDER_TYPE, 3>,
                                  ctx, std::cref(src), std::ref(dst), std::cref(border_value),
                                  border_buffer, std::ref(thread_buffer));
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
static Status BoxFilter5x5NeonHelper(Context *ctx, const Mat &src, Mat &dst, BorderType border_type,
                                     const Scalar &border_value, const OpTarget &target)
{
    Status ret = Status::ERROR;

    Tp *border_buffer = MI_NULL;
    std::vector<Tp> vec_border_value = border_value.ToVector<Tp>();

    MI_S32 width   = dst.GetSizes().m_width;
    MI_S32 channel = dst.GetSizes().m_channel;

    switch (border_type)
    {
        case BorderType::CONSTANT:
        {
            border_buffer = CreateBorderBuffer(ctx, width, channel, vec_border_value);
            if (MI_NULL == border_buffer)
            {
                AURA_ADD_ERROR_STRING(ctx, "CreateBorderBuffer failed");
                return Status::ERROR;
            }

            ret = BoxFilter5x5NeonHelper<Tp, BorderType::CONSTANT>(ctx, src, dst, vec_border_value, border_buffer, target);

            AURA_FREE(ctx, border_buffer);

            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "BoxFilter5x5NeonHelper<Tp, BorderType::CONSTANT> failed");
            }
            break;
        }

        case BorderType::REPLICATE:
        {
            ret = BoxFilter5x5NeonHelper<Tp, BorderType::REPLICATE>(ctx, src, dst, vec_border_value, border_buffer, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "BoxFilter5x5NeonHelper<Tp, BorderType::REPLICATE> failed");
            }
            break;
        }

        case BorderType::REFLECT_101:
        {
            ret = BoxFilter5x5NeonHelper<Tp, BorderType::REFLECT_101>(ctx, src, dst, vec_border_value, border_buffer, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "BoxFilter5x5NeonHelper<Tp, BorderType::REFLECT_101> failed");
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

Status BoxFilter5x5Neon(Context *ctx, const Mat &src, Mat &dst, BorderType border_type,
                        const Scalar &border_value, const OpTarget &target)
{
    Status ret = Status::ERROR;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = BoxFilter5x5NeonHelper<MI_U8>(ctx, src, dst, border_type, border_value, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "BoxFilter5x5NeonHelper<MI_U8> failed");
            }
            break;
        }

        case ElemType::S8:
        {
            ret = BoxFilter5x5NeonHelper<MI_S8>(ctx, src, dst, border_type, border_value, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "BoxFilter5x5NeonHelper<MI_S8> failed");
            }
            break;
        }

        case ElemType::U16:
        {
            ret = BoxFilter5x5NeonHelper<MI_U16>(ctx, src, dst, border_type, border_value, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "BoxFilter5x5NeonHelper<MI_U16> failed");
            }
            break;
        }

        case ElemType::S16:
        {
            ret = BoxFilter5x5NeonHelper<MI_S16>(ctx, src, dst, border_type, border_value, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "BoxFilter5x5NeonHelper<MI_S16> failed");
            }
            break;
        }

#if defined(AURA_ENABLE_NEON_FP16)
        case ElemType::F16:
        {
            ret = BoxFilter5x5NeonHelper<MI_F16>(ctx, src, dst, border_type, border_value, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "BoxFilter5x5NeonHelper<MI_F16> failed");
            }
            break;
        }
#endif // AURA_ENABLE_NEON_FP16

        case ElemType::F32:
        {
            ret = BoxFilter5x5NeonHelper<MI_F32>(ctx, src, dst, border_type, border_value, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "BoxFilter5x5NeonHelper<MI_F32> failed");
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