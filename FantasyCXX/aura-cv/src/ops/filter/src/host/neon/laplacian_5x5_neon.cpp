#include "laplacian_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

template <typename Tp>
AURA_ALWAYS_INLINE DT_VOID Laplacian5x5Prepare(Tp *mv_src_p1, Tp *mv_src_p0, Tp *mv_src_c,
                                               Tp *mv_src_n0, Tp *mv_src_n1)
{
    mv_src_p1[0] = mv_src_p1[1];
    mv_src_p0[0] = mv_src_p0[1];
    mv_src_c[0]  = mv_src_c[1];
    mv_src_n0[0] = mv_src_n0[1];
    mv_src_n1[0] = mv_src_n1[1];

    mv_src_p1[1] = mv_src_p1[2];
    mv_src_p0[1] = mv_src_p0[2];
    mv_src_c[1]  = mv_src_c[2];
    mv_src_n0[1] = mv_src_n0[2];
    mv_src_n1[1] = mv_src_n1[2];
}

template <typename Tp>
AURA_ALWAYS_INLINE DT_VOID Laplacian5x5Prepare(Tp *mv_src_p1, Tp *mv_src_p0, Tp *mv_src_c0,
                                               Tp *mv_src_c1, Tp *mv_src_n0, Tp *mv_src_n1)
{
    mv_src_p1[0] = mv_src_p1[1];
    mv_src_p0[0] = mv_src_p0[1];
    mv_src_c0[0] = mv_src_c0[1];
    mv_src_c1[0] = mv_src_c1[1];
    mv_src_n0[0] = mv_src_n0[1];
    mv_src_n1[0] = mv_src_n1[1];

    mv_src_p1[1] = mv_src_p1[2];
    mv_src_p0[1] = mv_src_p0[2];
    mv_src_c0[1] = mv_src_c0[2];
    mv_src_c1[1] = mv_src_c1[2];
    mv_src_n0[1] = mv_src_n0[2];
    mv_src_n1[1] = mv_src_n1[2];
}

// d16x4_t = uint16x4_t, int16x4_t
template <typename d16x4_t, typename std::enable_if<std::is_same<d16x4_t, uint16x4_t>::value ||
                                                    std::is_same<d16x4_t, int16x4_t>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE d16x4_t Laplacian5x5Core(d16x4_t &vd16_src_p1x0, d16x4_t &vd16_src_p1x1, d16x4_t &vd16_src_p1x2,
                                            d16x4_t &vd16_src_p0x0, d16x4_t &vd16_src_p0x1, d16x4_t &vd16_src_p0x2,
                                            d16x4_t &vd16_src_cx0,  d16x4_t &vd16_src_cx1,  d16x4_t &vd16_src_cx2,
                                            d16x4_t &vd16_src_n0x0, d16x4_t &vd16_src_n0x1, d16x4_t &vd16_src_n0x2,
                                            d16x4_t &vd16_src_n1x0, d16x4_t &vd16_src_n1x1, d16x4_t &vd16_src_n1x2)
{
    using D32     = typename Promote<typename neon::Scalar<d16x4_t>::SType>::Type;
    using d32x4_t = typename neon::QVector<D32>::VType;

    // p1n1
    d16x4_t vd16_src_p1l1         = neon::vext<2>(vd16_src_p1x0, vd16_src_p1x1);
    d16x4_t vd16_src_p1l0         = neon::vext<3>(vd16_src_p1x0, vd16_src_p1x1);
    d16x4_t vd16_src_p1r0         = neon::vext<1>(vd16_src_p1x1, vd16_src_p1x2);
    d16x4_t vd16_src_p1r1         = neon::vext<2>(vd16_src_p1x1, vd16_src_p1x2);

    d32x4_t vq32_sum_p1_l1r1      = neon::vaddl(vd16_src_p1l1, vd16_src_p1r1);
    d32x4_t vq32_sum_p1_l0r0      = neon::vaddl(vd16_src_p1l0, vd16_src_p1r0);

    d16x4_t vd16_src_n1l1         = neon::vext<2>(vd16_src_n1x0, vd16_src_n1x1);
    d16x4_t vd16_src_n1l0         = neon::vext<3>(vd16_src_n1x0, vd16_src_n1x1);
    d16x4_t vd16_src_n1r0         = neon::vext<1>(vd16_src_n1x1, vd16_src_n1x2);
    d16x4_t vd16_src_n1r1         = neon::vext<2>(vd16_src_n1x1, vd16_src_n1x2);

    d32x4_t vq32_sum_n1_l1r1      = neon::vaddl(vd16_src_n1l1, vd16_src_n1r1);
    d32x4_t vq32_sum_n1_l0r0      = neon::vaddl(vd16_src_n1l0, vd16_src_n1r0);

    d32x4_t vq32_sum_c_p1n1       = neon::vaddl(vd16_src_p1x1, vd16_src_n1x1);
    d32x4_t vq32_sum_p1n1_l1r1    = neon::vadd(vq32_sum_p1_l1r1, vq32_sum_n1_l1r1);
    d32x4_t vq32_sum_p1n1_l0r0    = neon::vadd(vq32_sum_p1_l0r0, vq32_sum_n1_l0r0);

    d32x4_t vq32_sum_p1n1         = neon::vmul(vq32_sum_c_p1n1, static_cast<D32>(4));
    vq32_sum_p1n1                 = neon::vmla(vq32_sum_p1n1, vq32_sum_p1n1_l1r1, static_cast<D32>(2));
    vq32_sum_p1n1                 = neon::vmla(vq32_sum_p1n1, vq32_sum_p1n1_l0r0, static_cast<D32>(4));

    // p0n0
    d16x4_t vd16_src_p0l1         = neon::vext<2>(vd16_src_p0x0, vd16_src_p0x1);
    d16x4_t vd16_src_p0r1         = neon::vext<2>(vd16_src_p0x1, vd16_src_p0x2);

    d32x4_t vq32_sum_p0_l1r1      = neon::vaddl(vd16_src_p0l1, vd16_src_p0r1);

    d16x4_t vd16_src_n0l1         = neon::vext<2>(vd16_src_n0x0, vd16_src_n0x1);
    d16x4_t vd16_src_n0r1         = neon::vext<2>(vd16_src_n0x1, vd16_src_n0x2);

    d32x4_t vq32_sum_n0_l1r1      = neon::vaddl(vd16_src_n0l1, vd16_src_n0r1);

    d32x4_t vq32_sum_c_p0n0       = neon::vaddl(vd16_src_p0x1, vd16_src_n0x1);
    d32x4_t vq32_sum_p0n0_l1r1    = neon::vadd(vq32_sum_p0_l1r1, vq32_sum_n0_l1r1);

    d32x4_t vq32_sum_neg_p0n0     = neon::vmul(vq32_sum_c_p0n0, static_cast<D32>(8));
    d32x4_t vq32_sum_pos_p0n0     = neon::vmul(vq32_sum_p0n0_l1r1, static_cast<D32>(4));

    // c
    d16x4_t vd16_src_cl1          = neon::vext<2>(vd16_src_cx0, vd16_src_cx1);
    d16x4_t vd16_src_cl0          = neon::vext<3>(vd16_src_cx0, vd16_src_cx1);
    d16x4_t vd16_src_cr0          = neon::vext<1>(vd16_src_cx1, vd16_src_cx2);
    d16x4_t vd16_src_cr1          = neon::vext<2>(vd16_src_cx1, vd16_src_cx2);

    d32x4_t vq32_sum_c_l1r1       = neon::vaddl(vd16_src_cl1, vd16_src_cr1);
    d32x4_t vq32_sum_c_l0r0       = neon::vaddl(vd16_src_cl0, vd16_src_cr0);

    d32x4_t vq32_sum_neg_c        = neon::vmul(neon::vmovl(vd16_src_cx1), static_cast<D32>(24));
    d32x4_t vq32_sum_pos_c        = neon::vmul(vq32_sum_c_l1r1, static_cast<D32>(4));
    vq32_sum_neg_c                = neon::vmla(vq32_sum_neg_c, vq32_sum_c_l0r0, static_cast<D32>(8));

    // sum
    d32x4_t vq32_sum_pos          = neon::vadd(neon::vadd(vq32_sum_p1n1, vq32_sum_pos_p0n0), vq32_sum_pos_c);
    d32x4_t vq32_sum_neg          = neon::vadd(vq32_sum_neg_p0n0, vq32_sum_neg_c);

    return neon::vqmovn(neon::vqsub(vq32_sum_pos, vq32_sum_neg));
}

AURA_ALWAYS_INLINE int16x8_t Laplacian5x5Core(uint8x8_t &vdu8_src_p1x0, uint8x8_t &vdu8_src_p1x1, uint8x8_t &vdu8_src_p1x2,
                                              uint8x8_t &vdu8_src_p0x0, uint8x8_t &vdu8_src_p0x1, uint8x8_t &vdu8_src_p0x2,
                                              uint8x8_t &vdu8_src_cx0,  uint8x8_t &vdu8_src_cx1,  uint8x8_t &vdu8_src_cx2,
                                              uint8x8_t &vdu8_src_n0x0, uint8x8_t &vdu8_src_n0x1, uint8x8_t &vdu8_src_n0x2,
                                              uint8x8_t &vdu8_src_n1x0, uint8x8_t &vdu8_src_n1x1, uint8x8_t &vdu8_src_n1x2)
{
    // p1n1
    uint8x8_t vdu8_src_p1l1       = neon::vext<6>(vdu8_src_p1x0, vdu8_src_p1x1);
    uint8x8_t vdu8_src_p1l0       = neon::vext<7>(vdu8_src_p1x0, vdu8_src_p1x1);
    uint8x8_t vdu8_src_p1r0       = neon::vext<1>(vdu8_src_p1x1, vdu8_src_p1x2);
    uint8x8_t vdu8_src_p1r1       = neon::vext<2>(vdu8_src_p1x1, vdu8_src_p1x2);

    uint16x8_t vqu16_sum_p1_l1r1  = neon::vaddl(vdu8_src_p1l1, vdu8_src_p1r1);
    uint16x8_t vqu16_sum_p1_l0r0  = neon::vaddl(vdu8_src_p1l0, vdu8_src_p1r0);

    uint8x8_t vdu8_src_n1l1       = neon::vext<6>(vdu8_src_n1x0, vdu8_src_n1x1);
    uint8x8_t vdu8_src_n1l0       = neon::vext<7>(vdu8_src_n1x0, vdu8_src_n1x1);
    uint8x8_t vdu8_src_n1r0       = neon::vext<1>(vdu8_src_n1x1, vdu8_src_n1x2);
    uint8x8_t vdu8_src_n1r1       = neon::vext<2>(vdu8_src_n1x1, vdu8_src_n1x2);

    uint16x8_t vqu16_sum_n1_l1r1  = neon::vaddl(vdu8_src_n1l1, vdu8_src_n1r1);
    uint16x8_t vqu16_sum_n1_l0r0  = neon::vaddl(vdu8_src_n1l0, vdu8_src_n1r0);

    int16x8_t vqs16_sum_c_p1n1    = neon::vreinterpret(neon::vaddl(vdu8_src_p1x1, vdu8_src_n1x1));
    int16x8_t vqs16_sum_p1n1_l1r1 = neon::vreinterpret(neon::vadd(vqu16_sum_p1_l1r1, vqu16_sum_n1_l1r1));
    int16x8_t vqs16_sum_p1n1_l0r0 = neon::vreinterpret(neon::vadd(vqu16_sum_p1_l0r0, vqu16_sum_n1_l0r0));

    int16x8_t vqs16_sum_p1n1      = neon::vmul(vqs16_sum_c_p1n1, static_cast<DT_S16>(4));
    vqs16_sum_p1n1                = neon::vmla(vqs16_sum_p1n1, vqs16_sum_p1n1_l1r1, static_cast<DT_S16>(2));
    vqs16_sum_p1n1                = neon::vmla(vqs16_sum_p1n1, vqs16_sum_p1n1_l0r0, static_cast<DT_S16>(4));

    // p0n0
    uint8x8_t vdu8_src_p0l1       = neon::vext<6>(vdu8_src_p0x0, vdu8_src_p0x1);
    uint8x8_t vdu8_src_p0r1       = neon::vext<2>(vdu8_src_p0x1, vdu8_src_p0x2);
    uint16x8_t vqu16_sum_p0_l1r1  = neon::vaddl(vdu8_src_p0l1, vdu8_src_p0r1);

    uint8x8_t vdu8_src_n0l1       = neon::vext<6>(vdu8_src_n0x0, vdu8_src_n0x1);
    uint8x8_t vdu8_src_n0r1       = neon::vext<2>(vdu8_src_n0x1, vdu8_src_n0x2);
    uint16x8_t vqu16_sum_n0_l1r1  = neon::vaddl(vdu8_src_n0l1, vdu8_src_n0r1);

    int16x8_t vqs16_sum_c_p0n0    = neon::vreinterpret(neon::vaddl(vdu8_src_p0x1, vdu8_src_n0x1));
    int16x8_t vqs16_sum_p0n0_l1r1 = neon::vreinterpret(neon::vadd(vqu16_sum_p0_l1r1, vqu16_sum_n0_l1r1));

    int16x8_t vqs16_sum_p0n0      = neon::vmul(vqs16_sum_c_p0n0, static_cast<DT_S16>(-8));
    vqs16_sum_p0n0                = neon::vmla(vqs16_sum_p0n0, vqs16_sum_p0n0_l1r1, static_cast<DT_S16>(4));

    // c
    uint8x8_t vdu8_src_cl1        = neon::vext<6>(vdu8_src_cx0, vdu8_src_cx1);
    uint8x8_t vdu8_src_cl0        = neon::vext<7>(vdu8_src_cx0, vdu8_src_cx1);
    uint8x8_t vdu8_src_cr0        = neon::vext<1>(vdu8_src_cx1, vdu8_src_cx2);
    uint8x8_t vdu8_src_cr1        = neon::vext<2>(vdu8_src_cx1, vdu8_src_cx2);

    uint16x8_t vqu16_sum_c_l1r1   = neon::vaddl(vdu8_src_cl1, vdu8_src_cr1);
    uint16x8_t vqu16_sum_c_l0r0   = neon::vaddl(vdu8_src_cl0, vdu8_src_cr0);

    int16x8_t vqs16_sum_c_c       = neon::vreinterpret(neon::vmovl(vdu8_src_cx1));
    int16x8_t vqs16_sum_c_l1r1    = neon::vreinterpret(vqu16_sum_c_l1r1);
    int16x8_t vqs16_sum_c_l0r0    = neon::vreinterpret(vqu16_sum_c_l0r0);

    int16x8_t vqs16_sum_c         = neon::vmul(vqs16_sum_c_c, static_cast<DT_S16>(-24));
    vqs16_sum_c                   = neon::vmla(vqs16_sum_c, vqs16_sum_c_l1r1, static_cast<DT_S16>(4));
    vqs16_sum_c                   = neon::vmla(vqs16_sum_c, vqs16_sum_c_l0r0, static_cast<DT_S16>(-8));

    // sum
    int16x8_t vqs16_sum_pn        = neon::vadd(vqs16_sum_p1n1, vqs16_sum_p0n0);
    int16x8_t vqs16_result        = neon::vadd(vqs16_sum_pn,       vqs16_sum_c);

    return vqs16_result;
}

#if defined(AURA_ENABLE_NEON_FP16)
AURA_ALWAYS_INLINE float16x4_t Laplacian5x5Core(float16x4_t &vdf16_src_p1x0, float16x4_t &vdf16_src_p1x1, float16x4_t &vdf16_src_p1x2,
                                                float16x4_t &vdf16_src_p0x0, float16x4_t &vdf16_src_p0x1, float16x4_t &vdf16_src_p0x2,
                                                float16x4_t &vdf16_src_cx0,  float16x4_t &vdf16_src_cx1,  float16x4_t &vdf16_src_cx2,
                                                float16x4_t &vdf16_src_n0x0, float16x4_t &vdf16_src_n0x1, float16x4_t &vdf16_src_n0x2,
                                                float16x4_t &vdf16_src_n1x0, float16x4_t &vdf16_src_n1x1, float16x4_t &vdf16_src_n1x2)
{
    // p1n1
    float32x4_t vqf32_src_p1x0      = neon::vcvt<DT_F32>(vdf16_src_p1x0);
    float32x4_t vqf32_src_p1x1      = neon::vcvt<DT_F32>(vdf16_src_p1x1);
    float32x4_t vqf32_src_p1x2      = neon::vcvt<DT_F32>(vdf16_src_p1x2);
    float32x4_t vqf32_src_n1x0      = neon::vcvt<DT_F32>(vdf16_src_n1x0);
    float32x4_t vqf32_src_n1x1      = neon::vcvt<DT_F32>(vdf16_src_n1x1);
    float32x4_t vqf32_src_n1x2      = neon::vcvt<DT_F32>(vdf16_src_n1x2);
    float32x4_t vqf32_src_p1l1      = neon::vext<2>(vqf32_src_p1x0, vqf32_src_p1x1);
    float32x4_t vqf32_src_p1l0      = neon::vext<3>(vqf32_src_p1x0, vqf32_src_p1x1);
    float32x4_t vqf32_src_p1r0      = neon::vext<1>(vqf32_src_p1x1, vqf32_src_p1x2);
    float32x4_t vqf32_src_p1r1      = neon::vext<2>(vqf32_src_p1x1, vqf32_src_p1x2);

    float32x4_t vqf32_sum_p1_l1r1   = neon::vadd(vqf32_src_p1l1, vqf32_src_p1r1);
    float32x4_t vqf32_sum_p1_l0r0   = neon::vadd(vqf32_src_p1l0, vqf32_src_p1r0);

    float32x4_t vqf32_src_n1l1      = neon::vext<2>(vqf32_src_n1x0, vqf32_src_n1x1);
    float32x4_t vqf32_src_n1l0      = neon::vext<3>(vqf32_src_n1x0, vqf32_src_n1x1);
    float32x4_t vqf32_src_n1r0      = neon::vext<1>(vqf32_src_n1x1, vqf32_src_n1x2);
    float32x4_t vqf32_src_n1r1      = neon::vext<2>(vqf32_src_n1x1, vqf32_src_n1x2);

    float32x4_t vqf32_sum_n1_l1r1   = neon::vadd(vqf32_src_n1l1, vqf32_src_n1r1);
    float32x4_t vqf32_sum_n1_l0r0   = neon::vadd(vqf32_src_n1l0, vqf32_src_n1r0);

    float32x4_t vqf32_sum_c_p1n1    = neon::vadd(vqf32_src_p1x1,    vqf32_src_n1x1);
    float32x4_t vqf32_sum_p1n1_l1r1 = neon::vadd(vqf32_sum_p1_l1r1, vqf32_sum_n1_l1r1);
    float32x4_t vqf32_sum_p1n1_l0r0 = neon::vadd(vqf32_sum_p1_l0r0, vqf32_sum_n1_l0r0);

    float32x4_t vqf32_sum_p1n1      = neon::vmul(vqf32_sum_c_p1n1, 4.f);
    vqf32_sum_p1n1                  = neon::vmla(vqf32_sum_p1n1, vqf32_sum_p1n1_l1r1, 2.f);
    vqf32_sum_p1n1                  = neon::vmla(vqf32_sum_p1n1, vqf32_sum_p1n1_l0r0, 4.f);

    // p0n0
    float32x4_t vqf32_src_p0x0      = neon::vcvt<DT_F32>(vdf16_src_p0x0);
    float32x4_t vqf32_src_p0x1      = neon::vcvt<DT_F32>(vdf16_src_p0x1);
    float32x4_t vqf32_src_p0x2      = neon::vcvt<DT_F32>(vdf16_src_p0x2);
    float32x4_t vqf32_src_n0x0      = neon::vcvt<DT_F32>(vdf16_src_n0x0);
    float32x4_t vqf32_src_n0x1      = neon::vcvt<DT_F32>(vdf16_src_n0x1);
    float32x4_t vqf32_src_n0x2      = neon::vcvt<DT_F32>(vdf16_src_n0x2);
    float32x4_t vqf32_src_p0l1      = neon::vext<2>(vqf32_src_p0x0, vqf32_src_p0x1);
    float32x4_t vqf32_src_p0r1      = neon::vext<2>(vqf32_src_p0x1, vqf32_src_p0x2);

    float32x4_t vqf32_sum_p0_l1r1   = neon::vadd(vqf32_src_p0l1, vqf32_src_p0r1);

    float32x4_t vqf32_src_n0l1      = neon::vext<2>(vqf32_src_n0x0, vqf32_src_n0x1);
    float32x4_t vqf32_src_n0r1      = neon::vext<2>(vqf32_src_n0x1, vqf32_src_n0x2);

    float32x4_t vqf32_sum_n0_l1r1   = neon::vadd(vqf32_src_n0l1, vqf32_src_n0r1);

    float32x4_t vqf32_sum_c_p0n0    = neon::vadd(vqf32_src_p0x1,    vqf32_src_n0x1);
    float32x4_t vqf32_sum_p0n0_l1r1 = neon::vadd(vqf32_sum_p0_l1r1, vqf32_sum_n0_l1r1);

    float32x4_t vqf32_sum_p0n0      = neon::vmul(vqf32_sum_c_p0n0, -8.f);
    vqf32_sum_p0n0                  = neon::vmla(vqf32_sum_p0n0, vqf32_sum_p0n0_l1r1, 4.f);

    // c
    float32x4_t vqf32_src_cx0       = neon::vcvt<DT_F32>(vdf16_src_cx0);
    float32x4_t vqf32_src_cx1       = neon::vcvt<DT_F32>(vdf16_src_cx1);
    float32x4_t vqf32_src_cx2       = neon::vcvt<DT_F32>(vdf16_src_cx2);
    float32x4_t vqf32_src_cl1       = neon::vext<2>(vqf32_src_cx0, vqf32_src_cx1);
    float32x4_t vqf32_src_cl0       = neon::vext<3>(vqf32_src_cx0, vqf32_src_cx1);
    float32x4_t vqf32_src_cr0       = neon::vext<1>(vqf32_src_cx1, vqf32_src_cx2);
    float32x4_t vqf32_src_cr1       = neon::vext<2>(vqf32_src_cx1, vqf32_src_cx2);

    float32x4_t vqf32_sum_c_l1r1    = neon::vadd(vqf32_src_cl1, vqf32_src_cr1);
    float32x4_t vqf32_sum_c_l0r0    = neon::vadd(vqf32_src_cl0, vqf32_src_cr0);

    float32x4_t vqf32_sum_c         = neon::vmul(vqf32_src_cx1, -24.f);
    vqf32_sum_c                     = neon::vmla(vqf32_sum_c, vqf32_sum_c_l1r1, 4.f);
    vqf32_sum_c                     = neon::vmla(vqf32_sum_c, vqf32_sum_c_l0r0, -8.f);

    // sum
    float32x4_t vqf32_sum_pn        = neon::vadd(vqf32_sum_p1n1, vqf32_sum_p0n0);
    float32x4_t vqf32_result        = neon::vadd(vqf32_sum_pn,       vqf32_sum_c);

    return neon::vcvt<MI_F16>(vqf32_result);
}
#endif // AURA_ENABLE_NEON_FP16

AURA_ALWAYS_INLINE float32x4_t Laplacian5x5Core(float32x4_t &vqf32_src_p1x0, float32x4_t &vqf32_src_p1x1, float32x4_t &vqf32_src_p1x2,
                                                float32x4_t &vqf32_src_p0x0, float32x4_t &vqf32_src_p0x1, float32x4_t &vqf32_src_p0x2,
                                                float32x4_t &vqf32_src_cx0,  float32x4_t &vqf32_src_cx1,  float32x4_t &vqf32_src_cx2,
                                                float32x4_t &vqf32_src_n0x0, float32x4_t &vqf32_src_n0x1, float32x4_t &vqf32_src_n0x2,
                                                float32x4_t &vqf32_src_n1x0, float32x4_t &vqf32_src_n1x1, float32x4_t &vqf32_src_n1x2)
{
    // p1n1
    float32x4_t vqf32_src_p1l1      = neon::vext<2>(vqf32_src_p1x0, vqf32_src_p1x1);
    float32x4_t vqf32_src_p1l0      = neon::vext<3>(vqf32_src_p1x0, vqf32_src_p1x1);
    float32x4_t vqf32_src_p1r0      = neon::vext<1>(vqf32_src_p1x1, vqf32_src_p1x2);
    float32x4_t vqf32_src_p1r1      = neon::vext<2>(vqf32_src_p1x1, vqf32_src_p1x2);

    float32x4_t vqf32_sum_p1_l1r1   = neon::vadd(vqf32_src_p1l1, vqf32_src_p1r1);
    float32x4_t vqf32_sum_p1_l0r0   = neon::vadd(vqf32_src_p1l0, vqf32_src_p1r0);

    float32x4_t vqf32_src_n1l1      = neon::vext<2>(vqf32_src_n1x0, vqf32_src_n1x1);
    float32x4_t vqf32_src_n1l0      = neon::vext<3>(vqf32_src_n1x0, vqf32_src_n1x1);
    float32x4_t vqf32_src_n1r0      = neon::vext<1>(vqf32_src_n1x1, vqf32_src_n1x2);
    float32x4_t vqf32_src_n1r1      = neon::vext<2>(vqf32_src_n1x1, vqf32_src_n1x2);

    float32x4_t vqf32_sum_n1_l1r1   = neon::vadd(vqf32_src_n1l1, vqf32_src_n1r1);
    float32x4_t vqf32_sum_n1_l0r0   = neon::vadd(vqf32_src_n1l0, vqf32_src_n1r0);

    float32x4_t vqf32_sum_c_p1n1    = neon::vadd(vqf32_src_p1x1,    vqf32_src_n1x1);
    float32x4_t vqf32_sum_p1n1_l1r1 = neon::vadd(vqf32_sum_p1_l1r1, vqf32_sum_n1_l1r1);
    float32x4_t vqf32_sum_p1n1_l0r0 = neon::vadd(vqf32_sum_p1_l0r0, vqf32_sum_n1_l0r0);

    float32x4_t vqf32_sum_p1n1      = neon::vmul(vqf32_sum_c_p1n1, 4.f);
    vqf32_sum_p1n1                  = neon::vmla(vqf32_sum_p1n1, vqf32_sum_p1n1_l1r1, 2.f);
    vqf32_sum_p1n1                  = neon::vmla(vqf32_sum_p1n1, vqf32_sum_p1n1_l0r0, 4.f);

    // p0n0
    float32x4_t vqf32_src_p0l1      = neon::vext<2>(vqf32_src_p0x0, vqf32_src_p0x1);
    float32x4_t vqf32_src_p0r1      = neon::vext<2>(vqf32_src_p0x1, vqf32_src_p0x2);

    float32x4_t vqf32_sum_p0_l1r1   = neon::vadd(vqf32_src_p0l1, vqf32_src_p0r1);

    float32x4_t vqf32_src_n0l1      = neon::vext<2>(vqf32_src_n0x0, vqf32_src_n0x1);
    float32x4_t vqf32_src_n0r1      = neon::vext<2>(vqf32_src_n0x1, vqf32_src_n0x2);

    float32x4_t vqf32_sum_n0_l1r1   = neon::vadd(vqf32_src_n0l1, vqf32_src_n0r1);

    float32x4_t vqf32_sum_c_p0n0    = neon::vadd(vqf32_src_p0x1,    vqf32_src_n0x1);
    float32x4_t vqf32_sum_p0n0_l1r1 = neon::vadd(vqf32_sum_p0_l1r1, vqf32_sum_n0_l1r1);

    float32x4_t vqf32_sum_p0n0      = neon::vmul(vqf32_sum_c_p0n0, -8.f);
    vqf32_sum_p0n0                  = neon::vmla(vqf32_sum_p0n0, vqf32_sum_p0n0_l1r1, 4.f);

    // c
    float32x4_t vqf32_src_cl1       = neon::vext<2>(vqf32_src_cx0, vqf32_src_cx1);
    float32x4_t vqf32_src_cl0       = neon::vext<3>(vqf32_src_cx0, vqf32_src_cx1);
    float32x4_t vqf32_src_cr0       = neon::vext<1>(vqf32_src_cx1, vqf32_src_cx2);
    float32x4_t vqf32_src_cr1       = neon::vext<2>(vqf32_src_cx1, vqf32_src_cx2);

    float32x4_t vqf32_sum_c_l1r1    = neon::vadd(vqf32_src_cl1, vqf32_src_cr1);
    float32x4_t vqf32_sum_c_l0r0    = neon::vadd(vqf32_src_cl0, vqf32_src_cr0);

    float32x4_t vqf32_sum_c         = neon::vmul(vqf32_src_cx1, -24.f);
    vqf32_sum_c                     = neon::vmla(vqf32_sum_c, vqf32_sum_c_l1r1, 4.f);
    vqf32_sum_c                     = neon::vmla(vqf32_sum_c, vqf32_sum_c_l0r0, -8.f);

    // sum
    float32x4_t vqf32_sum_pn        = neon::vadd(vqf32_sum_p1n1, vqf32_sum_p0n0);
    float32x4_t vqf32_result        = neon::vadd(vqf32_sum_pn,       vqf32_sum_c);

    return vqf32_result;
}

template <typename St, typename Dt, BorderType BORDER_TYPE, DT_S32 C>
static DT_VOID Laplacian5x5OneRow(const St *src_p1, const St *src_p0, const St *src_c, const St *src_n0,
                                  const St *src_n1, Dt *dst, const std::vector<St> &border_value, DT_S32 width)
{
    using MVSt = typename std::conditional<std::is_same<St, DT_F32>::value, typename neon::MQVector<St, C>::MVType,
                                                typename neon::MDVector<St, C>::MVType>::type;
    using MVDt = typename std::conditional<std::is_same<St, DT_U8>::value, typename neon::MQVector<Dt, C>::MVType, MVSt>::type;

    constexpr DT_S32 ELEM_COUNTS = static_cast<DT_S32>(sizeof(MVDt) / C / sizeof(Dt));
    constexpr DT_S32 VOFFSET     = ELEM_COUNTS * C;
    const DT_S32 width_align     = (width & -ELEM_COUNTS) * C;

    MVSt mv_src_p1[3], mv_src_p0[3], mv_src_c[3], mv_src_n0[3], mv_src_n1[3];
    MVDt mv_result;

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
            mv_result.val[ch]    = Laplacian5x5Core(mv_src_p1[0].val[ch], mv_src_p1[1].val[ch], mv_src_p1[2].val[ch],
                                                    mv_src_p0[0].val[ch], mv_src_p0[1].val[ch], mv_src_p0[2].val[ch],
                                                    mv_src_c[0].val[ch],  mv_src_c[1].val[ch],  mv_src_c[2].val[ch],
                                                    mv_src_n0[0].val[ch], mv_src_n0[1].val[ch], mv_src_n0[2].val[ch],
                                                    mv_src_n1[0].val[ch], mv_src_n1[1].val[ch], mv_src_n1[2].val[ch]);
        }
        neon::vstore(dst, mv_result);

        Laplacian5x5Prepare(mv_src_p1, mv_src_p0, mv_src_c, mv_src_n0, mv_src_n1);
    }

    // middle
    {
        for (DT_S32 x = VOFFSET; x < width_align - VOFFSET; x += VOFFSET)
        {
            neon::vload(src_p1 + x + VOFFSET, mv_src_p1[2]);
            neon::vload(src_p0 + x + VOFFSET, mv_src_p0[2]);
            neon::vload(src_c  + x + VOFFSET, mv_src_c[2]);
            neon::vload(src_n0 + x + VOFFSET, mv_src_n0[2]);
            neon::vload(src_n1 + x + VOFFSET, mv_src_n1[2]);

            for (DT_S32 ch = 0; ch < C; ch++)
            {
                mv_result.val[ch] = Laplacian5x5Core(mv_src_p1[0].val[ch], mv_src_p1[1].val[ch], mv_src_p1[2].val[ch],
                                                     mv_src_p0[0].val[ch], mv_src_p0[1].val[ch], mv_src_p0[2].val[ch],
                                                     mv_src_c[0].val[ch],  mv_src_c[1].val[ch],  mv_src_c[2].val[ch],
                                                     mv_src_n0[0].val[ch], mv_src_n0[1].val[ch], mv_src_n0[2].val[ch],
                                                     mv_src_n1[0].val[ch], mv_src_n1[1].val[ch], mv_src_n1[2].val[ch]);
            }
            neon::vstore(dst + x, mv_result);

            Laplacian5x5Prepare(mv_src_p1, mv_src_p0, mv_src_c, mv_src_n0, mv_src_n1);
        }
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
                mv_result.val[ch] = Laplacian5x5Core(mv_src_p1[0].val[ch], mv_src_p1[1].val[ch], mv_src_p1[2].val[ch],
                                                     mv_src_p0[0].val[ch], mv_src_p0[1].val[ch], mv_src_p0[2].val[ch],
                                                     mv_src_c[0].val[ch],  mv_src_c[1].val[ch],  mv_src_c[2].val[ch],
                                                     mv_src_n0[0].val[ch], mv_src_n0[1].val[ch], mv_src_n0[2].val[ch],
                                                     mv_src_n1[0].val[ch], mv_src_n1[1].val[ch], mv_src_n1[2].val[ch]);
            }
            neon::vstore(dst + x, mv_result);

            Laplacian5x5Prepare(mv_src_p1, mv_src_p0, mv_src_c, mv_src_n0, mv_src_n1);
        }
    }

    // right
    {
        DT_S32 x    = (width - ELEM_COUNTS) * C;
        DT_S32 last = (width - 1) * C;

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            mv_src_p1[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mv_src_p1[1].val[ch], src_p1[last], border_value[ch]);
            mv_src_p0[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mv_src_p0[1].val[ch], src_p0[last], border_value[ch]);
            mv_src_c[2].val[ch]  = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mv_src_c[1].val[ch],  src_c[last],  border_value[ch]);
            mv_src_n0[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mv_src_n0[1].val[ch], src_n0[last], border_value[ch]);
            mv_src_n1[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mv_src_n1[1].val[ch], src_n1[last], border_value[ch]);
            mv_result.val[ch]    = Laplacian5x5Core(mv_src_p1[0].val[ch], mv_src_p1[1].val[ch], mv_src_p1[2].val[ch],
                                                    mv_src_p0[0].val[ch], mv_src_p0[1].val[ch], mv_src_p0[2].val[ch],
                                                    mv_src_c[0].val[ch],  mv_src_c[1].val[ch],  mv_src_c[2].val[ch],
                                                    mv_src_n0[0].val[ch], mv_src_n0[1].val[ch], mv_src_n0[2].val[ch],
                                                    mv_src_n1[0].val[ch], mv_src_n1[1].val[ch], mv_src_n1[2].val[ch]);
            last++;
        }
        neon::vstore(dst + x, mv_result);
    }
}

template <typename St, typename Dt, BorderType BORDER_TYPE, DT_S32 C>
static DT_VOID Laplacian5x5TwoRow(const St *src_p1, const St *src_p0, const St *src_c0, const St *src_c1,
                                  const St *src_n0, const St *src_n1, Dt *dst_c0, Dt *dst_c1,
                                  const std::vector<St> &border_value, DT_S32 width)
{
    using MVSt = typename std::conditional<std::is_same<St, DT_F32>::value, typename neon::MQVector<St, C>::MVType,
                                                                            typename neon::MDVector<St, C>::MVType>::type;
    using MVDt = typename std::conditional<std::is_same<St, DT_U8>::value,  typename neon::MQVector<Dt, C>::MVType, MVSt>::type;

    constexpr DT_S32 ELEM_COUNTS = static_cast<DT_S32>(sizeof(MVDt) / C / sizeof(Dt));
    constexpr DT_S32 VOFFSET     = ELEM_COUNTS * C;
    const DT_S32 width_align     = (width & -ELEM_COUNTS) * C;

    MVSt mv_src_p1[3], mv_src_p0[3], mv_src_c0[3], mv_src_c1[3], mv_src_n0[3], mv_src_n1[3];
    MVDt mv_result0, mv_result1;

    // left
    {
        neon::vload(src_p1,           mv_src_p1[1]);
        neon::vload(src_p1 + VOFFSET, mv_src_p1[2]);
        neon::vload(src_p0,           mv_src_p0[1]);
        neon::vload(src_p0 + VOFFSET, mv_src_p0[2]);
        neon::vload(src_c0,           mv_src_c0[1]);
        neon::vload(src_c0 + VOFFSET, mv_src_c0[2]);
        neon::vload(src_c1,           mv_src_c1[1]);
        neon::vload(src_c1 + VOFFSET, mv_src_c1[2]);
        neon::vload(src_n0,           mv_src_n0[1]);
        neon::vload(src_n0 + VOFFSET, mv_src_n0[2]);
        neon::vload(src_n1,           mv_src_n1[1]);
        neon::vload(src_n1 + VOFFSET, mv_src_n1[2]);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            mv_src_p1[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mv_src_p1[1].val[ch], src_p1[ch], border_value[ch]);
            mv_src_p0[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mv_src_p0[1].val[ch], src_p0[ch], border_value[ch]);
            mv_src_c0[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mv_src_c0[1].val[ch], src_c0[ch], border_value[ch]);
            mv_src_c1[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mv_src_c1[1].val[ch], src_c1[ch], border_value[ch]);
            mv_src_n0[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mv_src_n0[1].val[ch], src_n0[ch], border_value[ch]);
            mv_src_n1[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mv_src_n1[1].val[ch], src_n1[ch], border_value[ch]);
            mv_result0.val[ch]   = Laplacian5x5Core(mv_src_p1[0].val[ch], mv_src_p1[1].val[ch], mv_src_p1[2].val[ch],
                                                    mv_src_p0[0].val[ch], mv_src_p0[1].val[ch], mv_src_p0[2].val[ch],
                                                    mv_src_c0[0].val[ch], mv_src_c0[1].val[ch], mv_src_c0[2].val[ch],
                                                    mv_src_c1[0].val[ch], mv_src_c1[1].val[ch], mv_src_c1[2].val[ch],
                                                    mv_src_n0[0].val[ch], mv_src_n0[1].val[ch], mv_src_n0[2].val[ch]);
            mv_result1.val[ch]   = Laplacian5x5Core(mv_src_p0[0].val[ch], mv_src_p0[1].val[ch], mv_src_p0[2].val[ch],
                                                    mv_src_c0[0].val[ch], mv_src_c0[1].val[ch], mv_src_c0[2].val[ch],
                                                    mv_src_c1[0].val[ch], mv_src_c1[1].val[ch], mv_src_c1[2].val[ch],
                                                    mv_src_n0[0].val[ch], mv_src_n0[1].val[ch], mv_src_n0[2].val[ch],
                                                    mv_src_n1[0].val[ch], mv_src_n1[1].val[ch], mv_src_n1[2].val[ch]);
        }
        neon::vstore(dst_c0, mv_result0);
        neon::vstore(dst_c1, mv_result1);

        Laplacian5x5Prepare(mv_src_p1, mv_src_p0, mv_src_c0, mv_src_c1, mv_src_n0, mv_src_n1);
    }

    // middle
    {
        for (DT_S32 x = VOFFSET; x < width_align - VOFFSET; x += VOFFSET)
        {
            neon::vload(src_p1 + x + VOFFSET, mv_src_p1[2]);
            neon::vload(src_p0 + x + VOFFSET, mv_src_p0[2]);
            neon::vload(src_c0 + x + VOFFSET, mv_src_c0[2]);
            neon::vload(src_c1 + x + VOFFSET, mv_src_c1[2]);
            neon::vload(src_n0 + x + VOFFSET, mv_src_n0[2]);
            neon::vload(src_n1 + x + VOFFSET, mv_src_n1[2]);

            for (DT_S32 ch = 0; ch < C; ch++)
            {
                mv_result0.val[ch] = Laplacian5x5Core(mv_src_p1[0].val[ch], mv_src_p1[1].val[ch], mv_src_p1[2].val[ch],
                                                      mv_src_p0[0].val[ch], mv_src_p0[1].val[ch], mv_src_p0[2].val[ch],
                                                      mv_src_c0[0].val[ch], mv_src_c0[1].val[ch], mv_src_c0[2].val[ch],
                                                      mv_src_c1[0].val[ch], mv_src_c1[1].val[ch], mv_src_c1[2].val[ch],
                                                      mv_src_n0[0].val[ch], mv_src_n0[1].val[ch], mv_src_n0[2].val[ch]);
                mv_result1.val[ch] = Laplacian5x5Core(mv_src_p0[0].val[ch], mv_src_p0[1].val[ch], mv_src_p0[2].val[ch],
                                                      mv_src_c0[0].val[ch], mv_src_c0[1].val[ch], mv_src_c0[2].val[ch],
                                                      mv_src_c1[0].val[ch], mv_src_c1[1].val[ch], mv_src_c1[2].val[ch],
                                                      mv_src_n0[0].val[ch], mv_src_n0[1].val[ch], mv_src_n0[2].val[ch],
                                                      mv_src_n1[0].val[ch], mv_src_n1[1].val[ch], mv_src_n1[2].val[ch]);
            }
            neon::vstore(dst_c0 + x, mv_result0);
            neon::vstore(dst_c1 + x, mv_result1);

            Laplacian5x5Prepare(mv_src_p1, mv_src_p0, mv_src_c0, mv_src_c1, mv_src_n0, mv_src_n1);
        }
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
            neon::vload(src_c0 + x - VOFFSET, mv_src_c0[0]);
            neon::vload(src_c0 + x,           mv_src_c0[1]);
            neon::vload(src_c0 + x + VOFFSET, mv_src_c0[2]);
            neon::vload(src_c1 + x - VOFFSET, mv_src_c1[0]);
            neon::vload(src_c1 + x,           mv_src_c1[1]);
            neon::vload(src_c1 + x + VOFFSET, mv_src_c1[2]);
            neon::vload(src_n0 + x - VOFFSET, mv_src_n0[0]);
            neon::vload(src_n0 + x,           mv_src_n0[1]);
            neon::vload(src_n0 + x + VOFFSET, mv_src_n0[2]);
            neon::vload(src_n1 + x - VOFFSET, mv_src_n1[0]);
            neon::vload(src_n1 + x,           mv_src_n1[1]);
            neon::vload(src_n1 + x + VOFFSET, mv_src_n1[2]);

            for (DT_S32 ch = 0; ch < C; ch++)
            {
                mv_result0.val[ch] = Laplacian5x5Core(mv_src_p1[0].val[ch], mv_src_p1[1].val[ch], mv_src_p1[2].val[ch],
                                                      mv_src_p0[0].val[ch], mv_src_p0[1].val[ch], mv_src_p0[2].val[ch],
                                                      mv_src_c0[0].val[ch], mv_src_c0[1].val[ch], mv_src_c0[2].val[ch],
                                                      mv_src_c1[0].val[ch], mv_src_c1[1].val[ch], mv_src_c1[2].val[ch],
                                                      mv_src_n0[0].val[ch], mv_src_n0[1].val[ch], mv_src_n0[2].val[ch]);
                mv_result1.val[ch] = Laplacian5x5Core(mv_src_p0[0].val[ch], mv_src_p0[1].val[ch], mv_src_p0[2].val[ch],
                                                      mv_src_c0[0].val[ch], mv_src_c0[1].val[ch], mv_src_c0[2].val[ch],
                                                      mv_src_c1[0].val[ch], mv_src_c1[1].val[ch], mv_src_c1[2].val[ch],
                                                      mv_src_n0[0].val[ch], mv_src_n0[1].val[ch], mv_src_n0[2].val[ch],
                                                      mv_src_n1[0].val[ch], mv_src_n1[1].val[ch], mv_src_n1[2].val[ch]);
            }
            neon::vstore(dst_c0 + x, mv_result0);
            neon::vstore(dst_c1 + x, mv_result1);

            Laplacian5x5Prepare(mv_src_p1, mv_src_p0, mv_src_c0, mv_src_c1, mv_src_n0, mv_src_n1);
        }
    }

    // right
    {
        DT_S32 x    = (width - ELEM_COUNTS) * C;
        DT_S32 last = (width - 1) * C;

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            mv_src_p1[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mv_src_p1[1].val[ch], src_p1[last], border_value[ch]);
            mv_src_p0[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mv_src_p0[1].val[ch], src_p0[last], border_value[ch]);
            mv_src_c0[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mv_src_c0[1].val[ch], src_c0[last], border_value[ch]);
            mv_src_c1[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mv_src_c1[1].val[ch], src_c1[last], border_value[ch]);
            mv_src_n0[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mv_src_n0[1].val[ch], src_n0[last], border_value[ch]);
            mv_src_n1[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mv_src_n1[1].val[ch], src_n1[last], border_value[ch]);
            mv_result0.val[ch]   = Laplacian5x5Core(mv_src_p1[0].val[ch], mv_src_p1[1].val[ch], mv_src_p1[2].val[ch],
                                                    mv_src_p0[0].val[ch], mv_src_p0[1].val[ch], mv_src_p0[2].val[ch],
                                                    mv_src_c0[0].val[ch], mv_src_c0[1].val[ch], mv_src_c0[2].val[ch],
                                                    mv_src_c1[0].val[ch], mv_src_c1[1].val[ch], mv_src_c1[2].val[ch],
                                                    mv_src_n0[0].val[ch], mv_src_n0[1].val[ch], mv_src_n0[2].val[ch]);
            mv_result1.val[ch]   = Laplacian5x5Core(mv_src_p0[0].val[ch], mv_src_p0[1].val[ch], mv_src_p0[2].val[ch],
                                                    mv_src_c0[0].val[ch], mv_src_c0[1].val[ch], mv_src_c0[2].val[ch],
                                                    mv_src_c1[0].val[ch], mv_src_c1[1].val[ch], mv_src_c1[2].val[ch],
                                                    mv_src_n0[0].val[ch], mv_src_n0[1].val[ch], mv_src_n0[2].val[ch],
                                                    mv_src_n1[0].val[ch], mv_src_n1[1].val[ch], mv_src_n1[2].val[ch]);
            last++;
        }
        neon::vstore(dst_c0 + x, mv_result0);
        neon::vstore(dst_c1 + x, mv_result1);
    }
}

template <typename St, typename Dt, BorderType BORDER_TYPE, DT_S32 C>
static Status Laplacian5x5NeonImpl(const Mat &src, Mat &dst, const std::vector<St> &border_value,
                                   const St *border_buffer, DT_S32 start_row, DT_S32 end_row)
{
    const St *src_p1 = DT_NULL, *src_p0 = DT_NULL, *src_n0 = DT_NULL, *src_n1 = DT_NULL;
    const St *src_c0 = DT_NULL, *src_c1 = DT_NULL;
    Dt *dst_c0 = DT_NULL, *dst_c1 = DT_NULL;

    DT_S32 width = dst.GetSizes().m_width;
    DT_S32 y     = start_row;

    src_p1 = src.Ptr<St, BORDER_TYPE>(y - 2, border_buffer);
    src_p0 = src.Ptr<St, BORDER_TYPE>(y - 1, border_buffer);
    src_c0 = src.Ptr<St>(y);
    src_c1 = src.Ptr<St>(y + 1);
    src_n0 = src.Ptr<St, BORDER_TYPE>(y + 2, border_buffer);
    src_n1 = src.Ptr<St, BORDER_TYPE>(y + 3, border_buffer);

    DT_S32 h_align2 = (end_row - start_row) & (-2);
    for (; y < start_row + h_align2; y += 2)
    {
        dst_c0 = dst.Ptr<Dt>(y);
        dst_c1 = dst.Ptr<Dt>(y + 1);
        Laplacian5x5TwoRow<St, Dt, BORDER_TYPE, C>(src_p1, src_p0, src_c0, src_c1, src_n0, src_n1, dst_c0, dst_c1, border_value, width);

        src_p1 = src_c0;
        src_p0 = src_c1;
        src_c0 = src_n0;
        src_c1 = src_n1;
        src_n0 = src.Ptr<St, BORDER_TYPE>(y + 4, border_buffer);
        src_n1 = src.Ptr<St, BORDER_TYPE>(y + 5, border_buffer);
    }

    src_n0 = src.Ptr<St, BORDER_TYPE>(y + 1, border_buffer);
    src_n1 = src.Ptr<St, BORDER_TYPE>(y + 2, border_buffer);

    if (y < end_row)
    {
        dst_c0 = dst.Ptr<Dt>(y);
        Laplacian5x5OneRow<St, Dt, BORDER_TYPE, C>(src_p1, src_p0, src_c0, src_n0, src_n1, dst_c0, border_value, width);
    }

    return Status::OK;
}

template <typename St, typename Dt, BorderType BORDER_TYPE>
static Status Laplacian5x5NeonHelper(Context *ctx, const Mat &src, Mat &dst, const std::vector<St> &border_value,
                                     const St *border_buffer, const OpTarget &target)
{
    AURA_UNUSED(target);

    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    DT_S32 height  = dst.GetSizes().m_height;
    DT_S32 channel = dst.GetSizes().m_channel;

    switch (channel)
    {
        case 1:
        {
            ret = wp->ParallelFor(0, height, Laplacian5x5NeonImpl<St, Dt, BORDER_TYPE, 1>, std::cref(src),
                                  std::ref(dst), std::cref(border_value), border_buffer);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Laplacian5x5NeonImpl<St, BORDER_TYPE, 1> failed");
            }
            break;
        }

        case 2:
        {
            ret = wp->ParallelFor(0, height, Laplacian5x5NeonImpl<St, Dt, BORDER_TYPE, 2>, std::cref(src),
                                  std::ref(dst), std::cref(border_value), border_buffer);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Laplacian5x5NeonImpl<St, BORDER_TYPE, 2> failed");
            }
            break;
        }

        case 3:
        {
            ret = wp->ParallelFor(0, height, Laplacian5x5NeonImpl<St, Dt, BORDER_TYPE, 3>, std::cref(src),
                                  std::ref(dst), std::cref(border_value), border_buffer);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Laplacian5x5NeonImpl<St, BORDER_TYPE, 3> failed");
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "unsupported channel");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

template <typename St, typename Dt>
static Status Laplacian5x5NeonHelper(Context *ctx, const Mat &src, Mat &dst, BorderType border_type,
                                     const Scalar &border_value, const OpTarget &target)
{
    Status ret = Status::ERROR;

    St *border_buffer = DT_NULL;
    std::vector<St> vec_border_value = border_value.ToVector<St>();

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

            ret = Laplacian5x5NeonHelper<St, Dt, BorderType::CONSTANT>(ctx, src, dst, vec_border_value, border_buffer, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Laplacian5x5NeonHelper<St, BorderType::CONSTANT> failed");
            }

            AURA_FREE(ctx, border_buffer);
            break;
        }

        case BorderType::REPLICATE:
        {
            ret = Laplacian5x5NeonHelper<St, Dt, BorderType::REPLICATE>(ctx, src, dst, vec_border_value, border_buffer, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Laplacian5x5NeonHelper<St, BorderType::REPLICATE> failed");
            }
            break;
        }

        case BorderType::REFLECT_101:
        {
            ret = Laplacian5x5NeonHelper<St, Dt, BorderType::REFLECT_101>(ctx, src, dst, vec_border_value, border_buffer, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Laplacian5x5NeonHelper<St, BorderType::REFLECT_101> failed");
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "unsupported border type");
            return ret;
        }
    }

    AURA_RETURN(ctx, ret);
}

Status Laplacian5x5Neon(Context *ctx, const Mat &src, Mat &dst, BorderType border_type,
                        const Scalar &border_value, const OpTarget &target)
{
    Status ret = Status::ERROR;
    DT_S32 pattern = AURA_MAKE_PATTERN(src.GetElemType(), dst.GetElemType());

    switch (pattern)
    {
        case AURA_MAKE_PATTERN(ElemType::U8, ElemType::S16):
        {
            ret = Laplacian5x5NeonHelper<DT_U8, DT_S16>(ctx, src, dst, border_type, border_value, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Laplacian5x5NeonHelper<DT_U8> failed");
            }
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::U16, ElemType::U16):
        {
            ret = Laplacian5x5NeonHelper<DT_U16, DT_U16>(ctx, src, dst, border_type, border_value, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Laplacian5x5NeonHelper<DT_U16> failed");
            }
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::S16, ElemType::S16):
        {
            ret = Laplacian5x5NeonHelper<DT_S16, DT_S16>(ctx, src, dst, border_type, border_value, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Laplacian5x5NeonHelper<DT_S16> failed");
            }
            break;
        }

#if defined(AURA_ENABLE_NEON_FP16)
        case AURA_MAKE_PATTERN(ElemType::F16, ElemType::F16):
        {
            ret = Laplacian5x5NeonHelper<MI_F16, MI_F16>(ctx, src, dst, border_type, border_value, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Laplacian5x5NeonHelper<MI_F16> failed");
            }
            break;
        }
#endif // AURA_ENABLE_NEON_FP16

        case AURA_MAKE_PATTERN(ElemType::F32, ElemType::F32):
        {
            ret = Laplacian5x5NeonHelper<DT_F32, DT_F32>(ctx, src, dst, border_type, border_value, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Laplacian5x5NeonHelper<DT_F32> failed");
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "unsupported combination of source format and destination format");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura
