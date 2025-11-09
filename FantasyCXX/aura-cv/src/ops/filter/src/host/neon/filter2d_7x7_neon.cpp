#include "aura/ops/core.h"
#include "aura/runtime/mat.h"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

template <typename MVqType>
AURA_ALWAYS_INLINE DT_VOID Filter2d7x7Prepare(MVqType *mvq_src_p2, MVqType *mvq_src_p1, MVqType *mvq_src_p0, MVqType *mvq_c_src,
                                              MVqType *mvq_src_n0, MVqType *mvq_src_n1, MVqType *mvq_src_n2)
{
    mvq_src_p2[0] = mvq_src_p2[1];
    mvq_src_p1[0] = mvq_src_p1[1];
    mvq_src_p0[0] = mvq_src_p0[1];
    mvq_c_src[0]  = mvq_c_src[1];
    mvq_src_n0[0] = mvq_src_n0[1];
    mvq_src_n1[0] = mvq_src_n1[1];
    mvq_src_n2[0] = mvq_src_n2[1];

    mvq_src_p2[1] = mvq_src_p2[2];
    mvq_src_p1[1] = mvq_src_p1[2];
    mvq_src_p0[1] = mvq_src_p0[2];
    mvq_c_src[1]  = mvq_c_src[2];
    mvq_src_n0[1] = mvq_src_n0[2];
    mvq_src_n1[1] = mvq_src_n1[2];
    mvq_src_n2[1] = mvq_src_n2[2];
}

template <typename MVqType>
AURA_ALWAYS_INLINE DT_VOID Filter2d7x7Prepare(MVqType *mvq_src_p2, MVqType *mvq_src_p1, MVqType *mvq_src_p0, MVqType *mvq_c0_src,
                                              MVqType *mvq_src_c1, MVqType *mvq_src_n0, MVqType *mvq_src_n1, MVqType *mvq_src_n2)
{
    mvq_src_p2[0] = mvq_src_p2[1];
    mvq_src_p1[0] = mvq_src_p1[1];
    mvq_src_p0[0] = mvq_src_p0[1];
    mvq_c0_src[0] = mvq_c0_src[1];
    mvq_src_c1[0] = mvq_src_c1[1];
    mvq_src_n0[0] = mvq_src_n0[1];
    mvq_src_n1[0] = mvq_src_n1[1];
    mvq_src_n2[0] = mvq_src_n2[1];

    mvq_src_p2[1] = mvq_src_p2[2];
    mvq_src_p1[1] = mvq_src_p1[2];
    mvq_src_p0[1] = mvq_src_p0[2];
    mvq_c0_src[1] = mvq_c0_src[2];
    mvq_src_c1[1] = mvq_src_c1[2];
    mvq_src_n0[1] = mvq_src_n0[2];
    mvq_src_n1[1] = mvq_src_n1[2];
    mvq_src_n2[1] = mvq_src_n2[2];
}

AURA_ALWAYS_INLINE DT_VOID Filter2d7x7Core(uint8x16_t &vqu8_src_x0, uint8x16_t &vqu8_src_x1, uint8x16_t &vqu8_src_x2,
                                           float32x4_t &vqf32_result_lo_lo, float32x4_t &vqf32_result_lo_hi,
                                           float32x4_t &vqf32_result_hi_lo, float32x4_t &vqf32_result_hi_hi,
                                           const std::vector<DT_F32> &kernel, DT_S32 line)
{
    const DT_S32 idx = 7 * line;

    uint8x16_t vqu8_src_l2      = neon::vext<13>(vqu8_src_x0, vqu8_src_x1);
    uint8x16_t vqu8_src_l1      = neon::vext<14>(vqu8_src_x0, vqu8_src_x1);
    uint8x16_t vqu8_src_l0      = neon::vext<15>(vqu8_src_x0, vqu8_src_x1);
    uint8x16_t vqu8_src_r0      = neon::vext<1>(vqu8_src_x1, vqu8_src_x2);
    uint8x16_t vqu8_src_r1      = neon::vext<2>(vqu8_src_x1, vqu8_src_x2);
    uint8x16_t vqu8_src_r2      = neon::vext<3>(vqu8_src_x1, vqu8_src_x2);

    uint16x8_t vqu16_src_l2_lo  = neon::vmovl(neon::vgetlow(vqu8_src_l2));
    uint16x8_t vqu16_src_l1_lo  = neon::vmovl(neon::vgetlow(vqu8_src_l1));
    uint16x8_t vqu16_src_l0_lo  = neon::vmovl(neon::vgetlow(vqu8_src_l0));
    uint16x8_t vqu16_src_c_lo   = neon::vmovl(neon::vgetlow(vqu8_src_x1));
    uint16x8_t vqu16_src_r0_lo  = neon::vmovl(neon::vgetlow(vqu8_src_r0));
    uint16x8_t vqu16_src_r1_lo  = neon::vmovl(neon::vgetlow(vqu8_src_r1));
    uint16x8_t vqu16_src_r2_lo  = neon::vmovl(neon::vgetlow(vqu8_src_r2));

    float32x4_t vqf32_src_l2_lo = neon::vcvt<DT_F32>(neon::vmovl(neon::vgetlow(vqu16_src_l2_lo)));
    float32x4_t vqf32_src_l2_hi = neon::vcvt<DT_F32>(neon::vmovl(neon::vgethigh(vqu16_src_l2_lo)));
    float32x4_t vqf32_src_l1_lo = neon::vcvt<DT_F32>(neon::vmovl(neon::vgetlow(vqu16_src_l1_lo)));
    float32x4_t vqf32_src_l1_hi = neon::vcvt<DT_F32>(neon::vmovl(neon::vgethigh(vqu16_src_l1_lo)));
    float32x4_t vqf32_src_l0_lo = neon::vcvt<DT_F32>(neon::vmovl(neon::vgetlow(vqu16_src_l0_lo)));
    float32x4_t vqf32_src_l0_hi = neon::vcvt<DT_F32>(neon::vmovl(neon::vgethigh(vqu16_src_l0_lo)));
    float32x4_t vqf32_src_c_lo  = neon::vcvt<DT_F32>(neon::vmovl(neon::vgetlow(vqu16_src_c_lo)));
    float32x4_t vqf32_src_c_hi  = neon::vcvt<DT_F32>(neon::vmovl(neon::vgethigh(vqu16_src_c_lo)));
    float32x4_t vqf32_src_r0_lo = neon::vcvt<DT_F32>(neon::vmovl(neon::vgetlow(vqu16_src_r0_lo)));
    float32x4_t vqf32_src_r0_hi = neon::vcvt<DT_F32>(neon::vmovl(neon::vgethigh(vqu16_src_r0_lo)));
    float32x4_t vqf32_src_r1_lo = neon::vcvt<DT_F32>(neon::vmovl(neon::vgetlow(vqu16_src_r1_lo)));
    float32x4_t vqf32_src_r1_hi = neon::vcvt<DT_F32>(neon::vmovl(neon::vgethigh(vqu16_src_r1_lo)));
    float32x4_t vqf32_src_r2_lo = neon::vcvt<DT_F32>(neon::vmovl(neon::vgetlow(vqu16_src_r2_lo)));
    float32x4_t vqf32_src_r2_hi = neon::vcvt<DT_F32>(neon::vmovl(neon::vgethigh(vqu16_src_r2_lo)));

    vqf32_result_lo_lo          = neon::vmla(vqf32_result_lo_lo, vqf32_src_l2_lo, kernel[idx + 0]);
    vqf32_result_lo_hi          = neon::vmla(vqf32_result_lo_hi, vqf32_src_l2_hi, kernel[idx + 0]);
    vqf32_result_lo_lo          = neon::vmla(vqf32_result_lo_lo, vqf32_src_l1_lo, kernel[idx + 1]);
    vqf32_result_lo_hi          = neon::vmla(vqf32_result_lo_hi, vqf32_src_l1_hi, kernel[idx + 1]);
    vqf32_result_lo_lo          = neon::vmla(vqf32_result_lo_lo, vqf32_src_l0_lo, kernel[idx + 2]);
    vqf32_result_lo_hi          = neon::vmla(vqf32_result_lo_hi, vqf32_src_l0_hi, kernel[idx + 2]);
    vqf32_result_lo_lo          = neon::vmla(vqf32_result_lo_lo, vqf32_src_c_lo,  kernel[idx + 3]);
    vqf32_result_lo_hi          = neon::vmla(vqf32_result_lo_hi, vqf32_src_c_hi,  kernel[idx + 3]);
    vqf32_result_lo_lo          = neon::vmla(vqf32_result_lo_lo, vqf32_src_r0_lo, kernel[idx + 4]);
    vqf32_result_lo_hi          = neon::vmla(vqf32_result_lo_hi, vqf32_src_r0_hi, kernel[idx + 4]);
    vqf32_result_lo_lo          = neon::vmla(vqf32_result_lo_lo, vqf32_src_r1_lo, kernel[idx + 5]);
    vqf32_result_lo_hi          = neon::vmla(vqf32_result_lo_hi, vqf32_src_r1_hi, kernel[idx + 5]);
    vqf32_result_lo_lo          = neon::vmla(vqf32_result_lo_lo, vqf32_src_r2_lo, kernel[idx + 6]);
    vqf32_result_lo_hi          = neon::vmla(vqf32_result_lo_hi, vqf32_src_r2_hi, kernel[idx + 6]);

    uint16x8_t vqu16_src_l2_hi  = neon::vmovl(neon::vgethigh(vqu8_src_l2));
    uint16x8_t vqu16_src_l1_hi  = neon::vmovl(neon::vgethigh(vqu8_src_l1));
    uint16x8_t vqu16_src_l0_hi  = neon::vmovl(neon::vgethigh(vqu8_src_l0));
    uint16x8_t vqu16_src_c_hi   = neon::vmovl(neon::vgethigh(vqu8_src_x1));
    uint16x8_t vqu16_src_r0_hi  = neon::vmovl(neon::vgethigh(vqu8_src_r0));
    uint16x8_t vqu16_src_r1_hi  = neon::vmovl(neon::vgethigh(vqu8_src_r1));
    uint16x8_t vqu16_src_r2_hi  = neon::vmovl(neon::vgethigh(vqu8_src_r2));

    vqf32_src_l2_lo             = neon::vcvt<DT_F32>(neon::vmovl(neon::vgetlow(vqu16_src_l2_hi)));
    vqf32_src_l2_hi             = neon::vcvt<DT_F32>(neon::vmovl(neon::vgethigh(vqu16_src_l2_hi)));
    vqf32_src_l1_lo             = neon::vcvt<DT_F32>(neon::vmovl(neon::vgetlow(vqu16_src_l1_hi)));
    vqf32_src_l1_hi             = neon::vcvt<DT_F32>(neon::vmovl(neon::vgethigh(vqu16_src_l1_hi)));
    vqf32_src_l0_lo             = neon::vcvt<DT_F32>(neon::vmovl(neon::vgetlow(vqu16_src_l0_hi)));
    vqf32_src_l0_hi             = neon::vcvt<DT_F32>(neon::vmovl(neon::vgethigh(vqu16_src_l0_hi)));
    vqf32_src_c_lo              = neon::vcvt<DT_F32>(neon::vmovl(neon::vgetlow(vqu16_src_c_hi)));
    vqf32_src_c_hi              = neon::vcvt<DT_F32>(neon::vmovl(neon::vgethigh(vqu16_src_c_hi)));
    vqf32_src_r0_lo             = neon::vcvt<DT_F32>(neon::vmovl(neon::vgetlow(vqu16_src_r0_hi)));
    vqf32_src_r0_hi             = neon::vcvt<DT_F32>(neon::vmovl(neon::vgethigh(vqu16_src_r0_hi)));
    vqf32_src_r1_lo             = neon::vcvt<DT_F32>(neon::vmovl(neon::vgetlow(vqu16_src_r1_hi)));
    vqf32_src_r1_hi             = neon::vcvt<DT_F32>(neon::vmovl(neon::vgethigh(vqu16_src_r1_hi)));
    vqf32_src_r2_lo             = neon::vcvt<DT_F32>(neon::vmovl(neon::vgetlow(vqu16_src_r2_hi)));
    vqf32_src_r2_hi             = neon::vcvt<DT_F32>(neon::vmovl(neon::vgethigh(vqu16_src_r2_hi)));

    vqf32_result_hi_lo          = neon::vmla(vqf32_result_hi_lo, vqf32_src_l2_lo, kernel[idx + 0]);
    vqf32_result_hi_hi          = neon::vmla(vqf32_result_hi_hi, vqf32_src_l2_hi, kernel[idx + 0]);
    vqf32_result_hi_lo          = neon::vmla(vqf32_result_hi_lo, vqf32_src_l1_lo, kernel[idx + 1]);
    vqf32_result_hi_hi          = neon::vmla(vqf32_result_hi_hi, vqf32_src_l1_hi, kernel[idx + 1]);
    vqf32_result_hi_lo          = neon::vmla(vqf32_result_hi_lo, vqf32_src_l0_lo, kernel[idx + 2]);
    vqf32_result_hi_hi          = neon::vmla(vqf32_result_hi_hi, vqf32_src_l0_hi, kernel[idx + 2]);
    vqf32_result_hi_lo          = neon::vmla(vqf32_result_hi_lo, vqf32_src_c_lo,  kernel[idx + 3]);
    vqf32_result_hi_hi          = neon::vmla(vqf32_result_hi_hi, vqf32_src_c_hi,  kernel[idx + 3]);
    vqf32_result_hi_lo          = neon::vmla(vqf32_result_hi_lo, vqf32_src_r0_lo, kernel[idx + 4]);
    vqf32_result_hi_hi          = neon::vmla(vqf32_result_hi_hi, vqf32_src_r0_hi, kernel[idx + 4]);
    vqf32_result_hi_lo          = neon::vmla(vqf32_result_hi_lo, vqf32_src_r1_lo, kernel[idx + 5]);
    vqf32_result_hi_hi          = neon::vmla(vqf32_result_hi_hi, vqf32_src_r1_hi, kernel[idx + 5]);
    vqf32_result_hi_lo          = neon::vmla(vqf32_result_hi_lo, vqf32_src_r2_lo, kernel[idx + 6]);
    vqf32_result_hi_hi          = neon::vmla(vqf32_result_hi_hi, vqf32_src_r2_hi, kernel[idx + 6]);
}

AURA_ALWAYS_INLINE uint8x16_t Filter2d7x7Vector(uint8x16_t &vqu8_src_p2x0, uint8x16_t &vqu8_src_p2x1, uint8x16_t &vqu8_src_p2x2,
                                                uint8x16_t &vqu8_src_p1x0, uint8x16_t &vqu8_src_p1x1, uint8x16_t &vqu8_src_p1x2,
                                                uint8x16_t &vqu8_src_p0x0, uint8x16_t &vqu8_src_p0x1, uint8x16_t &vqu8_src_p0x2,
                                                uint8x16_t &vqu8_src_cx0,  uint8x16_t &vqu8_src_cx1,  uint8x16_t &vqu8_src_cx2,
                                                uint8x16_t &vqu8_src_n0x0, uint8x16_t &vqu8_src_n0x1, uint8x16_t &vqu8_src_n0x2,
                                                uint8x16_t &vqu8_src_n1x0, uint8x16_t &vqu8_src_n1x1, uint8x16_t &vqu8_src_n1x2,
                                                uint8x16_t &vqu8_src_n2x0, uint8x16_t &vqu8_src_n2x1, uint8x16_t &vqu8_src_n2x2,
                                                const std::vector<DT_F32> &kernel)
{
    float32x4_t vqf32_result_lo_lo, vqf32_result_lo_hi,  vqf32_result_hi_lo, vqf32_result_hi_hi;
    neon::vdup(vqf32_result_lo_lo, 0.f);
    neon::vdup(vqf32_result_lo_hi, 0.f);
    neon::vdup(vqf32_result_hi_lo, 0.f);
    neon::vdup(vqf32_result_hi_hi, 0.f);

    Filter2d7x7Core(vqu8_src_p2x0, vqu8_src_p2x1, vqu8_src_p2x2, vqf32_result_lo_lo, vqf32_result_lo_hi, vqf32_result_hi_lo, vqf32_result_hi_hi, kernel, 0);
    Filter2d7x7Core(vqu8_src_p1x0, vqu8_src_p1x1, vqu8_src_p1x2, vqf32_result_lo_lo, vqf32_result_lo_hi, vqf32_result_hi_lo, vqf32_result_hi_hi, kernel, 1);
    Filter2d7x7Core(vqu8_src_p0x0, vqu8_src_p0x1, vqu8_src_p0x2, vqf32_result_lo_lo, vqf32_result_lo_hi, vqf32_result_hi_lo, vqf32_result_hi_hi, kernel, 2);
    Filter2d7x7Core(vqu8_src_cx0,  vqu8_src_cx1,  vqu8_src_cx2,  vqf32_result_lo_lo, vqf32_result_lo_hi, vqf32_result_hi_lo, vqf32_result_hi_hi, kernel, 3);
    Filter2d7x7Core(vqu8_src_n0x0, vqu8_src_n0x1, vqu8_src_n0x2, vqf32_result_lo_lo, vqf32_result_lo_hi, vqf32_result_hi_lo, vqf32_result_hi_hi, kernel, 4);
    Filter2d7x7Core(vqu8_src_n1x0, vqu8_src_n1x1, vqu8_src_n1x2, vqf32_result_lo_lo, vqf32_result_lo_hi, vqf32_result_hi_lo, vqf32_result_hi_hi, kernel, 5);
    Filter2d7x7Core(vqu8_src_n2x0, vqu8_src_n2x1, vqu8_src_n2x2, vqf32_result_lo_lo, vqf32_result_lo_hi, vqf32_result_hi_lo, vqf32_result_hi_hi, kernel, 6);

    uint32x4_t vqu32_result_lo_lo = neon::vcvt<DT_U32>(neon::vrndn(vqf32_result_lo_lo));
    uint32x4_t vqu32_result_lo_hi = neon::vcvt<DT_U32>(neon::vrndn(vqf32_result_lo_hi));
    uint32x4_t vqu32_result_hi_lo = neon::vcvt<DT_U32>(neon::vrndn(vqf32_result_hi_lo));
    uint32x4_t vqu32_result_hi_hi = neon::vcvt<DT_U32>(neon::vrndn(vqf32_result_hi_hi));
    uint8x8_t  vdu8_result_lo     = neon::vqmovn(neon::vcombine(neon::vqmovn(vqu32_result_lo_lo), neon::vqmovn(vqu32_result_lo_hi)));
    uint8x8_t  vdu8_result_hi     = neon::vqmovn(neon::vcombine(neon::vqmovn(vqu32_result_hi_lo), neon::vqmovn(vqu32_result_hi_hi)));

    return neon::vcombine(vdu8_result_lo, vdu8_result_hi);
}

template <typename d16x8_t, typename std::enable_if<(std::is_same<d16x8_t, uint16x8_t>::value ||
                                                     std::is_same<d16x8_t, int16x8_t>::value)>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Filter2d7x7Core(d16x8_t &vq16_src_x0, d16x8_t &vq16_src_x1, d16x8_t &vq16_src_x2,
                                           float32x4_t &vqf32_result_lo, float32x4_t &vqf32_result_hi,
                                           const std::vector<DT_F32> &kernel, DT_S32 line)
{
    const DT_S32 idx = 7 * line;

    d16x8_t vq16_src_l2         = neon::vext<5>(vq16_src_x0, vq16_src_x1);
    d16x8_t vq16_src_l1         = neon::vext<6>(vq16_src_x0, vq16_src_x1);
    d16x8_t vq16_src_l0         = neon::vext<7>(vq16_src_x0, vq16_src_x1);
    d16x8_t vq16_src_r0         = neon::vext<1>(vq16_src_x1, vq16_src_x2);
    d16x8_t vq16_src_r1         = neon::vext<2>(vq16_src_x1, vq16_src_x2);
    d16x8_t vq16_src_r2         = neon::vext<3>(vq16_src_x1, vq16_src_x2);

    float32x4_t vqf32_src_l2_lo = neon::vcvt<DT_F32>(neon::vmovl(neon::vgetlow(vq16_src_l2)));
    float32x4_t vqf32_src_l2_hi = neon::vcvt<DT_F32>(neon::vmovl(neon::vgethigh(vq16_src_l2)));
    float32x4_t vqf32_src_l1_lo = neon::vcvt<DT_F32>(neon::vmovl(neon::vgetlow(vq16_src_l1)));
    float32x4_t vqf32_src_l1_hi = neon::vcvt<DT_F32>(neon::vmovl(neon::vgethigh(vq16_src_l1)));
    float32x4_t vqf32_src_l0_lo = neon::vcvt<DT_F32>(neon::vmovl(neon::vgetlow(vq16_src_l0)));
    float32x4_t vqf32_src_l0_hi = neon::vcvt<DT_F32>(neon::vmovl(neon::vgethigh(vq16_src_l0)));
    float32x4_t vqf32_src_c_lo  = neon::vcvt<DT_F32>(neon::vmovl(neon::vgetlow(vq16_src_x1)));
    float32x4_t vqf32_src_c_hi  = neon::vcvt<DT_F32>(neon::vmovl(neon::vgethigh(vq16_src_x1)));
    float32x4_t vqf32_src_r0_lo = neon::vcvt<DT_F32>(neon::vmovl(neon::vgetlow(vq16_src_r0)));
    float32x4_t vqf32_src_r0_hi = neon::vcvt<DT_F32>(neon::vmovl(neon::vgethigh(vq16_src_r0)));
    float32x4_t vqf32_src_r1_lo = neon::vcvt<DT_F32>(neon::vmovl(neon::vgetlow(vq16_src_r1)));
    float32x4_t vqf32_src_r1_hi = neon::vcvt<DT_F32>(neon::vmovl(neon::vgethigh(vq16_src_r1)));
    float32x4_t vqf32_src_r2_lo = neon::vcvt<DT_F32>(neon::vmovl(neon::vgetlow(vq16_src_r2)));
    float32x4_t vqf32_src_r2_hi = neon::vcvt<DT_F32>(neon::vmovl(neon::vgethigh(vq16_src_r2)));

    vqf32_result_lo             = neon::vmla(vqf32_result_lo, vqf32_src_l2_lo, kernel[idx + 0]);
    vqf32_result_hi             = neon::vmla(vqf32_result_hi, vqf32_src_l2_hi, kernel[idx + 0]);
    vqf32_result_lo             = neon::vmla(vqf32_result_lo, vqf32_src_l1_lo, kernel[idx + 1]);
    vqf32_result_hi             = neon::vmla(vqf32_result_hi, vqf32_src_l1_hi, kernel[idx + 1]);
    vqf32_result_lo             = neon::vmla(vqf32_result_lo, vqf32_src_l0_lo, kernel[idx + 2]);
    vqf32_result_hi             = neon::vmla(vqf32_result_hi, vqf32_src_l0_hi, kernel[idx + 2]);
    vqf32_result_lo             = neon::vmla(vqf32_result_lo, vqf32_src_c_lo,  kernel[idx + 3]);
    vqf32_result_hi             = neon::vmla(vqf32_result_hi, vqf32_src_c_hi,  kernel[idx + 3]);
    vqf32_result_lo             = neon::vmla(vqf32_result_lo, vqf32_src_r0_lo, kernel[idx + 4]);
    vqf32_result_hi             = neon::vmla(vqf32_result_hi, vqf32_src_r0_hi, kernel[idx + 4]);
    vqf32_result_lo             = neon::vmla(vqf32_result_lo, vqf32_src_r1_lo, kernel[idx + 5]);
    vqf32_result_hi             = neon::vmla(vqf32_result_hi, vqf32_src_r1_hi, kernel[idx + 5]);
    vqf32_result_lo             = neon::vmla(vqf32_result_lo, vqf32_src_r2_lo, kernel[idx + 6]);
    vqf32_result_hi             = neon::vmla(vqf32_result_hi, vqf32_src_r2_hi, kernel[idx + 6]);
}

template <typename d16x8_t, typename D16 = typename neon::Scalar<d16x8_t>::SType,
          typename std::enable_if<(std::is_same<d16x8_t, uint16x8_t>::value || std::is_same<d16x8_t, int16x8_t>::value)>::type* = DT_NULL>
AURA_ALWAYS_INLINE d16x8_t Filter2d7x7Vector(d16x8_t &vq16_src_p2x0, d16x8_t &vq16_src_p2x1, d16x8_t &vq16_src_p2x2,
                                             d16x8_t &vq16_src_p1x0, d16x8_t &vq16_src_p1x1, d16x8_t &vq16_src_p1x2,
                                             d16x8_t &vq16_src_p0x0, d16x8_t &vq16_src_p0x1, d16x8_t &vq16_src_p0x2,
                                             d16x8_t &vq16_src_cx0,  d16x8_t &vq16_src_cx1,  d16x8_t &vq16_src_cx2,
                                             d16x8_t &vq16_src_n0x0, d16x8_t &vq16_src_n0x1, d16x8_t &vq16_src_n0x2,
                                             d16x8_t &vq16_src_n1x0, d16x8_t &vq16_src_n1x1, d16x8_t &vq16_src_n1x2,
                                             d16x8_t &vq16_src_n2x0, d16x8_t &vq16_src_n2x1, d16x8_t &vq16_src_n2x2,
                                             const std::vector<DT_F32> &kernel)
{
    using D32     = typename Promote<D16>::Type;
    using d32x4_t = typename neon::QVector<D32>::VType;

    float32x4_t vqf32_result_lo,  vqf32_result_hi;
    neon::vdup(vqf32_result_lo, 0.f);
    neon::vdup(vqf32_result_hi, 0.f);

    Filter2d7x7Core(vq16_src_p2x0, vq16_src_p2x1, vq16_src_p2x2, vqf32_result_lo, vqf32_result_hi, kernel, 0);
    Filter2d7x7Core(vq16_src_p1x0, vq16_src_p1x1, vq16_src_p1x2, vqf32_result_lo, vqf32_result_hi, kernel, 1);
    Filter2d7x7Core(vq16_src_p0x0, vq16_src_p0x1, vq16_src_p0x2, vqf32_result_lo, vqf32_result_hi, kernel, 2);
    Filter2d7x7Core(vq16_src_cx0,  vq16_src_cx1,  vq16_src_cx2,  vqf32_result_lo, vqf32_result_hi, kernel, 3);
    Filter2d7x7Core(vq16_src_n0x0, vq16_src_n0x1, vq16_src_n0x2, vqf32_result_lo, vqf32_result_hi, kernel, 4);
    Filter2d7x7Core(vq16_src_n1x0, vq16_src_n1x1, vq16_src_n1x2, vqf32_result_lo, vqf32_result_hi, kernel, 5);
    Filter2d7x7Core(vq16_src_n2x0, vq16_src_n2x1, vq16_src_n2x2, vqf32_result_lo, vqf32_result_hi, kernel, 6);

    d32x4_t vq32_result_lo = neon::vcvt<D32>(neon::vrndn(vqf32_result_lo));
    d32x4_t vq32_result_hi = neon::vcvt<D32>(neon::vrndn(vqf32_result_hi));

    d16x8_t vq_result = neon::vcombine(neon::vqmovn(vq32_result_lo), neon::vqmovn(vq32_result_hi));

    return vq_result;
}

AURA_ALWAYS_INLINE DT_VOID Filter2d7x7Core(float32x4_t &vqf32_src_x0, float32x4_t &vqf32_src_x1, float32x4_t &vqf32_src_x2,
                                           float32x4_t &vqf32_result, const std::vector<DT_F32> &kernel, DT_S32 line)
{
    const DT_S32 idx = 7 * line;

    float32x4_t vqf32_src_l2 = neon::vext<1>(vqf32_src_x0, vqf32_src_x1);
    float32x4_t vqf32_src_l1 = neon::vext<2>(vqf32_src_x0, vqf32_src_x1);
    float32x4_t vqf32_src_l0 = neon::vext<3>(vqf32_src_x0, vqf32_src_x1);
    float32x4_t vqf32_src_r0 = neon::vext<1>(vqf32_src_x1, vqf32_src_x2);
    float32x4_t vqf32_src_r1 = neon::vext<2>(vqf32_src_x1, vqf32_src_x2);
    float32x4_t vqf32_src_r2 = neon::vext<3>(vqf32_src_x1, vqf32_src_x2);

    vqf32_result = neon::vmla(vqf32_result, vqf32_src_l2, kernel[idx + 0]);
    vqf32_result = neon::vmla(vqf32_result, vqf32_src_l1, kernel[idx + 1]);
    vqf32_result = neon::vmla(vqf32_result, vqf32_src_l0, kernel[idx + 2]);
    vqf32_result = neon::vmla(vqf32_result, vqf32_src_x1, kernel[idx + 3]);
    vqf32_result = neon::vmla(vqf32_result, vqf32_src_r0, kernel[idx + 4]);
    vqf32_result = neon::vmla(vqf32_result, vqf32_src_r1, kernel[idx + 5]);
    vqf32_result = neon::vmla(vqf32_result, vqf32_src_r2, kernel[idx + 6]);
}

AURA_ALWAYS_INLINE float32x4_t Filter2d7x7Vector(float32x4_t &vqf32_src_p2x0, float32x4_t &vqf32_src_p2x1, float32x4_t &vqf32_src_p2x2,
                                                 float32x4_t &vqf32_src_p1x0, float32x4_t &vqf32_src_p1x1, float32x4_t &vqf32_src_p1x2,
                                                 float32x4_t &vqf32_src_p0x0, float32x4_t &vqf32_src_p0x1, float32x4_t &vqf32_src_p0x2,
                                                 float32x4_t &vqf32_src_cx0,  float32x4_t &vqf32_src_cx1,  float32x4_t &vqf32_src_cx2,
                                                 float32x4_t &vqf32_src_n0x0, float32x4_t &vqf32_src_n0x1, float32x4_t &vqf32_src_n0x2,
                                                 float32x4_t &vqf32_src_n1x0, float32x4_t &vqf32_src_n1x1, float32x4_t &vqf32_src_n1x2,
                                                 float32x4_t &vqf32_src_n2x0, float32x4_t &vqf32_src_n2x1, float32x4_t &vqf32_src_n2x2,
                                                 const std::vector<DT_F32> &kernel)
{
    float32x4_t vqf32_result;
    neon::vdup(vqf32_result, 0.f);

    Filter2d7x7Core(vqf32_src_p2x0, vqf32_src_p2x1, vqf32_src_p2x2, vqf32_result, kernel, 0);
    Filter2d7x7Core(vqf32_src_p1x0, vqf32_src_p1x1, vqf32_src_p1x2, vqf32_result, kernel, 1);
    Filter2d7x7Core(vqf32_src_p0x0, vqf32_src_p0x1, vqf32_src_p0x2, vqf32_result, kernel, 2);
    Filter2d7x7Core(vqf32_src_cx0,  vqf32_src_cx1,  vqf32_src_cx2,  vqf32_result, kernel, 3);
    Filter2d7x7Core(vqf32_src_n0x0, vqf32_src_n0x1, vqf32_src_n0x2, vqf32_result, kernel, 4);
    Filter2d7x7Core(vqf32_src_n1x0, vqf32_src_n1x1, vqf32_src_n1x2, vqf32_result, kernel, 5);
    Filter2d7x7Core(vqf32_src_n2x0, vqf32_src_n2x1, vqf32_src_n2x2, vqf32_result, kernel, 6);

    return vqf32_result;
}

#if defined(AURA_ENABLE_NEON_FP16)
AURA_ALWAYS_INLINE DT_VOID Filter2d7x7Core(float16x8_t &vqf16_src_x0, float16x8_t &vqf16_src_x1, float16x8_t &vqf16_src_x2,
                                           float32x4_t &vqf32_result_lo, float32x4_t &vqf32_result_hi,
                                           const std::vector<DT_F32> &kernel, DT_S32 line)
{
    const DT_S32 idx = 7 * line;

    float16x8_t vqf16_src_l2    = neon::vext<5>(vqf16_src_x0, vqf16_src_x1);
    float16x8_t vqf16_src_l1    = neon::vext<6>(vqf16_src_x0, vqf16_src_x1);
    float16x8_t vqf16_src_l0    = neon::vext<7>(vqf16_src_x0, vqf16_src_x1);
    float16x8_t vqf16_src_r0    = neon::vext<1>(vqf16_src_x1, vqf16_src_x2);
    float16x8_t vqf16_src_r1    = neon::vext<2>(vqf16_src_x1, vqf16_src_x2);
    float16x8_t vqf16_src_r2    = neon::vext<3>(vqf16_src_x1, vqf16_src_x2);

    float32x4_t vqf32_src_l2_lo = neon::vcvt<DT_F32>(neon::vgetlow(vqf16_src_l2));
    float32x4_t vqf32_src_l2_hi = neon::vcvt<DT_F32>(neon::vgethigh(vqf16_src_l2));
    float32x4_t vqf32_src_l1_lo = neon::vcvt<DT_F32>(neon::vgetlow(vqf16_src_l1));
    float32x4_t vqf32_src_l1_hi = neon::vcvt<DT_F32>(neon::vgethigh(vqf16_src_l1));
    float32x4_t vqf32_src_l0_lo = neon::vcvt<DT_F32>(neon::vgetlow(vqf16_src_l0));
    float32x4_t vqf32_src_l0_hi = neon::vcvt<DT_F32>(neon::vgethigh(vqf16_src_l0));
    float32x4_t vqf32_src_c_lo  = neon::vcvt<DT_F32>(neon::vgetlow(vqf16_src_x1));
    float32x4_t vqf32_src_c_hi  = neon::vcvt<DT_F32>(neon::vgethigh(vqf16_src_x1));
    float32x4_t vqf32_src_r0_lo = neon::vcvt<DT_F32>(neon::vgetlow(vqf16_src_r0));
    float32x4_t vqf32_src_r0_hi = neon::vcvt<DT_F32>(neon::vgethigh(vqf16_src_r0));
    float32x4_t vqf32_src_r1_lo = neon::vcvt<DT_F32>(neon::vgetlow(vqf16_src_r1));
    float32x4_t vqf32_src_r1_hi = neon::vcvt<DT_F32>(neon::vgethigh(vqf16_src_r1));
    float32x4_t vqf32_src_r2_lo = neon::vcvt<DT_F32>(neon::vgetlow(vqf16_src_r2));
    float32x4_t vqf32_src_r2_hi = neon::vcvt<DT_F32>(neon::vgethigh(vqf16_src_r2));

    vqf32_result_lo             = neon::vmla(vqf32_result_lo, vqf32_src_l2_lo, kernel[idx + 0]);
    vqf32_result_hi             = neon::vmla(vqf32_result_hi, vqf32_src_l2_hi, kernel[idx + 0]);
    vqf32_result_lo             = neon::vmla(vqf32_result_lo, vqf32_src_l1_lo, kernel[idx + 1]);
    vqf32_result_hi             = neon::vmla(vqf32_result_hi, vqf32_src_l1_hi, kernel[idx + 1]);
    vqf32_result_lo             = neon::vmla(vqf32_result_lo, vqf32_src_l0_lo, kernel[idx + 2]);
    vqf32_result_hi             = neon::vmla(vqf32_result_hi, vqf32_src_l0_hi, kernel[idx + 2]);
    vqf32_result_lo             = neon::vmla(vqf32_result_lo, vqf32_src_c_lo,  kernel[idx + 3]);
    vqf32_result_hi             = neon::vmla(vqf32_result_hi, vqf32_src_c_hi,  kernel[idx + 3]);
    vqf32_result_lo             = neon::vmla(vqf32_result_lo, vqf32_src_r0_lo, kernel[idx + 4]);
    vqf32_result_hi             = neon::vmla(vqf32_result_hi, vqf32_src_r0_hi, kernel[idx + 4]);
    vqf32_result_lo             = neon::vmla(vqf32_result_lo, vqf32_src_r1_lo, kernel[idx + 5]);
    vqf32_result_hi             = neon::vmla(vqf32_result_hi, vqf32_src_r1_hi, kernel[idx + 5]);
    vqf32_result_lo             = neon::vmla(vqf32_result_lo, vqf32_src_r2_lo, kernel[idx + 6]);
    vqf32_result_hi             = neon::vmla(vqf32_result_hi, vqf32_src_r2_hi, kernel[idx + 6]);
}

AURA_ALWAYS_INLINE float16x8_t Filter2d7x7Vector(float16x8_t &vqf16_src_p2x0, float16x8_t &vqf16_src_p2x1, float16x8_t &vqf16_src_p2x2,
                                                 float16x8_t &vqf16_src_p1x0, float16x8_t &vqf16_src_p1x1, float16x8_t &vqf16_src_p1x2,
                                                 float16x8_t &vqf16_src_p0x0, float16x8_t &vqf16_src_p0x1, float16x8_t &vqf16_src_p0x2,
                                                 float16x8_t &vqf16_src_cx0,  float16x8_t &vqf16_src_cx1,  float16x8_t &vqf16_src_cx2,
                                                 float16x8_t &vqf16_src_n0x0, float16x8_t &vqf16_src_n0x1, float16x8_t &vqf16_src_n0x2,
                                                 float16x8_t &vqf16_src_n1x0, float16x8_t &vqf16_src_n1x1, float16x8_t &vqf16_src_n1x2,
                                                 float16x8_t &vqf16_src_n2x0, float16x8_t &vqf16_src_n2x1, float16x8_t &vqf16_src_n2x2,
                                                 const std::vector<DT_F32> &kernel)
{
    float32x4_t vqf32_result_lo,  vqf32_result_hi;
    neon::vdup(vqf32_result_lo, 0.f);
    neon::vdup(vqf32_result_hi, 0.f);

    Filter2d7x7Core(vqf16_src_p2x0, vqf16_src_p2x1, vqf16_src_p2x2, vqf32_result_lo, vqf32_result_hi, kernel, 0);
    Filter2d7x7Core(vqf16_src_p1x0, vqf16_src_p1x1, vqf16_src_p1x2, vqf32_result_lo, vqf32_result_hi, kernel, 1);
    Filter2d7x7Core(vqf16_src_p0x0, vqf16_src_p0x1, vqf16_src_p0x2, vqf32_result_lo, vqf32_result_hi, kernel, 2);
    Filter2d7x7Core(vqf16_src_cx0,  vqf16_src_cx1,  vqf16_src_cx2,  vqf32_result_lo, vqf32_result_hi, kernel, 3);
    Filter2d7x7Core(vqf16_src_n0x0, vqf16_src_n0x1, vqf16_src_n0x2, vqf32_result_lo, vqf32_result_hi, kernel, 4);
    Filter2d7x7Core(vqf16_src_n1x0, vqf16_src_n1x1, vqf16_src_n1x2, vqf32_result_lo, vqf32_result_hi, kernel, 5);
    Filter2d7x7Core(vqf16_src_n2x0, vqf16_src_n2x1, vqf16_src_n2x2, vqf32_result_lo, vqf32_result_hi, kernel, 6);

    float16x4_t vdf16_result_lo = neon::vcvt<MI_F16>(vqf32_result_lo);
    float16x4_t vdf16_result_hi = neon::vcvt<MI_F16>(vqf32_result_hi);

    float16x8_t vqf16_result = neon::vcombine(vdf16_result_lo, vdf16_result_hi);

    return vqf16_result;
}
#endif // AURA_ENABLE_NEON_FP16

template <typename Tp, BorderType BORDER_TYPE, DT_S32 C>
AURA_ALWAYS_INLINE DT_VOID Filter2d7x7OneRow(const Tp *src_p2, const Tp *src_p1, const Tp *src_p0, const Tp *src_c,
                                             const Tp *src_n0, const Tp *src_n1, const Tp *src_n2, Tp *dst, DT_S32 width,
                                             const std::vector<DT_F32> &kdata, const std::vector<Tp> &border_value)
{
    using MVType = typename neon::MQVector<Tp, C>::MVType;

    MVType mvq_src_p2[3], mvq_src_p1[3], mvq_src_p0[3], mvq_c_src[3], mvq_src_n0[3], mvq_src_n1[3], mvq_src_n2[3];
    MVType mvq_result;

    constexpr DT_S32 ELEM_COUNTS = 16 / sizeof(Tp);
    constexpr DT_S32 VOFFSET     = ELEM_COUNTS * C;
    const DT_S32 width_align     = (width & -ELEM_COUNTS) * C;

    // left
    {
        neon::vload(src_p2,           mvq_src_p2[1]);
        neon::vload(src_p2 + VOFFSET, mvq_src_p2[2]);
        neon::vload(src_p1,           mvq_src_p1[1]);
        neon::vload(src_p1 + VOFFSET, mvq_src_p1[2]);
        neon::vload(src_p0,           mvq_src_p0[1]);
        neon::vload(src_p0 + VOFFSET, mvq_src_p0[2]);
        neon::vload(src_c,            mvq_c_src[1]);
        neon::vload(src_c  + VOFFSET, mvq_c_src[2]);
        neon::vload(src_n0,           mvq_src_n0[1]);
        neon::vload(src_n0 + VOFFSET, mvq_src_n0[2]);
        neon::vload(src_n1,           mvq_src_n1[1]);
        neon::vload(src_n1 + VOFFSET, mvq_src_n1[2]);
        neon::vload(src_n2,           mvq_src_n2[1]);
        neon::vload(src_n2 + VOFFSET, mvq_src_n2[2]);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            mvq_src_p2[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mvq_src_p2[1].val[ch], src_p2[ch], border_value[ch]);
            mvq_src_p1[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mvq_src_p1[1].val[ch], src_p1[ch], border_value[ch]);
            mvq_src_p0[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mvq_src_p0[1].val[ch], src_p0[ch], border_value[ch]);
            mvq_c_src[0].val[ch]  = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mvq_c_src[1].val[ch],  src_c[ch],  border_value[ch]);
            mvq_src_n0[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mvq_src_n0[1].val[ch], src_n0[ch], border_value[ch]);
            mvq_src_n1[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mvq_src_n1[1].val[ch], src_n1[ch], border_value[ch]);
            mvq_src_n2[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mvq_src_n2[1].val[ch], src_n2[ch], border_value[ch]);
            mvq_result.val[ch]    = Filter2d7x7Vector(mvq_src_p2[0].val[ch], mvq_src_p2[1].val[ch], mvq_src_p2[2].val[ch],
                                                      mvq_src_p1[0].val[ch], mvq_src_p1[1].val[ch], mvq_src_p1[2].val[ch],
                                                      mvq_src_p0[0].val[ch], mvq_src_p0[1].val[ch], mvq_src_p0[2].val[ch],
                                                      mvq_c_src[0].val[ch],  mvq_c_src[1].val[ch],  mvq_c_src[2].val[ch],
                                                      mvq_src_n0[0].val[ch], mvq_src_n0[1].val[ch], mvq_src_n0[2].val[ch],
                                                      mvq_src_n1[0].val[ch], mvq_src_n1[1].val[ch], mvq_src_n1[2].val[ch],
                                                      mvq_src_n2[0].val[ch], mvq_src_n2[1].val[ch], mvq_src_n2[2].val[ch],
                                                      kdata);
        }

        neon::vstore(dst, mvq_result);

        Filter2d7x7Prepare(mvq_src_p2, mvq_src_p1, mvq_src_p0, mvq_c_src, mvq_src_n0, mvq_src_n1, mvq_src_n2);
    }

    // middle
    {
        for (DT_S32 x = VOFFSET; x < (width_align - VOFFSET); x += VOFFSET)
        {
            neon::vload(src_p2 + x + VOFFSET, mvq_src_p2[2]);
            neon::vload(src_p1 + x + VOFFSET, mvq_src_p1[2]);
            neon::vload(src_p0 + x + VOFFSET, mvq_src_p0[2]);
            neon::vload(src_c  + x + VOFFSET, mvq_c_src[2]);
            neon::vload(src_n0 + x + VOFFSET, mvq_src_n0[2]);
            neon::vload(src_n1 + x + VOFFSET, mvq_src_n1[2]);
            neon::vload(src_n2 + x + VOFFSET, mvq_src_n2[2]);

            for (DT_S32 ch = 0; ch < C; ch++)
            {
                mvq_result.val[ch] = Filter2d7x7Vector(mvq_src_p2[0].val[ch], mvq_src_p2[1].val[ch], mvq_src_p2[2].val[ch],
                                                       mvq_src_p1[0].val[ch], mvq_src_p1[1].val[ch], mvq_src_p1[2].val[ch],
                                                       mvq_src_p0[0].val[ch], mvq_src_p0[1].val[ch], mvq_src_p0[2].val[ch],
                                                       mvq_c_src[0].val[ch],  mvq_c_src[1].val[ch],  mvq_c_src[2].val[ch],
                                                       mvq_src_n0[0].val[ch], mvq_src_n0[1].val[ch], mvq_src_n0[2].val[ch],
                                                       mvq_src_n1[0].val[ch], mvq_src_n1[1].val[ch], mvq_src_n1[2].val[ch],
                                                       mvq_src_n2[0].val[ch], mvq_src_n2[1].val[ch], mvq_src_n2[2].val[ch],
                                                       kdata);
            }

            neon::vstore(dst + x, mvq_result);

            Filter2d7x7Prepare(mvq_src_p2, mvq_src_p1, mvq_src_p0, mvq_c_src, mvq_src_n0, mvq_src_n1, mvq_src_n2);
        }
    }

    // back
    {
        if (width_align != width * C)
        {
            DT_S32 x = (width - (ELEM_COUNTS << 1)) * C;

            neon::vload(src_p2 + x - VOFFSET, mvq_src_p2[0]);
            neon::vload(src_p2 + x,           mvq_src_p2[1]);
            neon::vload(src_p2 + x + VOFFSET, mvq_src_p2[2]);
            neon::vload(src_p1 + x - VOFFSET, mvq_src_p1[0]);
            neon::vload(src_p1 + x,           mvq_src_p1[1]);
            neon::vload(src_p1 + x + VOFFSET, mvq_src_p1[2]);
            neon::vload(src_p0 + x - VOFFSET, mvq_src_p0[0]);
            neon::vload(src_p0 + x,           mvq_src_p0[1]);
            neon::vload(src_p0 + x + VOFFSET, mvq_src_p0[2]);
            neon::vload(src_c  + x - VOFFSET, mvq_c_src[0]);
            neon::vload(src_c  + x,           mvq_c_src[1]);
            neon::vload(src_c  + x + VOFFSET, mvq_c_src[2]);
            neon::vload(src_n0 + x - VOFFSET, mvq_src_n0[0]);
            neon::vload(src_n0 + x,           mvq_src_n0[1]);
            neon::vload(src_n0 + x + VOFFSET, mvq_src_n0[2]);
            neon::vload(src_n1 + x - VOFFSET, mvq_src_n1[0]);
            neon::vload(src_n1 + x,           mvq_src_n1[1]);
            neon::vload(src_n1 + x + VOFFSET, mvq_src_n1[2]);
            neon::vload(src_n2 + x - VOFFSET, mvq_src_n2[0]);
            neon::vload(src_n2 + x,           mvq_src_n2[1]);
            neon::vload(src_n2 + x + VOFFSET, mvq_src_n2[2]);

            for (DT_S32 ch = 0; ch < C; ch++)
            {
                mvq_result.val[ch] = Filter2d7x7Vector(mvq_src_p2[0].val[ch], mvq_src_p2[1].val[ch], mvq_src_p2[2].val[ch],
                                                       mvq_src_p1[0].val[ch], mvq_src_p1[1].val[ch], mvq_src_p1[2].val[ch],
                                                       mvq_src_p0[0].val[ch], mvq_src_p0[1].val[ch], mvq_src_p0[2].val[ch],
                                                       mvq_c_src[0].val[ch],  mvq_c_src[1].val[ch],  mvq_c_src[2].val[ch],
                                                       mvq_src_n0[0].val[ch], mvq_src_n0[1].val[ch], mvq_src_n0[2].val[ch],
                                                       mvq_src_n1[0].val[ch], mvq_src_n1[1].val[ch], mvq_src_n1[2].val[ch],
                                                       mvq_src_n2[0].val[ch], mvq_src_n2[1].val[ch], mvq_src_n2[2].val[ch],
                                                       kdata);
            }

            neon::vstore(dst + x, mvq_result);

            Filter2d7x7Prepare(mvq_src_p2, mvq_src_p1, mvq_src_p0, mvq_c_src, mvq_src_n0, mvq_src_n1, mvq_src_n2);
        }
    }

    // right
    {
        DT_S32 x    = (width - ELEM_COUNTS) * C;
        DT_S32 last = (width - 1) * C;
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            mvq_src_p2[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mvq_src_p2[1].val[ch], src_p2[last], border_value[ch]);
            mvq_src_p1[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mvq_src_p1[1].val[ch], src_p1[last], border_value[ch]);
            mvq_src_p0[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mvq_src_p0[1].val[ch], src_p0[last], border_value[ch]);
            mvq_c_src[2].val[ch]  = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mvq_c_src[1].val[ch],  src_c[last],  border_value[ch]);
            mvq_src_n0[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mvq_src_n0[1].val[ch], src_n0[last], border_value[ch]);
            mvq_src_n1[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mvq_src_n1[1].val[ch], src_n1[last], border_value[ch]);
            mvq_src_n2[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mvq_src_n2[1].val[ch], src_n2[last], border_value[ch]);
            mvq_result.val[ch]    = Filter2d7x7Vector(mvq_src_p2[0].val[ch], mvq_src_p2[1].val[ch], mvq_src_p2[2].val[ch],
                                                      mvq_src_p1[0].val[ch], mvq_src_p1[1].val[ch], mvq_src_p1[2].val[ch],
                                                      mvq_src_p0[0].val[ch], mvq_src_p0[1].val[ch], mvq_src_p0[2].val[ch],
                                                      mvq_c_src[0].val[ch],  mvq_c_src[1].val[ch],  mvq_c_src[2].val[ch],
                                                      mvq_src_n0[0].val[ch], mvq_src_n0[1].val[ch], mvq_src_n0[2].val[ch],
                                                      mvq_src_n1[0].val[ch], mvq_src_n1[1].val[ch], mvq_src_n1[2].val[ch],
                                                      mvq_src_n2[0].val[ch], mvq_src_n2[1].val[ch], mvq_src_n2[2].val[ch],
                                                      kdata);
            last++;
        }

        neon::vstore(dst + x, mvq_result);
    }
}

template <typename Tp, BorderType BORDER_TYPE, DT_S32 C>
AURA_ALWAYS_INLINE DT_VOID Filter2d7x7TwoRow(const Tp *src_p2, const Tp *src_p1, const Tp *src_p0, const Tp *src_c0, const Tp *src_c1,
                                             const Tp *src_n0, const Tp *src_n1, const Tp *src_n2, Tp *dst_c0, Tp *dst_c1, DT_S32 width,
                                             const std::vector<DT_F32> &kdata, const std::vector<Tp> &border_value)
{
    using MVqType = typename neon::MQVector<Tp, C>::MVType;

    MVqType mvq_src_p2[3], mvq_src_p1[3], mvq_src_p0[3], mvq_c0_src[3], mvq_src_c1[3], mvq_src_n0[3], mvq_src_n1[3], mvq_src_n2[3];
    MVqType mvq_result_c0, mvq_result_c1;

    constexpr DT_S32 ELEM_COUNTS = 16 / sizeof(Tp);
    constexpr DT_S32 VOFFSET     = ELEM_COUNTS * C;
    const DT_S32 width_align     = (width & -ELEM_COUNTS) * C;

    // left
    {
        neon::vload(src_p2,           mvq_src_p2[1]);
        neon::vload(src_p2 + VOFFSET, mvq_src_p2[2]);
        neon::vload(src_p1,           mvq_src_p1[1]);
        neon::vload(src_p1 + VOFFSET, mvq_src_p1[2]);
        neon::vload(src_p0,           mvq_src_p0[1]);
        neon::vload(src_p0 + VOFFSET, mvq_src_p0[2]);
        neon::vload(src_c0,           mvq_c0_src[1]);
        neon::vload(src_c0 + VOFFSET, mvq_c0_src[2]);
        neon::vload(src_c1,           mvq_src_c1[1]);
        neon::vload(src_c1 + VOFFSET, mvq_src_c1[2]);
        neon::vload(src_n0,           mvq_src_n0[1]);
        neon::vload(src_n0 + VOFFSET, mvq_src_n0[2]);
        neon::vload(src_n1,           mvq_src_n1[1]);
        neon::vload(src_n1 + VOFFSET, mvq_src_n1[2]);
        neon::vload(src_n2,           mvq_src_n2[1]);
        neon::vload(src_n2 + VOFFSET, mvq_src_n2[2]);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            mvq_src_p2[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mvq_src_p2[1].val[ch], src_p2[ch], border_value[ch]);
            mvq_src_p1[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mvq_src_p1[1].val[ch], src_p1[ch], border_value[ch]);
            mvq_src_p0[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mvq_src_p0[1].val[ch], src_p0[ch], border_value[ch]);
            mvq_c0_src[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mvq_c0_src[1].val[ch], src_c0[ch], border_value[ch]);
            mvq_src_c1[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mvq_src_c1[1].val[ch], src_c1[ch], border_value[ch]);
            mvq_src_n0[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mvq_src_n0[1].val[ch], src_n0[ch], border_value[ch]);
            mvq_src_n1[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mvq_src_n1[1].val[ch], src_n1[ch], border_value[ch]);
            mvq_src_n2[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mvq_src_n2[1].val[ch], src_n2[ch], border_value[ch]);
            mvq_result_c0.val[ch] = Filter2d7x7Vector(mvq_src_p2[0].val[ch], mvq_src_p2[1].val[ch], mvq_src_p2[2].val[ch],
                                                      mvq_src_p1[0].val[ch], mvq_src_p1[1].val[ch], mvq_src_p1[2].val[ch],
                                                      mvq_src_p0[0].val[ch], mvq_src_p0[1].val[ch], mvq_src_p0[2].val[ch],
                                                      mvq_c0_src[0].val[ch], mvq_c0_src[1].val[ch], mvq_c0_src[2].val[ch],
                                                      mvq_src_c1[0].val[ch], mvq_src_c1[1].val[ch], mvq_src_c1[2].val[ch],
                                                      mvq_src_n0[0].val[ch], mvq_src_n0[1].val[ch], mvq_src_n0[2].val[ch],
                                                      mvq_src_n1[0].val[ch], mvq_src_n1[1].val[ch], mvq_src_n1[2].val[ch],
                                                      kdata);
            mvq_result_c1.val[ch] = Filter2d7x7Vector(mvq_src_p1[0].val[ch], mvq_src_p1[1].val[ch], mvq_src_p1[2].val[ch],
                                                      mvq_src_p0[0].val[ch], mvq_src_p0[1].val[ch], mvq_src_p0[2].val[ch],
                                                      mvq_c0_src[0].val[ch], mvq_c0_src[1].val[ch], mvq_c0_src[2].val[ch],
                                                      mvq_src_c1[0].val[ch], mvq_src_c1[1].val[ch], mvq_src_c1[2].val[ch],
                                                      mvq_src_n0[0].val[ch], mvq_src_n0[1].val[ch], mvq_src_n0[2].val[ch],
                                                      mvq_src_n1[0].val[ch], mvq_src_n1[1].val[ch], mvq_src_n1[2].val[ch],
                                                      mvq_src_n2[0].val[ch], mvq_src_n2[1].val[ch], mvq_src_n2[2].val[ch],
                                                      kdata);
        }

        neon::vstore(dst_c0, mvq_result_c0);
        neon::vstore(dst_c1, mvq_result_c1);

        Filter2d7x7Prepare(mvq_src_p2, mvq_src_p1, mvq_src_p0, mvq_c0_src, mvq_src_c1, mvq_src_n0, mvq_src_n1, mvq_src_n2);
    }

    // middle
    {
        for (DT_S32 x = VOFFSET; x < (width_align - VOFFSET); x += VOFFSET)
        {
            neon::vload(src_p2 + x + VOFFSET, mvq_src_p2[2]);
            neon::vload(src_p1 + x + VOFFSET, mvq_src_p1[2]);
            neon::vload(src_p0 + x + VOFFSET, mvq_src_p0[2]);
            neon::vload(src_c0 + x + VOFFSET, mvq_c0_src[2]);
            neon::vload(src_c1 + x + VOFFSET, mvq_src_c1[2]);
            neon::vload(src_n0 + x + VOFFSET, mvq_src_n0[2]);
            neon::vload(src_n1 + x + VOFFSET, mvq_src_n1[2]);
            neon::vload(src_n2 + x + VOFFSET, mvq_src_n2[2]);

            for (DT_S32 ch = 0; ch < C; ch++)
            {
                mvq_result_c0.val[ch] = Filter2d7x7Vector(mvq_src_p2[0].val[ch], mvq_src_p2[1].val[ch], mvq_src_p2[2].val[ch],
                                                          mvq_src_p1[0].val[ch], mvq_src_p1[1].val[ch], mvq_src_p1[2].val[ch],
                                                          mvq_src_p0[0].val[ch], mvq_src_p0[1].val[ch], mvq_src_p0[2].val[ch],
                                                          mvq_c0_src[0].val[ch], mvq_c0_src[1].val[ch], mvq_c0_src[2].val[ch],
                                                          mvq_src_c1[0].val[ch], mvq_src_c1[1].val[ch], mvq_src_c1[2].val[ch],
                                                          mvq_src_n0[0].val[ch], mvq_src_n0[1].val[ch], mvq_src_n0[2].val[ch],
                                                          mvq_src_n1[0].val[ch], mvq_src_n1[1].val[ch], mvq_src_n1[2].val[ch],
                                                          kdata);
                mvq_result_c1.val[ch] = Filter2d7x7Vector(mvq_src_p1[0].val[ch], mvq_src_p1[1].val[ch], mvq_src_p1[2].val[ch],
                                                          mvq_src_p0[0].val[ch], mvq_src_p0[1].val[ch], mvq_src_p0[2].val[ch],
                                                          mvq_c0_src[0].val[ch], mvq_c0_src[1].val[ch], mvq_c0_src[2].val[ch],
                                                          mvq_src_c1[0].val[ch], mvq_src_c1[1].val[ch], mvq_src_c1[2].val[ch],
                                                          mvq_src_n0[0].val[ch], mvq_src_n0[1].val[ch], mvq_src_n0[2].val[ch],
                                                          mvq_src_n1[0].val[ch], mvq_src_n1[1].val[ch], mvq_src_n1[2].val[ch],
                                                          mvq_src_n2[0].val[ch], mvq_src_n2[1].val[ch], mvq_src_n2[2].val[ch],
                                                          kdata);
            }

            neon::vstore(dst_c0 + x, mvq_result_c0);
            neon::vstore(dst_c1 + x, mvq_result_c1);

            Filter2d7x7Prepare(mvq_src_p2, mvq_src_p1, mvq_src_p0, mvq_c0_src, mvq_src_c1, mvq_src_n0, mvq_src_n1, mvq_src_n2);
        }
    }

    // back
    {
        if (width_align != width * C)
        {
            DT_S32 x = (width - (ELEM_COUNTS << 1)) * C;

            neon::vload(src_p2 + x - VOFFSET, mvq_src_p2[0]);
            neon::vload(src_p2 + x,           mvq_src_p2[1]);
            neon::vload(src_p2 + x + VOFFSET, mvq_src_p2[2]);
            neon::vload(src_p1 + x - VOFFSET, mvq_src_p1[0]);
            neon::vload(src_p1 + x,           mvq_src_p1[1]);
            neon::vload(src_p1 + x + VOFFSET, mvq_src_p1[2]);
            neon::vload(src_p0 + x - VOFFSET, mvq_src_p0[0]);
            neon::vload(src_p0 + x,           mvq_src_p0[1]);
            neon::vload(src_p0 + x + VOFFSET, mvq_src_p0[2]);
            neon::vload(src_c0 + x - VOFFSET, mvq_c0_src[0]);
            neon::vload(src_c0 + x,           mvq_c0_src[1]);
            neon::vload(src_c0 + x + VOFFSET, mvq_c0_src[2]);
            neon::vload(src_c1 + x - VOFFSET, mvq_src_c1[0]);
            neon::vload(src_c1 + x,           mvq_src_c1[1]);
            neon::vload(src_c1 + x + VOFFSET, mvq_src_c1[2]);
            neon::vload(src_n0 + x - VOFFSET, mvq_src_n0[0]);
            neon::vload(src_n0 + x,           mvq_src_n0[1]);
            neon::vload(src_n0 + x + VOFFSET, mvq_src_n0[2]);
            neon::vload(src_n1 + x - VOFFSET, mvq_src_n1[0]);
            neon::vload(src_n1 + x,           mvq_src_n1[1]);
            neon::vload(src_n1 + x + VOFFSET, mvq_src_n1[2]);
            neon::vload(src_n2 + x - VOFFSET, mvq_src_n2[0]);
            neon::vload(src_n2 + x,           mvq_src_n2[1]);
            neon::vload(src_n2 + x + VOFFSET, mvq_src_n2[2]);

            for (DT_S32 ch = 0; ch < C; ch++)
            {
                mvq_result_c0.val[ch] = Filter2d7x7Vector(mvq_src_p2[0].val[ch], mvq_src_p2[1].val[ch], mvq_src_p2[2].val[ch],
                                                          mvq_src_p1[0].val[ch], mvq_src_p1[1].val[ch], mvq_src_p1[2].val[ch],
                                                          mvq_src_p0[0].val[ch], mvq_src_p0[1].val[ch], mvq_src_p0[2].val[ch],
                                                          mvq_c0_src[0].val[ch], mvq_c0_src[1].val[ch], mvq_c0_src[2].val[ch],
                                                          mvq_src_c1[0].val[ch], mvq_src_c1[1].val[ch], mvq_src_c1[2].val[ch],
                                                          mvq_src_n0[0].val[ch], mvq_src_n0[1].val[ch], mvq_src_n0[2].val[ch],
                                                          mvq_src_n1[0].val[ch], mvq_src_n1[1].val[ch], mvq_src_n1[2].val[ch],
                                                          kdata);
                mvq_result_c1.val[ch] = Filter2d7x7Vector(mvq_src_p1[0].val[ch], mvq_src_p1[1].val[ch], mvq_src_p1[2].val[ch],
                                                          mvq_src_p0[0].val[ch], mvq_src_p0[1].val[ch], mvq_src_p0[2].val[ch],
                                                          mvq_c0_src[0].val[ch], mvq_c0_src[1].val[ch], mvq_c0_src[2].val[ch],
                                                          mvq_src_c1[0].val[ch], mvq_src_c1[1].val[ch], mvq_src_c1[2].val[ch],
                                                          mvq_src_n0[0].val[ch], mvq_src_n0[1].val[ch], mvq_src_n0[2].val[ch],
                                                          mvq_src_n1[0].val[ch], mvq_src_n1[1].val[ch], mvq_src_n1[2].val[ch],
                                                          mvq_src_n2[0].val[ch], mvq_src_n2[1].val[ch], mvq_src_n2[2].val[ch],
                                                          kdata);
            }

            neon::vstore(dst_c0 + x, mvq_result_c0);
            neon::vstore(dst_c1 + x, mvq_result_c1);

            Filter2d7x7Prepare(mvq_src_p2, mvq_src_p1, mvq_src_p0, mvq_c0_src, mvq_src_c1, mvq_src_n0, mvq_src_n1, mvq_src_n2);
        }
    }

    // right
    {
        DT_S32 x    = (width - ELEM_COUNTS) * C;
        DT_S32 last = (width - 1) * C;
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            mvq_src_p2[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mvq_src_p2[1].val[ch], src_p2[last], border_value[ch]);
            mvq_src_p1[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mvq_src_p1[1].val[ch], src_p1[last], border_value[ch]);
            mvq_src_p0[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mvq_src_p0[1].val[ch], src_p0[last], border_value[ch]);
            mvq_c0_src[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mvq_c0_src[1].val[ch], src_c0[last], border_value[ch]);
            mvq_src_c1[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mvq_src_c1[1].val[ch], src_c1[last], border_value[ch]);
            mvq_src_n0[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mvq_src_n0[1].val[ch], src_n0[last], border_value[ch]);
            mvq_src_n1[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mvq_src_n1[1].val[ch], src_n1[last], border_value[ch]);
            mvq_src_n2[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mvq_src_n2[1].val[ch], src_n2[last], border_value[ch]);
            mvq_result_c0.val[ch] = Filter2d7x7Vector(mvq_src_p2[0].val[ch], mvq_src_p2[1].val[ch], mvq_src_p2[2].val[ch],
                                                      mvq_src_p1[0].val[ch], mvq_src_p1[1].val[ch], mvq_src_p1[2].val[ch],
                                                      mvq_src_p0[0].val[ch], mvq_src_p0[1].val[ch], mvq_src_p0[2].val[ch],
                                                      mvq_c0_src[0].val[ch], mvq_c0_src[1].val[ch], mvq_c0_src[2].val[ch],
                                                      mvq_src_c1[0].val[ch], mvq_src_c1[1].val[ch], mvq_src_c1[2].val[ch],
                                                      mvq_src_n0[0].val[ch], mvq_src_n0[1].val[ch], mvq_src_n0[2].val[ch],
                                                      mvq_src_n1[0].val[ch], mvq_src_n1[1].val[ch], mvq_src_n1[2].val[ch],
                                                      kdata);
            mvq_result_c1.val[ch] = Filter2d7x7Vector(mvq_src_p1[0].val[ch], mvq_src_p1[1].val[ch], mvq_src_p1[2].val[ch],
                                                      mvq_src_p0[0].val[ch], mvq_src_p0[1].val[ch], mvq_src_p0[2].val[ch],
                                                      mvq_c0_src[0].val[ch], mvq_c0_src[1].val[ch], mvq_c0_src[2].val[ch],
                                                      mvq_src_c1[0].val[ch], mvq_src_c1[1].val[ch], mvq_src_c1[2].val[ch],
                                                      mvq_src_n0[0].val[ch], mvq_src_n0[1].val[ch], mvq_src_n0[2].val[ch],
                                                      mvq_src_n1[0].val[ch], mvq_src_n1[1].val[ch], mvq_src_n1[2].val[ch],
                                                      mvq_src_n2[0].val[ch], mvq_src_n2[1].val[ch], mvq_src_n2[2].val[ch],
                                                      kdata);
            last++;
        }

        neon::vstore(dst_c0 + x, mvq_result_c0);
        neon::vstore(dst_c1 + x, mvq_result_c1);
    }
}

template <typename Tp, BorderType BORDER_TYPE, DT_S32 C>
static Status Filter2d7x7NeonImpl(const Mat &src, Mat &dst, const std::vector<DT_F32> &kdata,
                                  const std::vector<Tp> &border_value, const Tp *border_buffer,
                                  DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 width = dst.GetSizes().m_width;

    DT_S32 y = start_row;

    const Tp *src_p2 = src.Ptr<Tp, BORDER_TYPE>(y - 3, border_buffer);
    const Tp *src_p1 = src.Ptr<Tp, BORDER_TYPE>(y - 2, border_buffer);
    const Tp *src_p0 = src.Ptr<Tp, BORDER_TYPE>(y - 1, border_buffer);
    const Tp *src_c0 = src.Ptr<Tp>(y);
    const Tp *src_c1 = src.Ptr<Tp, BORDER_TYPE>(y + 1, border_buffer);
    const Tp *src_n0 = src.Ptr<Tp, BORDER_TYPE>(y + 2, border_buffer);
    const Tp *src_n1 = src.Ptr<Tp, BORDER_TYPE>(y + 3, border_buffer);
    const Tp *src_n2 = src.Ptr<Tp, BORDER_TYPE>(y + 4, border_buffer);

    DT_S32 h_align2 = (end_row - start_row) & (-2);
    for (; y < start_row + h_align2; y += 2)
    {
        Tp *dst_c0 = dst.Ptr<Tp>(y);
        Tp *dst_c1 = dst.Ptr<Tp>(y + 1);
        Filter2d7x7TwoRow<Tp, BORDER_TYPE, C>(src_p2, src_p1, src_p0, src_c0, src_c1, src_n0, src_n1, src_n2,
                                              dst_c0, dst_c1, width, kdata, border_value);

        src_p2 = src_p0;
        src_p1 = src_c0;
        src_p0 = src_c1;
        src_c0 = src_n0;
        src_c1 = src_n1;
        src_n0 = src_n2;
        src_n1 = src.Ptr<Tp, BORDER_TYPE>(y + 5, border_buffer);
        src_n2 = src.Ptr<Tp, BORDER_TYPE>(y + 6, border_buffer);
    }

    src_n0 = src.Ptr<Tp, BORDER_TYPE>(y + 1, border_buffer);
    src_n1 = src.Ptr<Tp, BORDER_TYPE>(y + 2, border_buffer);
    src_n2 = src.Ptr<Tp, BORDER_TYPE>(y + 3, border_buffer);

    if (y < end_row)
    {
        Tp *dst_c0 = dst.Ptr<Tp>(y);
        Filter2d7x7OneRow<Tp, BORDER_TYPE, C>(src_p2, src_p1, src_p0, src_c0, src_n0, src_n1, src_n2, dst_c0, width, kdata, border_value);
    }

    return Status::OK;
}

template <typename Tp, BorderType BORDER_TYPE>
static Status Filter2d7x7NeonHelper(Context *ctx, const Mat &src, Mat &dst, const std::vector<DT_F32> &kdata,
                                    const std::vector<Tp> &border_value, const Tp *border_buffer, const OpTarget &target)
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

    switch (dst.GetSizes().m_channel)
    {
        case 1:
        {
            ret = wp->ParallelFor(0, height, Filter2d7x7NeonImpl<Tp, BORDER_TYPE, 1>, std::cref(src), std::ref(dst),
                                  std::cref(kdata), std::cref(border_value), border_buffer);
            break;
        }

        case 2:
        {
            ret = wp->ParallelFor(0, height, Filter2d7x7NeonImpl<Tp, BORDER_TYPE, 2>, std::cref(src), std::ref(dst),
                                  std::cref(kdata), std::cref(border_value), border_buffer);
            break;
        }

        case 3:
        {
            ret = wp->ParallelFor(0, height, Filter2d7x7NeonImpl<Tp, BORDER_TYPE, 3>, std::cref(src), std::ref(dst),
                                  std::cref(kdata), std::cref(border_value), border_buffer);
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
static Status Filter2d7x7NeonHelper(Context *ctx, const Mat &src, Mat &dst, const std::vector<DT_F32> &kdata,
                                    BorderType border_type, const Scalar &border_value, const OpTarget &target)
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

            ret = Filter2d7x7NeonHelper<Tp, BorderType::CONSTANT>(ctx, src, dst, kdata, vec_border_value, border_buffer, target);

            AURA_FREE(ctx, border_buffer);
            break;
        }

        case BorderType::REPLICATE:
        {
            ret = Filter2d7x7NeonHelper<Tp, BorderType::REPLICATE>(ctx, src, dst, kdata, vec_border_value, border_buffer, target);
            break;
        }

        case BorderType::REFLECT_101:
        {
            ret = Filter2d7x7NeonHelper<Tp, BorderType::REFLECT_101>(ctx, src, dst, kdata, vec_border_value, border_buffer, target);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "Unsupported border type");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

Status Filter2d7x7Neon(Context *ctx, const Mat &src, Mat &dst, const std::vector<DT_F32> &kdata,
                       BorderType border_type, const Scalar &border_value, const OpTarget &target)
{
    Status ret = Status::ERROR;
    DT_S32 pattern = AURA_MAKE_PATTERN(src.GetElemType(), dst.GetElemType());

    switch (pattern)
    {
        case AURA_MAKE_PATTERN(ElemType::U8, ElemType::U8):
        {
            ret = Filter2d7x7NeonHelper<DT_U8>(ctx, src, dst, kdata, border_type, border_value, target);
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::U16, ElemType::U16):
        {
            ret = Filter2d7x7NeonHelper<DT_U16>(ctx, src, dst, kdata, border_type, border_value, target);
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::S16, ElemType::S16):
        {
            ret = Filter2d7x7NeonHelper<DT_S16>(ctx, src, dst, kdata, border_type, border_value, target);
            break;
        }

#if defined(AURA_ENABLE_NEON_FP16)
        case AURA_MAKE_PATTERN(ElemType::F16, ElemType::F16):
        {
            ret = Filter2d7x7NeonHelper<MI_F16>(ctx, src, dst, kdata, border_type, border_value, target);
            break;
        }
#endif // AURA_ENABLE_NEON_FP16

        case AURA_MAKE_PATTERN(ElemType::F32, ElemType::F32):
        {
            ret = Filter2d7x7NeonHelper<DT_F32>(ctx, src, dst, kdata, border_type, border_value, target);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "Unsupported combination of source format and destination format");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura
