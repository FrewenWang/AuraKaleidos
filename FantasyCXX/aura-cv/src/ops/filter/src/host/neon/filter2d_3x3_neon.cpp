#include "aura/ops/core.h"
#include "aura/runtime/mat.h"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

template <typename MVqType>
AURA_ALWAYS_INLINE DT_VOID Filter2d3x3Prepare(MVqType *mvq_src_p, MVqType *mvq_src_c, MVqType *mvq_src_n)
{
    mvq_src_p[0] = mvq_src_p[1];
    mvq_src_c[0] = mvq_src_c[1];
    mvq_src_n[0] = mvq_src_n[1];

    mvq_src_p[1] = mvq_src_p[2];
    mvq_src_c[1] = mvq_src_c[2];
    mvq_src_n[1] = mvq_src_n[2];
}

AURA_ALWAYS_INLINE DT_VOID Filter2d3x3Core(uint8x16_t &vqu8_src_x0, uint8x16_t &vqu8_src_x1, uint8x16_t &vqu8_src_x2,
                                           float32x4_t &vqf32_result_lo_lo, float32x4_t &vqf32_result_lo_hi,
                                           float32x4_t &vqf32_result_hi_lo, float32x4_t &vqf32_result_hi_hi,
                                           const std::vector<DT_F32> &kernel, DT_S32 line)
{
    const DT_S32 idx = 3 * line;

    uint8x16_t vqu8_src_l0      = neon::vext<15>(vqu8_src_x0, vqu8_src_x1);
    uint8x16_t vqu8_src_r0      = neon::vext<1>(vqu8_src_x1, vqu8_src_x2);

    uint16x8_t vqu16_src_l0_lo  = neon::vmovl(neon::vgetlow(vqu8_src_l0));
    uint16x8_t vqu16_src_c_lo   = neon::vmovl(neon::vgetlow(vqu8_src_x1));
    uint16x8_t vqu16_src_r0_lo  = neon::vmovl(neon::vgetlow(vqu8_src_r0));

    float32x4_t vqf32_src_l0_lo = neon::vcvt<DT_F32>(neon::vmovl(neon::vgetlow(vqu16_src_l0_lo)));
    float32x4_t vqf32_src_l0_hi = neon::vcvt<DT_F32>(neon::vmovl(neon::vgethigh(vqu16_src_l0_lo)));
    float32x4_t vqf32_src_c_lo  = neon::vcvt<DT_F32>(neon::vmovl(neon::vgetlow(vqu16_src_c_lo)));
    float32x4_t vqf32_src_c_hi  = neon::vcvt<DT_F32>(neon::vmovl(neon::vgethigh(vqu16_src_c_lo)));
    float32x4_t vqf32_src_r0_lo = neon::vcvt<DT_F32>(neon::vmovl(neon::vgetlow(vqu16_src_r0_lo)));
    float32x4_t vqf32_src_r0_hi = neon::vcvt<DT_F32>(neon::vmovl(neon::vgethigh(vqu16_src_r0_lo)));

    vqf32_result_lo_lo          = neon::vmla(vqf32_result_lo_lo, vqf32_src_l0_lo, kernel[idx + 0]);
    vqf32_result_lo_hi          = neon::vmla(vqf32_result_lo_hi, vqf32_src_l0_hi, kernel[idx + 0]);
    vqf32_result_lo_lo          = neon::vmla(vqf32_result_lo_lo, vqf32_src_c_lo,  kernel[idx + 1]);
    vqf32_result_lo_hi          = neon::vmla(vqf32_result_lo_hi, vqf32_src_c_hi,  kernel[idx + 1]);
    vqf32_result_lo_lo          = neon::vmla(vqf32_result_lo_lo, vqf32_src_r0_lo, kernel[idx + 2]);
    vqf32_result_lo_hi          = neon::vmla(vqf32_result_lo_hi, vqf32_src_r0_hi, kernel[idx + 2]);

    uint16x8_t vqu16_src_l0_hi  = neon::vmovl(neon::vgethigh(vqu8_src_l0));
    uint16x8_t vqu16_src_c_hi   = neon::vmovl(neon::vgethigh(vqu8_src_x1));
    uint16x8_t vqu16_src_r0_hi  = neon::vmovl(neon::vgethigh(vqu8_src_r0));

    vqf32_src_l0_lo             = neon::vcvt<DT_F32>(neon::vmovl(neon::vgetlow(vqu16_src_l0_hi)));
    vqf32_src_l0_hi             = neon::vcvt<DT_F32>(neon::vmovl(neon::vgethigh(vqu16_src_l0_hi)));
    vqf32_src_c_lo              = neon::vcvt<DT_F32>(neon::vmovl(neon::vgetlow(vqu16_src_c_hi)));
    vqf32_src_c_hi              = neon::vcvt<DT_F32>(neon::vmovl(neon::vgethigh(vqu16_src_c_hi)));
    vqf32_src_r0_lo             = neon::vcvt<DT_F32>(neon::vmovl(neon::vgetlow(vqu16_src_r0_hi)));
    vqf32_src_r0_hi             = neon::vcvt<DT_F32>(neon::vmovl(neon::vgethigh(vqu16_src_r0_hi)));

    vqf32_result_hi_lo          = neon::vmla(vqf32_result_hi_lo, vqf32_src_l0_lo, kernel[idx + 0]);
    vqf32_result_hi_hi          = neon::vmla(vqf32_result_hi_hi, vqf32_src_l0_hi, kernel[idx + 0]);
    vqf32_result_hi_lo          = neon::vmla(vqf32_result_hi_lo, vqf32_src_c_lo,  kernel[idx + 1]);
    vqf32_result_hi_hi          = neon::vmla(vqf32_result_hi_hi, vqf32_src_c_hi,  kernel[idx + 1]);
    vqf32_result_hi_lo          = neon::vmla(vqf32_result_hi_lo, vqf32_src_r0_lo, kernel[idx + 2]);
    vqf32_result_hi_hi          = neon::vmla(vqf32_result_hi_hi, vqf32_src_r0_hi, kernel[idx + 2]);
}

AURA_ALWAYS_INLINE uint8x16_t Filter2d3x3Vector(uint8x16_t &vqu8_src_px0, uint8x16_t &vqu8_src_px1, uint8x16_t &vqu8_src_px2,
                                                uint8x16_t &vqu8_src_cx0, uint8x16_t &vqu8_src_cx1, uint8x16_t &vqu8_src_cx2,
                                                uint8x16_t &vqu8_src_nx0, uint8x16_t &vqu8_src_nx1, uint8x16_t &vqu8_src_nx2,
                                                const std::vector<DT_F32> &kernel)
{
    float32x4_t vqf32_result_lo_lo, vqf32_result_lo_hi,  vqf32_result_hi_lo, vqf32_result_hi_hi;
    neon::vdup(vqf32_result_lo_lo, 0.f);
    neon::vdup(vqf32_result_lo_hi, 0.f);
    neon::vdup(vqf32_result_hi_lo, 0.f);
    neon::vdup(vqf32_result_hi_hi, 0.f);

    Filter2d3x3Core(vqu8_src_px0, vqu8_src_px1, vqu8_src_px2, vqf32_result_lo_lo, vqf32_result_lo_hi, vqf32_result_hi_lo, vqf32_result_hi_hi, kernel, 0);
    Filter2d3x3Core(vqu8_src_cx0, vqu8_src_cx1, vqu8_src_cx2, vqf32_result_lo_lo, vqf32_result_lo_hi, vqf32_result_hi_lo, vqf32_result_hi_hi, kernel, 1);
    Filter2d3x3Core(vqu8_src_nx0, vqu8_src_nx1, vqu8_src_nx2, vqf32_result_lo_lo, vqf32_result_lo_hi, vqf32_result_hi_lo, vqf32_result_hi_hi, kernel, 2);

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
AURA_ALWAYS_INLINE DT_VOID Filter2d3x3Core(d16x8_t &vq16_src_x0, d16x8_t &vq16_src_x1, d16x8_t &vq16_src_x2,
                                           float32x4_t &vqf32_result_lo, float32x4_t &vqf32_result_hi,
                                           const std::vector<DT_F32> &kernel, DT_S32 line)
{
    const DT_S32 idx = 3 * line;

    d16x8_t vq16_src_l0 = neon::vext<7>(vq16_src_x0, vq16_src_x1);
    d16x8_t vq16_src_r0 = neon::vext<1>(vq16_src_x1, vq16_src_x2);

    float32x4_t vqf32_src_l0_lo = neon::vcvt<DT_F32>(neon::vmovl(neon::vgetlow(vq16_src_l0)));
    float32x4_t vqf32_src_l0_hi = neon::vcvt<DT_F32>(neon::vmovl(neon::vgethigh(vq16_src_l0)));
    float32x4_t vqf32_src_c_lo  = neon::vcvt<DT_F32>(neon::vmovl(neon::vgetlow(vq16_src_x1)));
    float32x4_t vqf32_src_c_hi  = neon::vcvt<DT_F32>(neon::vmovl(neon::vgethigh(vq16_src_x1)));
    float32x4_t vqf32_src_r0_lo = neon::vcvt<DT_F32>(neon::vmovl(neon::vgetlow(vq16_src_r0)));
    float32x4_t vqf32_src_r0_hi = neon::vcvt<DT_F32>(neon::vmovl(neon::vgethigh(vq16_src_r0)));

    vqf32_result_lo = neon::vmla(vqf32_result_lo, vqf32_src_l0_lo, kernel[idx + 0]);
    vqf32_result_hi = neon::vmla(vqf32_result_hi, vqf32_src_l0_hi, kernel[idx + 0]);
    vqf32_result_lo = neon::vmla(vqf32_result_lo, vqf32_src_c_lo,  kernel[idx + 1]);
    vqf32_result_hi = neon::vmla(vqf32_result_hi, vqf32_src_c_hi,  kernel[idx + 1]);
    vqf32_result_lo = neon::vmla(vqf32_result_lo, vqf32_src_r0_lo, kernel[idx + 2]);
    vqf32_result_hi = neon::vmla(vqf32_result_hi, vqf32_src_r0_hi, kernel[idx + 2]);
}

template <typename d16x8_t, typename D16 = typename neon::Scalar<d16x8_t>::SType,
          typename std::enable_if<(std::is_same<d16x8_t, uint16x8_t>::value || std::is_same<d16x8_t, int16x8_t>::value)>::type* = DT_NULL>
AURA_ALWAYS_INLINE d16x8_t Filter2d3x3Vector(d16x8_t &vq16_src_px0, d16x8_t &vq16_src_px1, d16x8_t &vq16_src_px2,
                                             d16x8_t &vq16_src_cx0, d16x8_t &vq16_src_cx1, d16x8_t &vq16_src_cx2,
                                             d16x8_t &vq16_src_nx0, d16x8_t &vq16_src_nx1, d16x8_t &vq16_src_nx2,
                                             const std::vector<DT_F32> &kernel)
{
    using D32     = typename Promote<D16>::Type;
    using d32x4_t = typename neon::QVector<D32>::VType;

    float32x4_t vqf32_result_lo,  vqf32_result_hi;
    neon::vdup(vqf32_result_lo, 0.f);
    neon::vdup(vqf32_result_hi, 0.f);

    Filter2d3x3Core(vq16_src_px0, vq16_src_px1, vq16_src_px2, vqf32_result_lo, vqf32_result_hi, kernel, 0);
    Filter2d3x3Core(vq16_src_cx0, vq16_src_cx1, vq16_src_cx2, vqf32_result_lo, vqf32_result_hi, kernel, 1);
    Filter2d3x3Core(vq16_src_nx0, vq16_src_nx1, vq16_src_nx2, vqf32_result_lo, vqf32_result_hi, kernel, 2);

    d32x4_t vq32_result_lo = neon::vcvt<D32>(neon::vrndn(vqf32_result_lo));
    d32x4_t vq32_result_hi = neon::vcvt<D32>(neon::vrndn(vqf32_result_hi));

    d16x8_t vq16_result = neon::vcombine(neon::vqmovn(vq32_result_lo), neon::vqmovn(vq32_result_hi));

    return vq16_result;
}

AURA_ALWAYS_INLINE DT_VOID Filter2d3x3Core(float32x4_t &vqf32_src_x0, float32x4_t &vqf32_src_x1, float32x4_t &vqf32_src_x2,
                                           float32x4_t &vqf32_result, const std::vector<DT_F32> &kernel, DT_S32 line)
{
    const DT_S32 idx = 3 * line;

    float32x4_t vqf32_src_l0 = neon::vext<3>(vqf32_src_x0, vqf32_src_x1);
    float32x4_t vqf32_src_r0 = neon::vext<1>(vqf32_src_x1, vqf32_src_x2);

    vqf32_result = neon::vmla(vqf32_result, vqf32_src_l0, kernel[idx + 0]);
    vqf32_result = neon::vmla(vqf32_result, vqf32_src_x1, kernel[idx + 1]);
    vqf32_result = neon::vmla(vqf32_result, vqf32_src_r0, kernel[idx + 2]);
}

AURA_ALWAYS_INLINE float32x4_t Filter2d3x3Vector(float32x4_t &vqf32_src_px0, float32x4_t &vqf32_src_px1, float32x4_t &vqf32_src_px2,
                                                 float32x4_t &vqf32_src_cx0, float32x4_t &vqf32_src_cx1, float32x4_t &vqf32_src_cx2,
                                                 float32x4_t &vqf32_src_nx0, float32x4_t &vqf32_src_nx1, float32x4_t &vqf32_src_nx2,
                                                 const std::vector<DT_F32> &kernel)
{
    float32x4_t vqf32_result;
    neon::vdup(vqf32_result, 0.f);

    Filter2d3x3Core(vqf32_src_px0, vqf32_src_px1, vqf32_src_px2, vqf32_result, kernel, 0);
    Filter2d3x3Core(vqf32_src_cx0, vqf32_src_cx1, vqf32_src_cx2, vqf32_result, kernel, 1);
    Filter2d3x3Core(vqf32_src_nx0, vqf32_src_nx1, vqf32_src_nx2, vqf32_result, kernel, 2);

    return vqf32_result;
}

#if defined(AURA_ENABLE_NEON_FP16)
AURA_ALWAYS_INLINE DT_VOID Filter2d3x3Core(float16x8_t &vqf16_src_x0, float16x8_t &vqf16_src_x1, float16x8_t &vqf16_src_x2,
                                           float32x4_t &vqf32_result_lo, float32x4_t &vqf32_result_hi,
                                           const std::vector<DT_F32> &kernel, DT_S32 line)
{
    const DT_S32 idx = 3 * line;

    float16x8_t vqf16_src_l0 = neon::vext<7>(vqf16_src_x0, vqf16_src_x1);
    float16x8_t vqf16_src_r0 = neon::vext<1>(vqf16_src_x1, vqf16_src_x2);

    float32x4_t vqf32_src_l0_lo = neon::vcvt<DT_F32>(neon::vgetlow(vqf16_src_l0));
    float32x4_t vqf32_src_l0_hi = neon::vcvt<DT_F32>(neon::vgethigh(vqf16_src_l0));
    float32x4_t vqf32_src_c_lo  = neon::vcvt<DT_F32>(neon::vgetlow(vqf16_src_x1));
    float32x4_t vqf32_src_c_hi  = neon::vcvt<DT_F32>(neon::vgethigh(vqf16_src_x1));
    float32x4_t vqf32_src_r0_lo = neon::vcvt<DT_F32>(neon::vgetlow(vqf16_src_r0));
    float32x4_t vqf32_src_r0_hi = neon::vcvt<DT_F32>(neon::vgethigh(vqf16_src_r0));

    vqf32_result_lo = neon::vmla(vqf32_result_lo, vqf32_src_l0_lo, kernel[idx + 0]);
    vqf32_result_hi = neon::vmla(vqf32_result_hi, vqf32_src_l0_hi, kernel[idx + 0]);
    vqf32_result_lo = neon::vmla(vqf32_result_lo, vqf32_src_c_lo,  kernel[idx + 1]);
    vqf32_result_hi = neon::vmla(vqf32_result_hi, vqf32_src_c_hi,  kernel[idx + 1]);
    vqf32_result_lo = neon::vmla(vqf32_result_lo, vqf32_src_r0_lo, kernel[idx + 2]);
    vqf32_result_hi = neon::vmla(vqf32_result_hi, vqf32_src_r0_hi, kernel[idx + 2]);
}

AURA_ALWAYS_INLINE float16x8_t Filter2d3x3Vector(float16x8_t vqf16_src_px0, float16x8_t vqf16_src_px1, float16x8_t vqf16_src_px2,
                                                 float16x8_t vqf16_src_cx0, float16x8_t vqf16_src_cx1, float16x8_t vqf16_src_cx2,
                                                 float16x8_t vqf16_src_nx0, float16x8_t vqf16_src_nx1, float16x8_t vqf16_src_nx2,
                                                 const std::vector<DT_F32> &kernel)
{
    float32x4_t vqf32_result_lo,  vqf32_result_hi;
    neon::vdup(vqf32_result_lo, 0.f);
    neon::vdup(vqf32_result_hi, 0.f);

    Filter2d3x3Core(vqf16_src_px0, vqf16_src_px1, vqf16_src_px2, vqf32_result_lo, vqf32_result_hi, kernel, 0);
    Filter2d3x3Core(vqf16_src_cx0, vqf16_src_cx1, vqf16_src_cx2, vqf32_result_lo, vqf32_result_hi, kernel, 1);
    Filter2d3x3Core(vqf16_src_nx0, vqf16_src_nx1, vqf16_src_nx2, vqf32_result_lo, vqf32_result_hi, kernel, 2);

    float16x4_t vdf16_result_lo = neon::vcvt<MI_F16>(vqf32_result_lo);
    float16x4_t vdf16_result_hi = neon::vcvt<MI_F16>(vqf32_result_hi);

    float16x8_t vqf16_result = neon::vcombine(vdf16_result_lo, vdf16_result_hi);

    return vqf16_result;
}
#endif // AURA_ENABLE_NEON_FP16

template <typename Tp, BorderType BORDER_TYPE, DT_S32 C>
AURA_ALWAYS_INLINE DT_VOID Filter2d3x3Row(const Tp *src_p, const Tp *src_c, const Tp *src_n, Tp *dst, DT_S32 width,
                                          const std::vector<DT_F32> &kdata, const std::vector<Tp> &border_value)
{
    using MVqType = typename neon::MQVector<Tp, C>::MVType;

    MVqType mvq_src_p[3], mvq_src_c[3], mvq_src_n[3];
    MVqType mvq_result;

    constexpr DT_S32 ELEM_COUNTS = 16 / sizeof(Tp);
    constexpr DT_S32 VOFFSET     = ELEM_COUNTS * C;
    const DT_S32 width_align     = (width & -ELEM_COUNTS) * C;

    // left
    {
        neon::vload(src_p,           mvq_src_p[1]);
        neon::vload(src_p + VOFFSET, mvq_src_p[2]);
        neon::vload(src_c,           mvq_src_c[1]);
        neon::vload(src_c + VOFFSET, mvq_src_c[2]);
        neon::vload(src_n,           mvq_src_n[1]);
        neon::vload(src_n + VOFFSET, mvq_src_n[2]);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            mvq_src_p[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mvq_src_p[1].val[ch], src_p[ch], border_value[ch]);
            mvq_src_c[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mvq_src_c[1].val[ch], src_c[ch], border_value[ch]);
            mvq_src_n[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mvq_src_n[1].val[ch], src_n[ch], border_value[ch]);
            mvq_result.val[ch]   = Filter2d3x3Vector(mvq_src_p[0].val[ch], mvq_src_p[1].val[ch], mvq_src_p[2].val[ch],
                                                     mvq_src_c[0].val[ch], mvq_src_c[1].val[ch], mvq_src_c[2].val[ch],
                                                     mvq_src_n[0].val[ch], mvq_src_n[1].val[ch], mvq_src_n[2].val[ch],
                                                     kdata);
        }

        neon::vstore(dst, mvq_result);

        Filter2d3x3Prepare(mvq_src_p, mvq_src_c, mvq_src_n);
    }

    // middle
    {
        for (DT_S32 x = VOFFSET; x < (width_align - VOFFSET); x += VOFFSET)
        {
            neon::vload(src_p + x + VOFFSET, mvq_src_p[2]);
            neon::vload(src_c + x + VOFFSET, mvq_src_c[2]);
            neon::vload(src_n + x + VOFFSET, mvq_src_n[2]);

            for (DT_S32 ch = 0; ch < C; ch++)
            {
                mvq_result.val[ch] = Filter2d3x3Vector(mvq_src_p[0].val[ch], mvq_src_p[1].val[ch], mvq_src_p[2].val[ch],
                                                       mvq_src_c[0].val[ch], mvq_src_c[1].val[ch], mvq_src_c[2].val[ch],
                                                       mvq_src_n[0].val[ch], mvq_src_n[1].val[ch], mvq_src_n[2].val[ch],
                                                       kdata);
            }

            neon::vstore(dst + x, mvq_result);

            Filter2d3x3Prepare(mvq_src_p, mvq_src_c, mvq_src_n);
        }
    }

    // back
    {
        if (width_align != width * C)
        {
            DT_S32 x = (width - (ELEM_COUNTS << 1)) * C;

            neon::vload(src_p + x - VOFFSET, mvq_src_p[0]);
            neon::vload(src_p + x,           mvq_src_p[1]);
            neon::vload(src_p + x + VOFFSET, mvq_src_p[2]);
            neon::vload(src_c + x - VOFFSET, mvq_src_c[0]);
            neon::vload(src_c + x,           mvq_src_c[1]);
            neon::vload(src_c + x + VOFFSET, mvq_src_c[2]);
            neon::vload(src_n + x - VOFFSET, mvq_src_n[0]);
            neon::vload(src_n + x,           mvq_src_n[1]);
            neon::vload(src_n + x + VOFFSET, mvq_src_n[2]);

            for (DT_S32 ch = 0; ch < C; ch++)
            {
                mvq_result.val[ch] = Filter2d3x3Vector(mvq_src_p[0].val[ch], mvq_src_p[1].val[ch], mvq_src_p[2].val[ch],
                                                       mvq_src_c[0].val[ch], mvq_src_c[1].val[ch], mvq_src_c[2].val[ch],
                                                       mvq_src_n[0].val[ch], mvq_src_n[1].val[ch], mvq_src_n[2].val[ch],
                                                       kdata);
            }

            neon::vstore(dst + x, mvq_result);

            Filter2d3x3Prepare(mvq_src_p, mvq_src_c, mvq_src_n);
        }
    }

    // right
    {
        DT_S32 x    = (width - ELEM_COUNTS) * C;
        DT_S32 last = (width - 1) * C;
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            mvq_src_p[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mvq_src_p[1].val[ch], src_p[last], border_value[ch]);
            mvq_src_c[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mvq_src_c[1].val[ch], src_c[last], border_value[ch]);
            mvq_src_n[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mvq_src_n[1].val[ch], src_n[last], border_value[ch]);
            mvq_result.val[ch]   = Filter2d3x3Vector(mvq_src_p[0].val[ch], mvq_src_p[1].val[ch], mvq_src_p[2].val[ch],
                                                     mvq_src_c[0].val[ch], mvq_src_c[1].val[ch], mvq_src_c[2].val[ch],
                                                     mvq_src_n[0].val[ch], mvq_src_n[1].val[ch], mvq_src_n[2].val[ch],
                                                     kdata);
            last++;
        }

        neon::vstore(dst + x, mvq_result);
    }
}

template <typename Tp, BorderType BORDER_TYPE, DT_S32 C>
static Status Filter2d3x3NeonImpl(const Mat &src, Mat &dst, const std::vector<DT_F32> &kdata,
                                  const std::vector<Tp> &border_value, const Tp *border_buffer, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 width = dst.GetSizes().m_width;

    DT_S32 y = start_row;

    const Tp *src_p = src.Ptr<Tp, BORDER_TYPE>(y - 1, border_buffer);
    const Tp *src_c = src.Ptr<Tp>(y);
    const Tp *src_n = src.Ptr<Tp, BORDER_TYPE>(y + 1, border_buffer);

    for (; y < end_row; y++)
    {
        Tp *dst_c = dst.Ptr<Tp>(y);
        Filter2d3x3Row<Tp, BORDER_TYPE, C>(src_p, src_c, src_n, dst_c, width, kdata, border_value);

        src_p = src_c;
        src_c = src_n;
        src_n = src.Ptr<Tp, BORDER_TYPE>(y + 2, border_buffer);
    }

    return Status::OK;
}

template <typename Tp, BorderType BORDER_TYPE>
static Status Filter2d3x3NeonHelper(Context *ctx, const Mat &src, Mat &dst, const std::vector<DT_F32> &kdata,
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
            ret = wp->ParallelFor(0, height, Filter2d3x3NeonImpl<Tp, BORDER_TYPE, 1>, std::cref(src), std::ref(dst),
                                  std::cref(kdata), std::cref(border_value), border_buffer);
            break;
        }

        case 2:
        {
            ret = wp->ParallelFor(0, height, Filter2d3x3NeonImpl<Tp, BORDER_TYPE, 2>, std::cref(src), std::ref(dst),
                                  std::cref(kdata), std::cref(border_value), border_buffer);
            break;
        }

        case 3:
        {
            ret = wp->ParallelFor(0, height, Filter2d3x3NeonImpl<Tp, BORDER_TYPE, 3>, std::cref(src), std::ref(dst),
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
static Status Filter2d3x3NeonHelper(Context *ctx, const Mat &src, Mat &dst, const std::vector<DT_F32> &kdata,
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

            ret = Filter2d3x3NeonHelper<Tp, BorderType::CONSTANT>(ctx, src, dst, kdata, vec_border_value, border_buffer, target);

            AURA_FREE(ctx, border_buffer);
            break;
        }

        case BorderType::REPLICATE:
        {
            ret = Filter2d3x3NeonHelper<Tp, BorderType::REPLICATE>(ctx, src, dst, kdata, vec_border_value, border_buffer, target);
            break;
        }

        case BorderType::REFLECT_101:
        {
            ret = Filter2d3x3NeonHelper<Tp, BorderType::REFLECT_101>(ctx, src, dst, kdata, vec_border_value, border_buffer, target);
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

Status Filter2d3x3Neon(Context *ctx, const Mat &src, Mat &dst, const std::vector<DT_F32> &kdata,
                       BorderType border_type, const Scalar &border_value, const OpTarget &target)
{
    Status ret = Status::ERROR;
    DT_S32 pattern = AURA_MAKE_PATTERN(src.GetElemType(), dst.GetElemType());

    switch (pattern)
    {
        case AURA_MAKE_PATTERN(ElemType::U8, ElemType::U8):
        {
            ret = Filter2d3x3NeonHelper<DT_U8>(ctx, src, dst, kdata, border_type, border_value, target);
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::U16, ElemType::U16):
        {
            ret = Filter2d3x3NeonHelper<DT_U16>(ctx, src, dst, kdata, border_type, border_value, target);
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::S16, ElemType::S16):
        {
            ret = Filter2d3x3NeonHelper<DT_S16>(ctx, src, dst, kdata, border_type, border_value, target);
            break;
        }

#if defined(AURA_ENABLE_NEON_FP16)
        case AURA_MAKE_PATTERN(ElemType::F16, ElemType::F16):
        {
            ret = Filter2d3x3NeonHelper<MI_F16>(ctx, src, dst, kdata, border_type, border_value, target);
            break;
        }
#endif // AURA_ENABLE_NEON_FP16

        case AURA_MAKE_PATTERN(ElemType::F32, ElemType::F32):
        {
            ret = Filter2d3x3NeonHelper<DT_F32>(ctx, src, dst, kdata, border_type, border_value, target);
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
