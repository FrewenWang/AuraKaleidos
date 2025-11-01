#include "cvtcolor_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"

namespace aura
{

template <MI_U32 MODE>
AURA_ALWAYS_INLINE AURA_VOID CvtRgb2Y(uint8x16x3_t &v3qu8_rgb, MI_U8 *dy)
{
    uint16x8x3_t v3qu16_rgb_lo;
    uint16x8x3_t v3qu16_rgb_hi;

    uint32x4x3_t v3qu32_rgb_lo_lo;
    uint32x4x3_t v3qu32_rgb_lo_hi;

    uint32x4x3_t v3qu32_rgb_hi_lo;
    uint32x4x3_t v3qu32_rgb_hi_hi;

    constexpr MI_S32 R2Y = Rgb2YuvParamTraits<MODE>::R2Y;
    constexpr MI_S32 G2Y = Rgb2YuvParamTraits<MODE>::G2Y;
    constexpr MI_S32 B2Y = Rgb2YuvParamTraits<MODE>::B2Y;
    constexpr MI_S32 YC  = Rgb2YuvParamTraits<MODE>::YC;

    uint32x4_t vqu32_cont_uv, vqu32_scale;
    neon::vdup(vqu32_cont_uv, static_cast<MI_U32>(YC));
    neon::vdup(vqu32_scale, static_cast<MI_U32>(1 << (CVTCOLOR_COEF_BITS - 1)));

    // u8->u16
    v3qu16_rgb_lo.val[0] = neon::vmovl(neon::vgetlow(v3qu8_rgb.val[0]));
    v3qu16_rgb_lo.val[1] = neon::vmovl(neon::vgetlow(v3qu8_rgb.val[1]));
    v3qu16_rgb_lo.val[2] = neon::vmovl(neon::vgetlow(v3qu8_rgb.val[2]));

    v3qu16_rgb_hi.val[0] = neon::vmovl(neon::vgethigh(v3qu8_rgb.val[0]));
    v3qu16_rgb_hi.val[1] = neon::vmovl(neon::vgethigh(v3qu8_rgb.val[1]));
    v3qu16_rgb_hi.val[2] = neon::vmovl(neon::vgethigh(v3qu8_rgb.val[2]));

    // u16->u32: low 16x4
    v3qu32_rgb_lo_lo.val[0] = neon::vmovl(neon::vgetlow(v3qu16_rgb_lo.val[0]));
    v3qu32_rgb_lo_lo.val[1] = neon::vmovl(neon::vgetlow(v3qu16_rgb_lo.val[1]));
    v3qu32_rgb_lo_lo.val[2] = neon::vmovl(neon::vgetlow(v3qu16_rgb_lo.val[2]));

    // u16->u32: high 16x4
    v3qu32_rgb_lo_hi.val[0] = neon::vmovl(neon::vgethigh(v3qu16_rgb_lo.val[0]));
    v3qu32_rgb_lo_hi.val[1] = neon::vmovl(neon::vgethigh(v3qu16_rgb_lo.val[1]));
    v3qu32_rgb_lo_hi.val[2] = neon::vmovl(neon::vgethigh(v3qu16_rgb_lo.val[2]));

    // u16->u32: low 16x4
    v3qu32_rgb_hi_lo.val[0] = neon::vmovl(neon::vgetlow(v3qu16_rgb_hi.val[0]));
    v3qu32_rgb_hi_lo.val[1] = neon::vmovl(neon::vgetlow(v3qu16_rgb_hi.val[1]));
    v3qu32_rgb_hi_lo.val[2] = neon::vmovl(neon::vgetlow(v3qu16_rgb_hi.val[2]));

    // u16->u32: high 16x4
    v3qu32_rgb_hi_hi.val[0] = neon::vmovl(neon::vgethigh(v3qu16_rgb_hi.val[0]));
    v3qu32_rgb_hi_hi.val[1] = neon::vmovl(neon::vgethigh(v3qu16_rgb_hi.val[1]));
    v3qu32_rgb_hi_hi.val[2] = neon::vmovl(neon::vgethigh(v3qu16_rgb_hi.val[2]));

    // r2y * r
    uint32x4_t vqu32_y_lo_lo = neon::vmul(v3qu32_rgb_lo_lo.val[0], static_cast<MI_U32>(R2Y));
    uint32x4_t vqu32_y_lo_hi = neon::vmul(v3qu32_rgb_lo_hi.val[0], static_cast<MI_U32>(R2Y));

    vqu32_y_lo_lo = neon::vmla(vqu32_y_lo_lo, v3qu32_rgb_lo_lo.val[1], static_cast<MI_U32>(G2Y));
    vqu32_y_lo_hi = neon::vmla(vqu32_y_lo_hi, v3qu32_rgb_lo_hi.val[1], static_cast<MI_U32>(G2Y));

    vqu32_y_lo_lo = neon::vmla(vqu32_y_lo_lo, v3qu32_rgb_lo_lo.val[2], static_cast<MI_U32>(B2Y));
    vqu32_y_lo_hi = neon::vmla(vqu32_y_lo_hi, v3qu32_rgb_lo_hi.val[2], static_cast<MI_U32>(B2Y));

    uint32x4_t vqu32_y_hi_lo = neon::vmul(v3qu32_rgb_hi_lo.val[0], static_cast<MI_U32>(R2Y));
    uint32x4_t vqu32_y_hi_hi = neon::vmul(v3qu32_rgb_hi_hi.val[0], static_cast<MI_U32>(R2Y));

    vqu32_y_hi_lo = neon::vmla(vqu32_y_hi_lo, v3qu32_rgb_hi_lo.val[1], static_cast<MI_U32>(G2Y));
    vqu32_y_hi_hi = neon::vmla(vqu32_y_hi_hi, v3qu32_rgb_hi_hi.val[1], static_cast<MI_U32>(G2Y));

    vqu32_y_hi_lo = neon::vmla(vqu32_y_hi_lo, v3qu32_rgb_hi_lo.val[2], static_cast<MI_U32>(B2Y));
    vqu32_y_hi_hi = neon::vmla(vqu32_y_hi_hi, v3qu32_rgb_hi_hi.val[2], static_cast<MI_U32>(B2Y));

    uint16x4_t vdu16_y_lo_lo = neon::vshrn_n<5>(neon::vshr_n<15>(neon::vadd(neon::vadd(vqu32_cont_uv, vqu32_y_lo_lo), vqu32_scale)));
    uint16x4_t vdu16_y_lo_hi = neon::vshrn_n<5>(neon::vshr_n<15>(neon::vadd(neon::vadd(vqu32_cont_uv, vqu32_y_lo_hi), vqu32_scale)));
    uint8x8_t  vdu8_y_lo     = neon::vqmovn(neon::vcombine(vdu16_y_lo_lo, vdu16_y_lo_hi));

    uint16x4_t vdu16_y_hi_lo = neon::vshrn_n<5>(neon::vshr_n<15>(neon::vadd(neon::vadd(vqu32_cont_uv, vqu32_y_hi_lo), vqu32_scale)));
    uint16x4_t vdu16_y_hi_hi = neon::vshrn_n<5>(neon::vshr_n<15>(neon::vadd(neon::vadd(vqu32_cont_uv, vqu32_y_hi_hi), vqu32_scale)));
    uint8x8_t  vdu8_y_hi     = neon::vqmovn(neon::vcombine(vdu16_y_hi_lo, vdu16_y_hi_hi));

    uint8x16_t vqu8_y = neon::vcombine(vdu8_y_lo, vdu8_y_hi);
    neon::vstore(dy, vqu8_y);
}

template <MI_U32 MODE>
AURA_ALWAYS_INLINE AURA_VOID CvtRgb2Uv(const uint8x8_t &vdu8_r, const uint8x8_t &vdu8_g, const uint8x8_t &vdu8_b, int32x4_t &vqs32_uv_const,
                                     uint8x8_t &vdu8_u, uint8x8_t &vdu8_v)
{
    uint16x8x3_t v3qu16_rgb;
    int32x4x3_t  v3qs32_rgb_lo, v3qs32_rgb_hi;

    constexpr MI_S32 R2U = Rgb2YuvParamTraits<MODE>::R2U;
    constexpr MI_S32 G2U = Rgb2YuvParamTraits<MODE>::G2U;
    constexpr MI_S32 B2U = Rgb2YuvParamTraits<MODE>::B2U;
    constexpr MI_S32 G2V = Rgb2YuvParamTraits<MODE>::G2V;
    constexpr MI_S32 B2V = Rgb2YuvParamTraits<MODE>::B2V;

    int32x4_t vqs32_scale;
    neon::vdup(vqs32_scale, static_cast<MI_S32>(1 << (CVTCOLOR_COEF_BITS - 1)));

    // u8->u16
    v3qu16_rgb.val[0] = neon::vmovl(vdu8_r);
    v3qu16_rgb.val[1] = neon::vmovl(vdu8_g);
    v3qu16_rgb.val[2] = neon::vmovl(vdu8_b);

    // u16->s32
    v3qs32_rgb_lo.val[0] = neon::vmovl(neon::vreinterpret(neon::vgetlow(v3qu16_rgb.val[0])));
    v3qs32_rgb_hi.val[0] = neon::vmovl(neon::vreinterpret(neon::vgethigh(v3qu16_rgb.val[0])));

    v3qs32_rgb_lo.val[1] = neon::vmovl(neon::vreinterpret(neon::vgetlow(v3qu16_rgb.val[1])));
    v3qs32_rgb_hi.val[1] = neon::vmovl(neon::vreinterpret(neon::vgethigh(v3qu16_rgb.val[1])));

    v3qs32_rgb_lo.val[2] = neon::vmovl(neon::vreinterpret(neon::vgetlow(v3qu16_rgb.val[2])));
    v3qs32_rgb_hi.val[2] = neon::vmovl(neon::vreinterpret(neon::vgethigh(v3qu16_rgb.val[2])));

    int32x4_t vqs32_u_lo = neon::vadd(neon::vmul(v3qs32_rgb_lo.val[2], static_cast<MI_S32>(B2U)), vqs32_uv_const);
    int32x4_t vqs32_u_hi = neon::vadd(neon::vmul(v3qs32_rgb_hi.val[2], static_cast<MI_S32>(B2U)), vqs32_uv_const);

    vqs32_u_lo = neon::vmls(vqs32_u_lo, v3qs32_rgb_lo.val[0], static_cast<MI_S32>(-R2U));
    vqs32_u_hi = neon::vmls(vqs32_u_hi, v3qs32_rgb_hi.val[0], static_cast<MI_S32>(-R2U));

    vqs32_u_lo = neon::vmls(vqs32_u_lo, v3qs32_rgb_lo.val[1], static_cast<MI_S32>(-G2U));
    vqs32_u_hi = neon::vmls(vqs32_u_hi, v3qs32_rgb_hi.val[1], static_cast<MI_S32>(-G2U));

    vdu8_u = neon::vqmovun(neon::vcombine(neon::vshrn_n<5>(neon::vshr_n<15>(neon::vadd(vqs32_u_lo, vqs32_scale))),
                                          neon::vshrn_n<5>(neon::vshr_n<15>(neon::vadd(vqs32_u_hi, vqs32_scale)))));

    int32x4_t vqs32_v_lo = neon::vadd(neon::vmul(v3qs32_rgb_lo.val[0], static_cast<MI_S32>(B2U)), vqs32_uv_const);
    int32x4_t vqs32_v_hi = neon::vadd(neon::vmul(v3qs32_rgb_hi.val[0], static_cast<MI_S32>(B2U)), vqs32_uv_const);

    vqs32_v_lo = neon::vmls(vqs32_v_lo, v3qs32_rgb_lo.val[1], static_cast<MI_S32>(-G2V));
    vqs32_v_hi = neon::vmls(vqs32_v_hi, v3qs32_rgb_hi.val[1], static_cast<MI_S32>(-G2V));

    vqs32_v_lo = neon::vmls(vqs32_v_lo, v3qs32_rgb_lo.val[2], static_cast<MI_S32>(-B2V));
    vqs32_v_hi = neon::vmls(vqs32_v_hi, v3qs32_rgb_hi.val[2], static_cast<MI_S32>(-B2V));

    vdu8_v = neon::vqmovun(neon::vcombine(neon::vshrn_n<5>(neon::vshr_n<15>(neon::vadd(vqs32_v_lo, vqs32_scale))),
                                          neon::vshrn_n<5>(neon::vshr_n<15>(neon::vadd(vqs32_v_hi, vqs32_scale)))));
}

template <MI_U32 MODE>
static Status CvtRgb2NvNeonImpl(const Mat &src, Mat &dst_y, Mat &dst_uv, MI_S32 uv_const, MI_BOOL swapuv, MI_S32 start_row, MI_S32 end_row)
{
    MI_S32 width       = src.GetSizes().m_width;
    MI_S32 ichannel    = src.GetSizes().m_channel;
    MI_S32 width_align = width & (-16);
    MI_S32 offset      = 0;

    const MI_U8 index[8] = {0, 2, 4, 6, 8, 10, 12, 14};

    const MI_S32 uidx = swapuv;
    const MI_S32 vidx = 1 - uidx;

    int32x4_t vqs32_uv_const;
    neon::vdup(vqs32_uv_const, uv_const);

    for (MI_S32 y = start_row; y < end_row; ++y)
    {
        MI_S32 y_idx = y << 1;

        const MI_U8 *src_c = src.Ptr<MI_U8>(y_idx);
        const MI_U8 *src_n = src.Ptr<MI_U8>(y_idx + 1);

        // y line ptr
        MI_U8 *dsty_c = dst_y.Ptr<MI_U8>(y_idx);
        MI_U8 *dsty_n = dst_y.Ptr<MI_U8>(y_idx + 1);
        // uv
        MI_U8 *dst_uv_c = dst_uv.Ptr<MI_U8>(y);

        MI_S32 x = 0;
        for (; x < width_align; x += 16)
        {
LOOP_BODY:
            offset = x * ichannel;

            // even line
            uint8x16x3_t v3qu8_src_lo = neon::vload3q(src_c + offset);
            CvtRgb2Y<MODE>(v3qu8_src_lo, dsty_c + x);

            // select uv's input
            uint8x8_t   vdu8_index = neon::vload1(index);
            uint8x8x3_t v3du8_uv_in;
            uint8x8x2_t v2du8_tab;

            // uv r
            v2du8_tab.val[0]   = neon::vgetlow(v3qu8_src_lo.val[0]);
            v2du8_tab.val[1]   = neon::vgethigh(v3qu8_src_lo.val[0]);
            v3du8_uv_in.val[0] = neon::vtbl(v2du8_tab, vdu8_index);

            // uv g
            v2du8_tab.val[0]   = neon::vgetlow(v3qu8_src_lo.val[1]);
            v2du8_tab.val[1]   = neon::vgethigh(v3qu8_src_lo.val[1]);
            v3du8_uv_in.val[1] = neon::vtbl(v2du8_tab, vdu8_index);

            // uv b
            v2du8_tab.val[0]   = neon::vgetlow(v3qu8_src_lo.val[2]);
            v2du8_tab.val[1]   = neon::vgethigh(v3qu8_src_lo.val[2]);
            v3du8_uv_in.val[2] = neon::vtbl(v2du8_tab, vdu8_index);

            uint8x8x2_t v2du8_dst_uv;
            uint8x8_t   vdu8_u, vdu8_v;
            CvtRgb2Uv<MODE>(v3du8_uv_in.val[0], v3du8_uv_in.val[1], v3du8_uv_in.val[2], vqs32_uv_const, vdu8_u, vdu8_v);

            v2du8_dst_uv.val[uidx] = vdu8_u;
            v2du8_dst_uv.val[vidx] = vdu8_v;
            neon::vstore(dst_uv_c + x, v2du8_dst_uv);

            // odd line
            uint8x16x3_t v3qu8_src_hi = neon::vload3q(src_n + offset);
            CvtRgb2Y<MODE>(v3qu8_src_hi, dsty_n + x);
        }

        // the remaining part
        if (x < width)
        {
            x = width - 16;
            goto LOOP_BODY;
        }
    }

    return Status::OK;
}

Status CvtRgb2NvNeon(Context *ctx, const Mat &src, Mat &dst_y, Mat &dst_uv, MI_BOOL swapuv, CvtColorType type, const OpTarget &target)
{
    AURA_UNUSED(target);
    Status ret = Status::ERROR;

    if ((src.GetSizes().m_width & 1) || (src.GetSizes().m_height & 1))
    {
        AURA_ADD_ERROR_STRING(ctx, "src sizes only support even");
        return ret;
    }

    if (dst_y.GetSizes() * Sizes3(1, 1, 2) != dst_uv.GetSizes() * Sizes3(2, 2, 1) ||
        dst_y.GetSizes() * Sizes3(1, 1, 3) != src.GetSizes() ||
        dst_y.GetSizes().m_channel != 1)
    {
        AURA_ADD_ERROR_STRING(ctx, "the sizes of src and dst do not match");
        return ret;
    }

    MI_S32 height   = dst_uv.GetSizes().m_height;
    MI_S32 uv_const = 128 * (1 << CVTCOLOR_COEF_BITS);

    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return ret;
    }

    switch (type)
    {
        case CvtColorType::RGB2YUV_NV12:
        case CvtColorType::RGB2YUV_NV21:
        {
            ret = wp->ParallelFor((MI_S32)0, height, CvtRgb2NvNeonImpl<0>, std::cref(src), std::ref(dst_y),
                                  std::ref(dst_uv), uv_const, swapuv);
            break;
        }

        case CvtColorType::RGB2YUV_NV12_601:
        case CvtColorType::RGB2YUV_NV21_601:
        {
            ret = wp->ParallelFor((MI_S32)0, height, CvtRgb2NvNeonImpl<1>, std::cref(src), std::ref(dst_y),
                                  std::ref(dst_uv), uv_const, swapuv);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "cvtcolor type error");
            ret = Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

template <MI_U32 MODE>
static Status CvtRgb2Y420NeonImpl(const Mat &src, Mat &dst_y, Mat &dst_u, Mat &dst_v, MI_S32 uv_const, MI_BOOL swapuv, MI_S32 start_row, MI_S32 end_row)
{
    MI_S32 width       = src.GetSizes().m_width;
    MI_S32 ichannel    = src.GetSizes().m_channel;
    MI_S32 width_align = width & (-16);
    MI_S32 offset      = 0;

    const MI_U8 index[8] = {0, 2, 4, 6, 8, 10, 12, 14};

    int32x4_t vqs32_uv_const;
    neon::vdup(vqs32_uv_const, uv_const);

    for (MI_S32 y = start_row; y < end_row; ++y)
    {
        MI_S32 y_idx = y << 1;

        const MI_U8 *src_c = src.Ptr<MI_U8>(y_idx);
        const MI_U8 *src_n = src.Ptr<MI_U8>(y_idx + 1);

        // y line ptr
        MI_U8 *dsty_c = dst_y.Ptr<MI_U8>(y_idx);
        MI_U8 *dsty_n = dst_y.Ptr<MI_U8>(y_idx + 1);
        // uv
        MI_U8 *dstu_c = swapuv ? dst_v.Ptr<MI_U8>(y) : dst_u.Ptr<MI_U8>(y);
        MI_U8 *dstv_c = swapuv ? dst_u.Ptr<MI_U8>(y) : dst_v.Ptr<MI_U8>(y);

        MI_S32 x = 0;
        for (; x < width_align; x += 16)
        {
LOOP_BODY:
            offset = x * ichannel;

            // even line
            uint8x16x3_t v3qu8_src_lo = neon::vload3q(src_c + offset);
            CvtRgb2Y<MODE>(v3qu8_src_lo, dsty_c + x);

            // select uv's input
            uint8x8_t   vdu8_index = neon::vload1(index);
            uint8x8x3_t v3du8_uv_in;
            uint8x8x2_t v2du8_tab;

            // uv r
            v2du8_tab.val[0]   = neon::vgetlow(v3qu8_src_lo.val[0]);
            v2du8_tab.val[1]   = neon::vgethigh(v3qu8_src_lo.val[0]);
            v3du8_uv_in.val[0] = neon::vtbl(v2du8_tab, vdu8_index);

            // uv g
            v2du8_tab.val[0]   = neon::vgetlow(v3qu8_src_lo.val[1]);
            v2du8_tab.val[1]   = neon::vgethigh(v3qu8_src_lo.val[1]);
            v3du8_uv_in.val[1] = neon::vtbl(v2du8_tab, vdu8_index);

            // uv b
            v2du8_tab.val[0]   = neon::vgetlow(v3qu8_src_lo.val[2]);
            v2du8_tab.val[1]   = neon::vgethigh(v3qu8_src_lo.val[2]);
            v3du8_uv_in.val[2] = neon::vtbl(v2du8_tab, vdu8_index);

            uint8x8_t vdu8_dst_u, vdu8_dst_v;
            CvtRgb2Uv<MODE>(v3du8_uv_in.val[0], v3du8_uv_in.val[1], v3du8_uv_in.val[2], vqs32_uv_const, vdu8_dst_u, vdu8_dst_v);
            neon::vstore(dstu_c + (x >> 1), vdu8_dst_u);
            neon::vstore(dstv_c + (x >> 1), vdu8_dst_v);

            // odd line
            uint8x16x3_t v3qu8_src_hi = neon::vload3q(src_n + offset);
            CvtRgb2Y<MODE>(v3qu8_src_hi, dsty_n + x);
        }

        // the remaining part
        if (x < width)
        {
            x = width - 16;
            goto LOOP_BODY;
        }
    }

    return Status::OK;
}

Status CvtRgb2Y420Neon(Context *ctx, const Mat &src, Mat &dst_y, Mat &dst_u, Mat &dst_v, MI_BOOL swapuv, CvtColorType type, const OpTarget &target)
{
    AURA_UNUSED(target);
    Status ret = Status::ERROR;

    if ((src.GetSizes().m_width & 1) || (src.GetSizes().m_height & 1))
    {
        AURA_ADD_ERROR_STRING(ctx, "src and dst mats size only support even");
        return ret;
    }

    if (dst_y.GetSizes() != dst_u.GetSizes() * Sizes3(2, 2, 1) ||
        dst_y.GetSizes() != dst_v.GetSizes() * Sizes3(2, 2, 1) ||
        dst_y.GetSizes() * Sizes3(1, 1, 3) != src.GetSizes() ||
        dst_y.GetSizes().m_channel != 1)
    {
        AURA_ADD_ERROR_STRING(ctx, "the sizes of src and dst do not match");
        return ret;
    }

    MI_S32 height   = dst_u.GetSizes().m_height;
    MI_S32 uv_const = 128 * (1 << CVTCOLOR_COEF_BITS);

    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return ret;
    }

    switch (type)
    {
        case CvtColorType::RGB2YUV_YU12:
        case CvtColorType::RGB2YUV_YV12:
        {
            ret = wp->ParallelFor((MI_S32)0, height, CvtRgb2Y420NeonImpl<0>, std::cref(src), std::ref(dst_y), std::ref(dst_u),
                                  std::ref(dst_v), uv_const, swapuv);
            break;
        }

        case CvtColorType::RGB2YUV_YU12_601:
        case CvtColorType::RGB2YUV_YV12_601:
        {
            ret = wp->ParallelFor((MI_S32)0, height, CvtRgb2Y420NeonImpl<1>, std::cref(src), std::ref(dst_y), std::ref(dst_u),
                                  std::ref(dst_v), uv_const, swapuv);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "cvtcolor type error");
            ret = Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

template <MI_U32 MODE>
static Status CvtRgb2Y444NeonImpl(const Mat &src, Mat &dst_y, Mat &dst_u, Mat &dst_v, MI_S32 uv_const, MI_S32 start_row, MI_S32 end_row)
{
    MI_S32 width       = src.GetSizes().m_width;
    MI_S32 ichannel    = src.GetSizes().m_channel;
    MI_S32 width_align = width & (-16);
    MI_S32 offset      = 0;

    int32x4_t vqs32_uv_const;
    neon::vdup(vqs32_uv_const, uv_const);

    for (MI_S32 y = start_row; y < end_row; ++y)
    {
        const MI_U8 *src_c = src.Ptr<MI_U8>(y);

        // y line ptr
        MI_U8 *dsty_c = dst_y.Ptr<MI_U8>(y);
        // uv
        MI_U8 *dstu_c = dst_u.Ptr<MI_U8>(y);
        MI_U8 *dstv_c = dst_v.Ptr<MI_U8>(y);

        MI_S32 x = 0;
        for (; x < width_align; x += 16)
        {
LOOP_BODY:
            offset = x * ichannel;

            uint8x16x3_t v3qu8_src = neon::vload3q(src_c + offset);
            CvtRgb2Y<MODE>(v3qu8_src, dsty_c + x);

            uint8x8_t vdu8_u_lo, vdu8_u_hi, vdu8_v_lo, vdu8_v_hi;
            CvtRgb2Uv<MODE>(neon::vgetlow(v3qu8_src.val[0]), neon::vgetlow(v3qu8_src.val[1]), neon::vgetlow(v3qu8_src.val[2]),
                            vqs32_uv_const, vdu8_u_lo, vdu8_v_lo);
            CvtRgb2Uv<MODE>(neon::vgethigh(v3qu8_src.val[0]), neon::vgethigh(v3qu8_src.val[1]), neon::vgethigh(v3qu8_src.val[2]),
                            vqs32_uv_const, vdu8_u_hi, vdu8_v_hi);

            uint8x16_t vqu8_dst_u = neon::vcombine(vdu8_u_lo, vdu8_u_hi);
            neon::vstore(dstu_c + x, vqu8_dst_u);

            uint8x16_t vqu8_dst_v = neon::vcombine(vdu8_v_lo, vdu8_v_hi);
            neon::vstore(dstv_c + x, vqu8_dst_v);
        }

        // the remaining part
        if (x < width)
        {
            x = width - 16;
            goto LOOP_BODY;
        }
    }

    return Status::OK;
}

Status CvtRgb2Y444Neon(Context *ctx, const Mat &src, Mat &dst_y, Mat &dst_u, Mat &dst_v, CvtColorType type, const OpTarget &target)
{
    AURA_UNUSED(target);
    Status ret = Status::ERROR;

    if (dst_y.GetSizes() != dst_u.GetSizes() ||
        dst_y.GetSizes() != dst_v.GetSizes() ||
        dst_y.GetSizes() * Sizes3(1, 1, 3) != src.GetSizes() ||
        dst_y.GetSizes().m_channel != 1)
    {
        AURA_ADD_ERROR_STRING(ctx, "the sizes of src and dst do not match");
        return ret;
    }

    MI_S32 height   = src.GetSizes().m_height;
    MI_S32 uv_const = 128 * (1 << CVTCOLOR_COEF_BITS);

    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return ret;
    }

    switch (type)
    {
        case CvtColorType::RGB2YUV_Y444:
        {
            ret = wp->ParallelFor((MI_S32)0, height, CvtRgb2Y444NeonImpl<0>, std::cref(src), std::ref(dst_y),
                                  std::ref(dst_u), std::ref(dst_v), uv_const);
            break;
        }

        case CvtColorType::RGB2YUV_Y444_601:
        {
            ret = wp->ParallelFor((MI_S32)0, height, CvtRgb2Y444NeonImpl<1>, std::cref(src), std::ref(dst_y),
                                  std::ref(dst_u), std::ref(dst_v), uv_const);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "cvtcolor type error");
            ret = Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

template <MI_U32 MODE>
AURA_ALWAYS_INLINE AURA_VOID CvtRgb2YP010(uint16x8x3_t &v3qu16_rgb, MI_U16 *dy)
{
    constexpr MI_S32 SHIFT_BACK = CVTCOLOR_COEF_BITS - 6;

    uint32x4x3_t v3qu32_rgb_lo;
    uint32x4x3_t v3qu32_rgb_hi;

    constexpr MI_S32 R2Y = Rgb2YuvParamTraits<MODE>::R2Y;
    constexpr MI_S32 G2Y = Rgb2YuvParamTraits<MODE>::G2Y;
    constexpr MI_S32 B2Y = Rgb2YuvParamTraits<MODE>::B2Y;

    // u16->u32: low 16x4
    v3qu32_rgb_lo.val[0] = neon::vmovl(neon::vgetlow(v3qu16_rgb.val[0]));
    v3qu32_rgb_lo.val[1] = neon::vmovl(neon::vgetlow(v3qu16_rgb.val[1]));
    v3qu32_rgb_lo.val[2] = neon::vmovl(neon::vgetlow(v3qu16_rgb.val[2]));

    // u16->u32: high 16x4
    v3qu32_rgb_hi.val[0] = neon::vmovl(neon::vgethigh(v3qu16_rgb.val[0]));
    v3qu32_rgb_hi.val[1] = neon::vmovl(neon::vgethigh(v3qu16_rgb.val[1]));
    v3qu32_rgb_hi.val[2] = neon::vmovl(neon::vgethigh(v3qu16_rgb.val[2]));

    // r2y * r
    uint32x4_t vqu32_lo = neon::vmul(v3qu32_rgb_lo.val[0], static_cast<MI_U32>(R2Y));
    uint32x4_t vqu32_hi = neon::vmul(v3qu32_rgb_hi.val[0], static_cast<MI_U32>(R2Y));

    vqu32_lo = neon::vmla(vqu32_lo, v3qu32_rgb_lo.val[1], static_cast<MI_U32>(G2Y));
    vqu32_hi = neon::vmla(vqu32_hi, v3qu32_rgb_hi.val[1], static_cast<MI_U32>(G2Y));

    vqu32_lo = neon::vmla(vqu32_lo, v3qu32_rgb_lo.val[2], static_cast<MI_U32>(B2Y));
    vqu32_hi = neon::vmla(vqu32_hi, v3qu32_rgb_hi.val[2], static_cast<MI_U32>(B2Y));

    uint16x4_t vdu16_lo = neon::vrshrn_n<SHIFT_BACK>(vqu32_lo);
    uint16x4_t vdu16_hi = neon::vrshrn_n<SHIFT_BACK>(vqu32_hi);

    neon::vstore(dy, neon::vcombine(vdu16_lo, vdu16_hi));
}

template <MI_U32 MODE>
AURA_ALWAYS_INLINE AURA_VOID CvtRgb2UvP010(uint16x4x3_t &v3qu16_rgb, int32x4_t &vqs32_uv_const, uint16x4_t &vdu16_u, uint16x4_t &vdu16_v)
{
    int32x4x3_t v3qs32_rgb;

    constexpr MI_S32 R2U = Rgb2YuvParamTraits<MODE>::R2U;
    constexpr MI_S32 G2U = Rgb2YuvParamTraits<MODE>::G2U;
    constexpr MI_S32 B2U = Rgb2YuvParamTraits<MODE>::B2U;
    constexpr MI_S32 G2V = Rgb2YuvParamTraits<MODE>::G2V;
    constexpr MI_S32 B2V = Rgb2YuvParamTraits<MODE>::B2V;

    int32x4_t vqs32_scale;
    neon::vdup(vqs32_scale, static_cast<MI_S32>(1 << ((CVTCOLOR_COEF_BITS - 6) - 1)));

    // u16->s32
    v3qs32_rgb.val[0] = neon::vreinterpret(neon::vmovl(v3qu16_rgb.val[0]));
    v3qs32_rgb.val[1] = neon::vreinterpret(neon::vmovl(v3qu16_rgb.val[1]));
    v3qs32_rgb.val[2] = neon::vreinterpret(neon::vmovl(v3qu16_rgb.val[2]));

    int32x4_t vqs32_u = neon::vmla(vqs32_uv_const, v3qs32_rgb.val[2], static_cast<MI_S32>(B2U));
    int32x4_t vqs32_v = neon::vmla(vqs32_uv_const, v3qs32_rgb.val[0], static_cast<MI_S32>(B2U));

    vqs32_u = neon::vmls(vqs32_u, v3qs32_rgb.val[0], static_cast<MI_S32>(-R2U));
    vqs32_v = neon::vmls(vqs32_v, v3qs32_rgb.val[1], static_cast<MI_S32>(-G2V));

    vqs32_u = neon::vmls(vqs32_u, v3qs32_rgb.val[1], static_cast<MI_S32>(-G2U));
    vqs32_v = neon::vmls(vqs32_v, v3qs32_rgb.val[2], static_cast<MI_S32>(-B2V));

    vdu16_u = neon::vshrn_n<14>(neon::vadd(vqs32_u, vqs32_scale));
    vdu16_v = neon::vshrn_n<14>(neon::vadd(vqs32_v, vqs32_scale));
}

template <MI_U32 MODE>
static Status CvtRgb2NvP010NeonImpl(const Mat &src, Mat &dst_y, Mat &dst_uv, MI_S32 uv_const, MI_BOOL swapuv, MI_S32 start_row, MI_S32 end_row)
{
    MI_S32 width       = src.GetSizes().m_width;
    MI_S32 ichannel    = src.GetSizes().m_channel;
    MI_S32 width_align = width & (-8);
    MI_S32 offset      = 0;

    const MI_S32 uidx = swapuv;
    const MI_S32 vidx = 1 - uidx;

    int32x4_t vqs32_uv_const;
    neon::vdup(vqs32_uv_const, uv_const);

    for (MI_S32 y = start_row; y < end_row; ++y)
    {
        MI_S32 y_idx = y << 1;
        // rgb
        const MI_U16 *src_c = src.Ptr<MI_U16>(y_idx);
        const MI_U16 *src_n = src.Ptr<MI_U16>(y_idx + 1);
        // y line ptr
        MI_U16 *dsty_c = dst_y.Ptr<MI_U16>(y_idx);
        MI_U16 *dsty_n = dst_y.Ptr<MI_U16>(y_idx + 1);
        // uv
        MI_U16 *dst_uv_c = dst_uv.Ptr<MI_U16>(y);

        MI_S32 x = 0;
        for (; x < width_align; x += 8)
        {
LOOP_BODY:
            offset = x * ichannel;

            // even line
            uint16x8x3_t v3qu16_src_c = neon::vload3q(src_c + offset);
            CvtRgb2YP010<MODE>(v3qu16_src_c, dsty_c + x);

            // select uv's input
            uint16x4x3_t v3du16_uv_in;
            uint16x4x2_t v2du16_in;

            // uv r
            v2du16_in           = neon::vuzp(neon::vgetlow(v3qu16_src_c.val[0]), neon::vgethigh(v3qu16_src_c.val[0]));
            v3du16_uv_in.val[0] = v2du16_in.val[0];

            // uv g
            v2du16_in           = neon::vuzp(neon::vgetlow(v3qu16_src_c.val[1]), neon::vgethigh(v3qu16_src_c.val[1]));
            v3du16_uv_in.val[1] = v2du16_in.val[0];

            // uv b
            v2du16_in           = neon::vuzp(neon::vgetlow(v3qu16_src_c.val[2]), neon::vgethigh(v3qu16_src_c.val[2]));
            v3du16_uv_in.val[2] = v2du16_in.val[0];

            uint16x4x2_t v2du16_dst_uv;
            CvtRgb2UvP010<MODE>(v3du16_uv_in, vqs32_uv_const, v2du16_dst_uv.val[uidx], v2du16_dst_uv.val[vidx]);
            neon::vstore(dst_uv_c + x, v2du16_dst_uv);

            // odd line
            uint16x8x3_t v3qu16_src_n = neon::vload3q(src_n + offset);
            CvtRgb2YP010<MODE>(v3qu16_src_n, dsty_n + x);
        }

        // the remaining part
        if (x < width)
        {
            x = width - 8;
            goto LOOP_BODY;
        }
    }

    return Status::OK;
}

Status CvtRgb2NvP010Neon(Context *ctx, const Mat &src, Mat &dst_y, Mat &dst_uv, MI_BOOL swapuv, const OpTarget &target)
{
    AURA_UNUSED(target);
    Status ret = Status::ERROR;

    if ((src.GetSizes().m_width & 1) || (src.GetSizes().m_height & 1))
    {
        AURA_ADD_ERROR_STRING(ctx, "src sizes only support even");
        return ret;
    }

    if (dst_y.GetSizes() * Sizes3(1, 1, 2) != dst_uv.GetSizes() * Sizes3(2, 2, 1) ||
        dst_y.GetSizes() * Sizes3(1, 1, 3) != src.GetSizes() ||
        dst_y.GetSizes().m_channel != 1)
    {
        AURA_ADD_ERROR_STRING(ctx, "the sizes of src and dst do not match");
        return ret;
    }

    MI_S32 height   = dst_uv.GetSizes().m_height;
    MI_S32 uv_const = 512 * (1 << CVTCOLOR_COEF_BITS);

    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return ret;
    }

    if ((ret = wp->ParallelFor(0, height, CvtRgb2NvP010NeonImpl<1>, std::cref(src), std::ref(dst_y), std::ref(dst_uv),
                               uv_const, swapuv)) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "ParallelFor run failed");
        return ret;
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura
