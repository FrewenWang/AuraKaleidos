#include "cvtcolor_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

template <MI_U32 MODE>
AURA_ALWAYS_INLINE AURA_VOID CvtRgb2Nv(MI_U8 r00, MI_U8 g00, MI_U8 b00, MI_U8 r01, MI_U8 g01, MI_U8 b01,
                                     MI_U8 r10, MI_U8 g10, MI_U8 b10, MI_U8 r11, MI_U8 g11, MI_U8 b11,
                                     MI_S32 uv_const, MI_U8 &y00, MI_U8 &y01, MI_U8 &y10, MI_U8 &y11,
                                     MI_U8 &u, MI_U8 &v)
{
    constexpr MI_S32 R2Y = Rgb2YuvParamTraits<MODE>::R2Y;
    constexpr MI_S32 G2Y = Rgb2YuvParamTraits<MODE>::G2Y;
    constexpr MI_S32 B2Y = Rgb2YuvParamTraits<MODE>::B2Y;
    constexpr MI_S32 R2U = Rgb2YuvParamTraits<MODE>::R2U;
    constexpr MI_S32 G2U = Rgb2YuvParamTraits<MODE>::G2U;
    constexpr MI_S32 B2U = Rgb2YuvParamTraits<MODE>::B2U;
    constexpr MI_S32 G2V = Rgb2YuvParamTraits<MODE>::G2V;
    constexpr MI_S32 B2V = Rgb2YuvParamTraits<MODE>::B2V;
    constexpr MI_S32 YC  = Rgb2YuvParamTraits<MODE>::YC;

    MI_S32 y00t = YC + R2Y * r00 + G2Y * g00 + B2Y * b00;
    MI_S32 y01t = YC + R2Y * r01 + G2Y * g01 + B2Y * b01;
    MI_S32 y10t = YC + R2Y * r10 + G2Y * g10 + B2Y * b10;
    MI_S32 y11t = YC + R2Y * r11 + G2Y * g11 + B2Y * b11;

    MI_S32 ut = uv_const + R2U * r00 + G2U * g00 + B2U * b00;
    MI_S32 vt = uv_const + B2U * r00 + G2V * g00 + B2V * b00;

    y00 = SaturateCast<MI_U8>(CVTCOLOR_DESCALE(y00t, CVTCOLOR_COEF_BITS));
    y01 = SaturateCast<MI_U8>(CVTCOLOR_DESCALE(y01t, CVTCOLOR_COEF_BITS));
    y10 = SaturateCast<MI_U8>(CVTCOLOR_DESCALE(y10t, CVTCOLOR_COEF_BITS));
    y11 = SaturateCast<MI_U8>(CVTCOLOR_DESCALE(y11t, CVTCOLOR_COEF_BITS));

    u = SaturateCast<MI_U8>(CVTCOLOR_DESCALE(ut, CVTCOLOR_COEF_BITS));
    v = SaturateCast<MI_U8>(CVTCOLOR_DESCALE(vt, CVTCOLOR_COEF_BITS));
}

template <MI_U32 MODE>
AURA_ALWAYS_INLINE AURA_VOID CvtRgb2Y444(MI_U8 r, MI_U8 g, MI_U8 b, MI_S32 uv_const, MI_U8 &y, MI_U8 &u, MI_U8 &v)
{
    constexpr MI_S32 R2Y = Rgb2YuvParamTraits<MODE>::R2Y;
    constexpr MI_S32 G2Y = Rgb2YuvParamTraits<MODE>::G2Y;
    constexpr MI_S32 B2Y = Rgb2YuvParamTraits<MODE>::B2Y;
    constexpr MI_S32 R2U = Rgb2YuvParamTraits<MODE>::R2U;
    constexpr MI_S32 G2U = Rgb2YuvParamTraits<MODE>::G2U;
    constexpr MI_S32 B2U = Rgb2YuvParamTraits<MODE>::B2U;
    constexpr MI_S32 G2V = Rgb2YuvParamTraits<MODE>::G2V;
    constexpr MI_S32 B2V = Rgb2YuvParamTraits<MODE>::B2V;
    constexpr MI_S32 YC  = Rgb2YuvParamTraits<MODE>::YC;

    MI_S32 yt = YC + R2Y * r + G2Y * g + B2Y * b;
    MI_S32 ut = uv_const + R2U * r + G2U * g + B2U * b;
    MI_S32 vt = uv_const + B2U * r + G2V * g + B2V * b;

    y = SaturateCast<MI_U8>(CVTCOLOR_DESCALE(yt, CVTCOLOR_COEF_BITS));
    u = SaturateCast<MI_U8>(CVTCOLOR_DESCALE(ut, CVTCOLOR_COEF_BITS));
    v = SaturateCast<MI_U8>(CVTCOLOR_DESCALE(vt, CVTCOLOR_COEF_BITS));
}

template <MI_U32 MODE>
static Status CvtRgb2NvNoneCore(const Mat &src, Mat &dst_y, Mat &dst_uv, MI_S32 uv_const, MI_BOOL swapuv, MI_S32 start_row, MI_S32 end_row)
{
    MI_S32 width    = src.GetSizes().m_width;
    MI_S32 ichannel = src.GetSizes().m_channel;

    const MI_S32 uidx = swapuv;
    const MI_S32 vidx = 1 - uidx;

    for (MI_S32 uv = start_row; uv < end_row; uv++)
    {
        MI_S32 y = uv * 2;

        MI_U8 *dst_y_c  = dst_y.Ptr<MI_U8>(y);
        MI_U8 *dst_y_n  = dst_y.Ptr<MI_U8>(y + 1);
        MI_U8 *dst_uv_c = dst_uv.Ptr<MI_U8>(uv);

        const MI_U8 *src_c = src.Ptr<MI_U8>(y);
        const MI_U8 *src_n = src.Ptr<MI_U8>(y + 1);

        for (MI_S32 x = 0; x < width; x += 2)
        {
            MI_S32 offset = x * ichannel;
            CvtRgb2Nv<MODE>(src_c[offset + 0], src_c[offset + 1], src_c[offset + 2],
                            src_c[offset + 3], src_c[offset + 4], src_c[offset + 5],
                            src_n[offset + 0], src_n[offset + 1], src_n[offset + 2],
                            src_n[offset + 3], src_n[offset + 4], src_n[offset + 5],
                            uv_const, dst_y_c[x + 0], dst_y_c[x + 1],
                            dst_y_n[x + 0], dst_y_n[x + 1],
                            dst_uv_c[x + uidx], dst_uv_c[x + vidx]);
        }
    }

    return Status::OK;
}

static Status CvtRgb2NvNoneImpl(const Mat &src, Mat &dst_y, Mat &dst_uv, MI_S32 uv_const, MI_BOOL swapuv,
                                CvtColorType type, MI_S32 start_row, MI_S32 end_row)
{
    Status ret = Status::ERROR;

    switch (type)
    {
        case CvtColorType::RGB2YUV_NV12:
        case CvtColorType::RGB2YUV_NV21:
        {
            ret = CvtRgb2NvNoneCore<0>(src, dst_y, dst_uv, uv_const, swapuv, start_row, end_row);
            break;
        }

        case CvtColorType::RGB2YUV_NV12_601:
        case CvtColorType::RGB2YUV_NV21_601:
        {
            ret = CvtRgb2NvNoneCore<1>(src, dst_y, dst_uv, uv_const, swapuv, start_row, end_row);
            break;
        }

        default:
        {
            ret = Status::ERROR;
        }
    }

    return ret;
}

Status CvtRgb2NvNone(Context *ctx, const Mat &src, Mat &dst_y, Mat &dst_uv, MI_BOOL swapuv, CvtColorType type, const OpTarget &target)
{
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

    if (target.m_data.none.enable_mt)
    {
        WorkerPool *wp = ctx->GetWorkerPool();
        if (MI_NULL == wp)
        {
            AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
            return Status::ERROR;
        }

        ret = wp->ParallelFor((MI_S32)0, height, CvtRgb2NvNoneImpl, std::cref(src), std::ref(dst_y), std::ref(dst_uv), uv_const, swapuv, type);
    }
    else
    {
        ret = CvtRgb2NvNoneImpl(src, dst_y, dst_uv, uv_const, swapuv, type, 0, height);
    }

    AURA_RETURN(ctx, ret);
}

template <MI_U32 MODE>
static Status CvtRgb2Y420NoneCore(const Mat &src, Mat &dst_y, Mat &dst_u, Mat &dst_v, MI_S32 uv_const, MI_BOOL swapuv, MI_S32 start_row, MI_S32 end_row)
{
    MI_S32 width    = src.GetSizes().m_width;
    MI_S32 ichannel = src.GetSizes().m_channel;

    for (MI_S32 uv = start_row; uv < end_row; uv++)
    {
        MI_S32 y = uv * 2;

        MI_U8 *dst_y_c = dst_y.Ptr<MI_U8>(y);
        MI_U8 *dst_y_n = dst_y.Ptr<MI_U8>(y + 1);
        MI_U8 *dst_u_c = swapuv ? dst_v.Ptr<MI_U8>(uv) : dst_u.Ptr<MI_U8>(uv);
        MI_U8 *dst_v_c = swapuv ? dst_u.Ptr<MI_U8>(uv) : dst_v.Ptr<MI_U8>(uv);

        const MI_U8 *src_c = src.Ptr<MI_U8>(y);
        const MI_U8 *src_n = src.Ptr<MI_U8>(y + 1);

        for (MI_S32 x = 0; x < width; x += 2)
        {
            MI_S32 offset = x * ichannel;
            CvtRgb2Nv<MODE>(src_c[offset + 0], src_c[offset + 1], src_c[offset + 2],
                            src_c[offset + 3], src_c[offset + 4], src_c[offset + 5],
                            src_n[offset + 0], src_n[offset + 1], src_n[offset + 2],
                            src_n[offset + 3], src_n[offset + 4], src_n[offset + 5],
                            uv_const, dst_y_c[x + 0], dst_y_c[x + 1],
                            dst_y_n[x + 0], dst_y_n[x + 1],
                            dst_u_c[x >> 1], dst_v_c[x >> 1]);
        }
    }

    return Status::OK;
}

static Status CvtRgb2Y420NoneImpl(const Mat &src, Mat &dst_y, Mat &dst_u, Mat &dst_v, MI_S32 uv_const, MI_BOOL swapuv,
                                  CvtColorType type, MI_S32 start_row, MI_S32 end_row)
{
    Status ret = Status::ERROR;

    switch (type)
    {
        case CvtColorType::RGB2YUV_YU12:
        case CvtColorType::RGB2YUV_YV12:
        {
            ret = CvtRgb2Y420NoneCore<0>(src, dst_y, dst_u, dst_v, uv_const, swapuv, start_row, end_row);
            break;
        }

        case CvtColorType::RGB2YUV_YU12_601:
        case CvtColorType::RGB2YUV_YV12_601:
        {
            ret = CvtRgb2Y420NoneCore<1>(src, dst_y, dst_u, dst_v, uv_const, swapuv, start_row, end_row);
            break;
        }

        default:
        {
            ret = Status::ERROR;
        }
    }

    return ret;
}

Status CvtRgb2Y420None(Context *ctx, const Mat &src, Mat &dst_y, Mat &dst_u, Mat &dst_v, MI_BOOL swapuv, CvtColorType type, const OpTarget &target)
{
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

    if (target.m_data.none.enable_mt)
    {
        WorkerPool *wp = ctx->GetWorkerPool();
        if (MI_NULL == wp)
        {
            AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
            return Status::ERROR;
        }

        ret = wp->ParallelFor((MI_S32)0, height, CvtRgb2Y420NoneImpl, std::cref(src), std::ref(dst_y), std::ref(dst_u),
                              std::ref(dst_v), uv_const, swapuv, type);
    }
    else
    {
        ret = CvtRgb2Y420NoneImpl(src, dst_y, dst_u, dst_v, uv_const, swapuv, type, 0, height);
    }

    AURA_RETURN(ctx, ret);
}

template <MI_U32 MODE>
static Status CvtRgb2Y444NoneCore(const Mat &src, Mat &dst_y, Mat &dst_u, Mat &dst_v, MI_S32 uv_const, MI_S32 start_row, MI_S32 end_row)
{
    MI_S32 width    = src.GetSizes().m_width;
    MI_S32 ichannel = src.GetSizes().m_channel;

    for (MI_S32 y = start_row; y < end_row; ++y)
    {
        MI_U8 *dst_y_c = dst_y.Ptr<MI_U8>(y);
        MI_U8 *dst_u_c = dst_u.Ptr<MI_U8>(y);
        MI_U8 *dst_v_c = dst_v.Ptr<MI_U8>(y);

        const MI_U8 *src_c = src.Ptr<MI_U8>(y);

        for (MI_S32 x = 0; x < width; ++x)
        {
            MI_S32 offset = x * ichannel;
            CvtRgb2Y444<MODE>(src_c[offset + 0], src_c[offset + 1], src_c[offset + 2], uv_const, dst_y_c[x], dst_u_c[x], dst_v_c[x]);
        }
    }

    return Status::OK;
}

static Status CvtRgb2Y444NoneImpl(const Mat &src, Mat &dst_y, Mat &dst_u, Mat &dst_v, MI_S32 uv_const, CvtColorType type, MI_S32 start_row, MI_S32 end_row)
{
    Status ret = Status::ERROR;

    switch (type)
    {
        case CvtColorType::RGB2YUV_Y444:
        {
            ret = CvtRgb2Y444NoneCore<0>(src, dst_y, dst_u, dst_v, uv_const, start_row, end_row);
            break;
        }

        case CvtColorType::RGB2YUV_Y444_601:
        {
            ret = CvtRgb2Y444NoneCore<1>(src, dst_y, dst_u, dst_v, uv_const, start_row, end_row);
            break;
        }

        default:
        {
            ret = Status::ERROR;
        }
    }

    return ret;
}

Status CvtRgb2Y444None(Context *ctx, const Mat &src, Mat &dst_y, Mat &dst_u, Mat &dst_v, CvtColorType type, const OpTarget &target)
{
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

    if (target.m_data.none.enable_mt)
    {
        WorkerPool *wp = ctx->GetWorkerPool();
        if (MI_NULL == wp)
        {
            AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
            return Status::ERROR;
        }

        ret = wp->ParallelFor((MI_S32)0, height, CvtRgb2Y444NoneImpl, std::cref(src), std::ref(dst_y), std::ref(dst_u),
                              std::ref(dst_v), uv_const, type);
    }
    else
    {
        ret = CvtRgb2Y444NoneImpl(src, dst_y, dst_u, dst_v, uv_const, type, 0, height);
    }

    AURA_RETURN(ctx, ret);
}

template <MI_U32 MODE>
AURA_ALWAYS_INLINE AURA_VOID CvtRgb2NvP010(MI_U16 r00, MI_U16 g00, MI_U16 b00, MI_U16 r01, MI_U16 g01, MI_U16 b01,
                                         MI_U16 r10, MI_U16 g10, MI_U16 b10, MI_U16 r11, MI_U16 g11, MI_U16 b11,
                                         MI_S32 uv_const, MI_U16 &y00, MI_U16 &y01, MI_U16 &y10, MI_U16 &y11,
                                         MI_U16 &u, MI_U16 &v)
{
    constexpr MI_S32 R2Y = Rgb2YuvParamTraits<MODE>::R2Y;
    constexpr MI_S32 G2Y = Rgb2YuvParamTraits<MODE>::G2Y;
    constexpr MI_S32 B2Y = Rgb2YuvParamTraits<MODE>::B2Y;
    constexpr MI_S32 R2U = Rgb2YuvParamTraits<MODE>::R2U;
    constexpr MI_S32 G2U = Rgb2YuvParamTraits<MODE>::G2U;
    constexpr MI_S32 B2U = Rgb2YuvParamTraits<MODE>::B2U;
    constexpr MI_S32 G2V = Rgb2YuvParamTraits<MODE>::G2V;
    constexpr MI_S32 B2V = Rgb2YuvParamTraits<MODE>::B2V;

    MI_S32 y00t = R2Y * r00 + G2Y * g00 + B2Y * b00;
    MI_S32 y01t = R2Y * r01 + G2Y * g01 + B2Y * b01;
    MI_S32 y10t = R2Y * r10 + G2Y * g10 + B2Y * b10;
    MI_S32 y11t = R2Y * r11 + G2Y * g11 + B2Y * b11;

    MI_S32 ut = uv_const + R2U * r00 + G2U * g00 + B2U * b00;
    MI_S32 vt = uv_const + B2U * r00 + G2V * g00 + B2V * b00;

    y00 = SaturateCast<MI_U16>(CVTCOLOR_DESCALE(y00t, (CVTCOLOR_COEF_BITS - 6)));
    y01 = SaturateCast<MI_U16>(CVTCOLOR_DESCALE(y01t, (CVTCOLOR_COEF_BITS - 6)));
    y10 = SaturateCast<MI_U16>(CVTCOLOR_DESCALE(y10t, (CVTCOLOR_COEF_BITS - 6)));
    y11 = SaturateCast<MI_U16>(CVTCOLOR_DESCALE(y11t, (CVTCOLOR_COEF_BITS - 6)));

    u = SaturateCast<MI_U16>(CVTCOLOR_DESCALE(ut, (CVTCOLOR_COEF_BITS - 6)));
    v = SaturateCast<MI_U16>(CVTCOLOR_DESCALE(vt, (CVTCOLOR_COEF_BITS - 6)));
}

template <MI_U32 MODE>
static Status CvtRgb2NvP010NoneImpl(const Mat &src, Mat &dst_y, Mat &dst_uv, MI_S32 uv_const, MI_BOOL swapuv, MI_S32 start_row, MI_S32 end_row)
{
    MI_S32 width    = src.GetSizes().m_width;
    MI_S32 ichannel = src.GetSizes().m_channel;

    const MI_S32 uidx = swapuv;
    const MI_S32 vidx = 1 - uidx;

    for (MI_S32 uv = start_row; uv < end_row; uv++)
    {
        MI_S32 y = uv * 2;

        const MI_U16 *src_c = src.Ptr<MI_U16>(y);
        const MI_U16 *src_n = src.Ptr<MI_U16>(y + 1);

        MI_U16 *dst_y_c  = dst_y.Ptr<MI_U16>(y);
        MI_U16 *dst_y_n  = dst_y.Ptr<MI_U16>(y + 1);
        MI_U16 *dst_uv_c = dst_uv.Ptr<MI_U16>(uv);

        for (MI_S32 x = 0; x < width; x += 2)
        {
            MI_S32 offset = x * ichannel;
            CvtRgb2NvP010<MODE>(src_c[offset + 0], src_c[offset + 1], src_c[offset + 2],
                                src_c[offset + 3], src_c[offset + 4], src_c[offset + 5],
                                src_n[offset + 0], src_n[offset + 1], src_n[offset + 2],
                                src_n[offset + 3], src_n[offset + 4], src_n[offset + 5],
                                uv_const, dst_y_c[x + 0], dst_y_c[x + 1],
                                dst_y_n[x + 0], dst_y_n[x + 1],
                                dst_uv_c[x + uidx], dst_uv_c[x + vidx]);
        }
    }

    return Status::OK;
}

Status CvtRgb2NvP010None(Context *ctx, const Mat &src, Mat &dst_y, Mat &dst_uv, MI_BOOL swapuv, const OpTarget &target)
{
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

    if (target.m_data.none.enable_mt)
    {
        WorkerPool *wp = ctx->GetWorkerPool();
        if (MI_NULL == wp)
        {
            AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
            return Status::ERROR;
        }

        ret = wp->ParallelFor((MI_S32)0, height, CvtRgb2NvP010NoneImpl<1>, std::cref(src), std::ref(dst_y), std::ref(dst_uv), uv_const, swapuv);
    }
    else
    {
        ret = CvtRgb2NvP010NoneImpl<1>(src, dst_y, dst_uv, uv_const, swapuv, 0, height);
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura