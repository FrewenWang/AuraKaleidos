#include "cvtcolor_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

template <DT_U32 MODE>
AURA_ALWAYS_INLINE DT_VOID CvtRgb2Nv(DT_U8 r00, DT_U8 g00, DT_U8 b00, DT_U8 r01, DT_U8 g01, DT_U8 b01,
                                     DT_U8 r10, DT_U8 g10, DT_U8 b10, DT_U8 r11, DT_U8 g11, DT_U8 b11,
                                     DT_S32 uv_const, DT_U8 &y00, DT_U8 &y01, DT_U8 &y10, DT_U8 &y11,
                                     DT_U8 &u, DT_U8 &v)
{
    constexpr DT_S32 R2Y = Rgb2YuvParamTraits<MODE>::R2Y;
    constexpr DT_S32 G2Y = Rgb2YuvParamTraits<MODE>::G2Y;
    constexpr DT_S32 B2Y = Rgb2YuvParamTraits<MODE>::B2Y;
    constexpr DT_S32 R2U = Rgb2YuvParamTraits<MODE>::R2U;
    constexpr DT_S32 G2U = Rgb2YuvParamTraits<MODE>::G2U;
    constexpr DT_S32 B2U = Rgb2YuvParamTraits<MODE>::B2U;
    constexpr DT_S32 G2V = Rgb2YuvParamTraits<MODE>::G2V;
    constexpr DT_S32 B2V = Rgb2YuvParamTraits<MODE>::B2V;
    constexpr DT_S32 YC  = Rgb2YuvParamTraits<MODE>::YC;

    DT_S32 y00t = YC + R2Y * r00 + G2Y * g00 + B2Y * b00;
    DT_S32 y01t = YC + R2Y * r01 + G2Y * g01 + B2Y * b01;
    DT_S32 y10t = YC + R2Y * r10 + G2Y * g10 + B2Y * b10;
    DT_S32 y11t = YC + R2Y * r11 + G2Y * g11 + B2Y * b11;

    DT_S32 ut = uv_const + R2U * r00 + G2U * g00 + B2U * b00;
    DT_S32 vt = uv_const + B2U * r00 + G2V * g00 + B2V * b00;

    y00 = SaturateCast<DT_U8>(CVTCOLOR_DESCALE(y00t, CVTCOLOR_COEF_BITS));
    y01 = SaturateCast<DT_U8>(CVTCOLOR_DESCALE(y01t, CVTCOLOR_COEF_BITS));
    y10 = SaturateCast<DT_U8>(CVTCOLOR_DESCALE(y10t, CVTCOLOR_COEF_BITS));
    y11 = SaturateCast<DT_U8>(CVTCOLOR_DESCALE(y11t, CVTCOLOR_COEF_BITS));

    u = SaturateCast<DT_U8>(CVTCOLOR_DESCALE(ut, CVTCOLOR_COEF_BITS));
    v = SaturateCast<DT_U8>(CVTCOLOR_DESCALE(vt, CVTCOLOR_COEF_BITS));
}

template <DT_U32 MODE>
AURA_ALWAYS_INLINE DT_VOID CvtRgb2Y444(DT_U8 r, DT_U8 g, DT_U8 b, DT_S32 uv_const, DT_U8 &y, DT_U8 &u, DT_U8 &v)
{
    constexpr DT_S32 R2Y = Rgb2YuvParamTraits<MODE>::R2Y;
    constexpr DT_S32 G2Y = Rgb2YuvParamTraits<MODE>::G2Y;
    constexpr DT_S32 B2Y = Rgb2YuvParamTraits<MODE>::B2Y;
    constexpr DT_S32 R2U = Rgb2YuvParamTraits<MODE>::R2U;
    constexpr DT_S32 G2U = Rgb2YuvParamTraits<MODE>::G2U;
    constexpr DT_S32 B2U = Rgb2YuvParamTraits<MODE>::B2U;
    constexpr DT_S32 G2V = Rgb2YuvParamTraits<MODE>::G2V;
    constexpr DT_S32 B2V = Rgb2YuvParamTraits<MODE>::B2V;
    constexpr DT_S32 YC  = Rgb2YuvParamTraits<MODE>::YC;

    DT_S32 yt = YC + R2Y * r + G2Y * g + B2Y * b;
    DT_S32 ut = uv_const + R2U * r + G2U * g + B2U * b;
    DT_S32 vt = uv_const + B2U * r + G2V * g + B2V * b;

    y = SaturateCast<DT_U8>(CVTCOLOR_DESCALE(yt, CVTCOLOR_COEF_BITS));
    u = SaturateCast<DT_U8>(CVTCOLOR_DESCALE(ut, CVTCOLOR_COEF_BITS));
    v = SaturateCast<DT_U8>(CVTCOLOR_DESCALE(vt, CVTCOLOR_COEF_BITS));
}

template <DT_U32 MODE>
static Status CvtRgb2NvNoneCore(const Mat &src, Mat &dst_y, Mat &dst_uv, DT_S32 uv_const, DT_BOOL swapuv, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 width    = src.GetSizes().m_width;
    DT_S32 ichannel = src.GetSizes().m_channel;

    const DT_S32 uidx = swapuv;
    const DT_S32 vidx = 1 - uidx;

    for (DT_S32 uv = start_row; uv < end_row; uv++)
    {
        DT_S32 y = uv * 2;

        DT_U8 *dst_y_c  = dst_y.Ptr<DT_U8>(y);
        DT_U8 *dst_y_n  = dst_y.Ptr<DT_U8>(y + 1);
        DT_U8 *dst_uv_c = dst_uv.Ptr<DT_U8>(uv);

        const DT_U8 *src_c = src.Ptr<DT_U8>(y);
        const DT_U8 *src_n = src.Ptr<DT_U8>(y + 1);

        for (DT_S32 x = 0; x < width; x += 2)
        {
            DT_S32 offset = x * ichannel;
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

static Status CvtRgb2NvNoneImpl(const Mat &src, Mat &dst_y, Mat &dst_uv, DT_S32 uv_const, DT_BOOL swapuv,
                                CvtColorType type, DT_S32 start_row, DT_S32 end_row)
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

Status CvtRgb2NvNone(Context *ctx, const Mat &src, Mat &dst_y, Mat &dst_uv, DT_BOOL swapuv, CvtColorType type, const OpTarget &target)
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

    DT_S32 height   = dst_uv.GetSizes().m_height;
    DT_S32 uv_const = 128 * (1 << CVTCOLOR_COEF_BITS);

    if (target.m_data.none.enable_mt)
    {
        WorkerPool *wp = ctx->GetWorkerPool();
        if (DT_NULL == wp)
        {
            AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
            return Status::ERROR;
        }

        ret = wp->ParallelFor((DT_S32)0, height, CvtRgb2NvNoneImpl, std::cref(src), std::ref(dst_y), std::ref(dst_uv), uv_const, swapuv, type);
    }
    else
    {
        ret = CvtRgb2NvNoneImpl(src, dst_y, dst_uv, uv_const, swapuv, type, 0, height);
    }

    AURA_RETURN(ctx, ret);
}

template <DT_U32 MODE>
static Status CvtRgb2Y420NoneCore(const Mat &src, Mat &dst_y, Mat &dst_u, Mat &dst_v, DT_S32 uv_const, DT_BOOL swapuv, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 width    = src.GetSizes().m_width;
    DT_S32 ichannel = src.GetSizes().m_channel;

    for (DT_S32 uv = start_row; uv < end_row; uv++)
    {
        DT_S32 y = uv * 2;

        DT_U8 *dst_y_c = dst_y.Ptr<DT_U8>(y);
        DT_U8 *dst_y_n = dst_y.Ptr<DT_U8>(y + 1);
        DT_U8 *dst_u_c = swapuv ? dst_v.Ptr<DT_U8>(uv) : dst_u.Ptr<DT_U8>(uv);
        DT_U8 *dst_v_c = swapuv ? dst_u.Ptr<DT_U8>(uv) : dst_v.Ptr<DT_U8>(uv);

        const DT_U8 *src_c = src.Ptr<DT_U8>(y);
        const DT_U8 *src_n = src.Ptr<DT_U8>(y + 1);

        for (DT_S32 x = 0; x < width; x += 2)
        {
            DT_S32 offset = x * ichannel;
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

static Status CvtRgb2Y420NoneImpl(const Mat &src, Mat &dst_y, Mat &dst_u, Mat &dst_v, DT_S32 uv_const, DT_BOOL swapuv,
                                  CvtColorType type, DT_S32 start_row, DT_S32 end_row)
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

Status CvtRgb2Y420None(Context *ctx, const Mat &src, Mat &dst_y, Mat &dst_u, Mat &dst_v, DT_BOOL swapuv, CvtColorType type, const OpTarget &target)
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

    DT_S32 height   = dst_u.GetSizes().m_height;
    DT_S32 uv_const = 128 * (1 << CVTCOLOR_COEF_BITS);

    if (target.m_data.none.enable_mt)
    {
        WorkerPool *wp = ctx->GetWorkerPool();
        if (DT_NULL == wp)
        {
            AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
            return Status::ERROR;
        }

        ret = wp->ParallelFor((DT_S32)0, height, CvtRgb2Y420NoneImpl, std::cref(src), std::ref(dst_y), std::ref(dst_u),
                              std::ref(dst_v), uv_const, swapuv, type);
    }
    else
    {
        ret = CvtRgb2Y420NoneImpl(src, dst_y, dst_u, dst_v, uv_const, swapuv, type, 0, height);
    }

    AURA_RETURN(ctx, ret);
}

template <DT_U32 MODE>
static Status CvtRgb2Y444NoneCore(const Mat &src, Mat &dst_y, Mat &dst_u, Mat &dst_v, DT_S32 uv_const, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 width    = src.GetSizes().m_width;
    DT_S32 ichannel = src.GetSizes().m_channel;

    for (DT_S32 y = start_row; y < end_row; ++y)
    {
        DT_U8 *dst_y_c = dst_y.Ptr<DT_U8>(y);
        DT_U8 *dst_u_c = dst_u.Ptr<DT_U8>(y);
        DT_U8 *dst_v_c = dst_v.Ptr<DT_U8>(y);

        const DT_U8 *src_c = src.Ptr<DT_U8>(y);

        for (DT_S32 x = 0; x < width; ++x)
        {
            DT_S32 offset = x * ichannel;
            CvtRgb2Y444<MODE>(src_c[offset + 0], src_c[offset + 1], src_c[offset + 2], uv_const, dst_y_c[x], dst_u_c[x], dst_v_c[x]);
        }
    }

    return Status::OK;
}

static Status CvtRgb2Y444NoneImpl(const Mat &src, Mat &dst_y, Mat &dst_u, Mat &dst_v, DT_S32 uv_const, CvtColorType type, DT_S32 start_row, DT_S32 end_row)
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

    DT_S32 height   = src.GetSizes().m_height;
    DT_S32 uv_const = 128 * (1 << CVTCOLOR_COEF_BITS);

    if (target.m_data.none.enable_mt)
    {
        WorkerPool *wp = ctx->GetWorkerPool();
        if (DT_NULL == wp)
        {
            AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
            return Status::ERROR;
        }

        ret = wp->ParallelFor((DT_S32)0, height, CvtRgb2Y444NoneImpl, std::cref(src), std::ref(dst_y), std::ref(dst_u),
                              std::ref(dst_v), uv_const, type);
    }
    else
    {
        ret = CvtRgb2Y444NoneImpl(src, dst_y, dst_u, dst_v, uv_const, type, 0, height);
    }

    AURA_RETURN(ctx, ret);
}

template <DT_U32 MODE>
AURA_ALWAYS_INLINE DT_VOID CvtRgb2NvP010(DT_U16 r00, DT_U16 g00, DT_U16 b00, DT_U16 r01, DT_U16 g01, DT_U16 b01,
                                         DT_U16 r10, DT_U16 g10, DT_U16 b10, DT_U16 r11, DT_U16 g11, DT_U16 b11,
                                         DT_S32 uv_const, DT_U16 &y00, DT_U16 &y01, DT_U16 &y10, DT_U16 &y11,
                                         DT_U16 &u, DT_U16 &v)
{
    constexpr DT_S32 R2Y = Rgb2YuvParamTraits<MODE>::R2Y;
    constexpr DT_S32 G2Y = Rgb2YuvParamTraits<MODE>::G2Y;
    constexpr DT_S32 B2Y = Rgb2YuvParamTraits<MODE>::B2Y;
    constexpr DT_S32 R2U = Rgb2YuvParamTraits<MODE>::R2U;
    constexpr DT_S32 G2U = Rgb2YuvParamTraits<MODE>::G2U;
    constexpr DT_S32 B2U = Rgb2YuvParamTraits<MODE>::B2U;
    constexpr DT_S32 G2V = Rgb2YuvParamTraits<MODE>::G2V;
    constexpr DT_S32 B2V = Rgb2YuvParamTraits<MODE>::B2V;

    DT_S32 y00t = R2Y * r00 + G2Y * g00 + B2Y * b00;
    DT_S32 y01t = R2Y * r01 + G2Y * g01 + B2Y * b01;
    DT_S32 y10t = R2Y * r10 + G2Y * g10 + B2Y * b10;
    DT_S32 y11t = R2Y * r11 + G2Y * g11 + B2Y * b11;

    DT_S32 ut = uv_const + R2U * r00 + G2U * g00 + B2U * b00;
    DT_S32 vt = uv_const + B2U * r00 + G2V * g00 + B2V * b00;

    y00 = SaturateCast<DT_U16>(CVTCOLOR_DESCALE(y00t, (CVTCOLOR_COEF_BITS - 6)));
    y01 = SaturateCast<DT_U16>(CVTCOLOR_DESCALE(y01t, (CVTCOLOR_COEF_BITS - 6)));
    y10 = SaturateCast<DT_U16>(CVTCOLOR_DESCALE(y10t, (CVTCOLOR_COEF_BITS - 6)));
    y11 = SaturateCast<DT_U16>(CVTCOLOR_DESCALE(y11t, (CVTCOLOR_COEF_BITS - 6)));

    u = SaturateCast<DT_U16>(CVTCOLOR_DESCALE(ut, (CVTCOLOR_COEF_BITS - 6)));
    v = SaturateCast<DT_U16>(CVTCOLOR_DESCALE(vt, (CVTCOLOR_COEF_BITS - 6)));
}

template <DT_U32 MODE>
static Status CvtRgb2NvP010NoneImpl(const Mat &src, Mat &dst_y, Mat &dst_uv, DT_S32 uv_const, DT_BOOL swapuv, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 width    = src.GetSizes().m_width;
    DT_S32 ichannel = src.GetSizes().m_channel;

    const DT_S32 uidx = swapuv;
    const DT_S32 vidx = 1 - uidx;

    for (DT_S32 uv = start_row; uv < end_row; uv++)
    {
        DT_S32 y = uv * 2;

        const DT_U16 *src_c = src.Ptr<DT_U16>(y);
        const DT_U16 *src_n = src.Ptr<DT_U16>(y + 1);

        DT_U16 *dst_y_c  = dst_y.Ptr<DT_U16>(y);
        DT_U16 *dst_y_n  = dst_y.Ptr<DT_U16>(y + 1);
        DT_U16 *dst_uv_c = dst_uv.Ptr<DT_U16>(uv);

        for (DT_S32 x = 0; x < width; x += 2)
        {
            DT_S32 offset = x * ichannel;
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

Status CvtRgb2NvP010None(Context *ctx, const Mat &src, Mat &dst_y, Mat &dst_uv, DT_BOOL swapuv, const OpTarget &target)
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

    DT_S32 height   = dst_uv.GetSizes().m_height;
    DT_S32 uv_const = 512 * (1 << CVTCOLOR_COEF_BITS);

    if (target.m_data.none.enable_mt)
    {
        WorkerPool *wp = ctx->GetWorkerPool();
        if (DT_NULL == wp)
        {
            AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
            return Status::ERROR;
        }

        ret = wp->ParallelFor((DT_S32)0, height, CvtRgb2NvP010NoneImpl<1>, std::cref(src), std::ref(dst_y), std::ref(dst_uv), uv_const, swapuv);
    }
    else
    {
        ret = CvtRgb2NvP010NoneImpl<1>(src, dst_y, dst_uv, uv_const, swapuv, 0, height);
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura