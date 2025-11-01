#include "cvtcolor_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

template <MI_U32 MODE>
AURA_ALWAYS_INLINE AURA_VOID Uv2Rgbuv(MI_U8 u, MI_U8 v, MI_S32 &ruv, MI_S32 &guv, MI_S32 &buv)
{
    MI_S32 uu = (MI_S32)u - 128;
    MI_S32 vv = (MI_S32)v - 128;

    constexpr MI_S32 V2R = Yuv2RgbParamTraits<MODE>::V2R;
    constexpr MI_S32 V2G = Yuv2RgbParamTraits<MODE>::V2G;
    constexpr MI_S32 U2G = Yuv2RgbParamTraits<MODE>::U2G;
    constexpr MI_S32 U2B = Yuv2RgbParamTraits<MODE>::U2B;

    ruv = (1 << (CVTCOLOR_COEF_BITS - 1)) + V2R * vv;
    guv = (1 << (CVTCOLOR_COEF_BITS - 1)) + V2G * vv + U2G * uu;
    buv = (1 << (CVTCOLOR_COEF_BITS - 1)) + U2B * uu;
}

template <MI_U32 MODE>
AURA_ALWAYS_INLINE AURA_VOID Yrgbuv2Rgb(MI_U8 y, MI_S32 ruv, MI_S32 guv, MI_S32 buv, MI_U8 &r, MI_U8 &g, MI_U8 &b)
{
    constexpr MI_S32 Y2RGB = Yuv2RgbParamTraits<MODE>::Y2RGB;
    MI_S32           yy    = MODE * (y * Y2RGB) + (1 - MODE) * (Max((MI_S32)0, (MI_S32)y - 16) * Y2RGB);

    r = SaturateCast<MI_U8>((yy + ruv) >> CVTCOLOR_COEF_BITS);
    g = SaturateCast<MI_U8>((yy + guv) >> CVTCOLOR_COEF_BITS);
    b = SaturateCast<MI_U8>((yy + buv) >> CVTCOLOR_COEF_BITS);
}

template <MI_U32 MODE>
static Status CvtNv2RgbNoneCore(const Mat &src_y, const Mat &src_uv, Mat &dst, MI_BOOL swapuv, MI_S32 start_row, MI_S32 end_row)
{
    MI_S32 width = dst.GetSizes().m_width;

    const MI_S32 uidx = swapuv;
    const MI_S32 vidx = 1 - uidx;

    for (MI_S32 uv = start_row; uv < end_row; uv++)
    {
        MI_S32 y = uv * 2;

        const MI_U8 *src_y_c  = src_y.Ptr<MI_U8>(y);
        const MI_U8 *src_y_n  = src_y.Ptr<MI_U8>(y + 1);
        const MI_U8 *src_uv_c = src_uv.Ptr<MI_U8>(uv);

        MI_U8 *dst_c = dst.Ptr<MI_U8>(y);
        MI_U8 *dst_n = dst.Ptr<MI_U8>(y + 1);

        for (MI_S32 x = 0; x < width; x += 2)
        {
            MI_U8  y00 = src_y_c[x], y01 = src_y_c[x + 1];
            MI_U8  y10 = src_y_n[x], y11 = src_y_n[x + 1];
            MI_U8  u00 = src_uv_c[x + uidx], v00 = src_uv_c[x + vidx];

            MI_S32 ruv, guv, buv;
            Uv2Rgbuv<MODE>(u00, v00, ruv, guv, buv);
            Yrgbuv2Rgb<MODE>(y00, ruv, guv, buv, dst_c[3 * x + 0], dst_c[3 * x + 1], dst_c[3 * x + 2]);
            Yrgbuv2Rgb<MODE>(y01, ruv, guv, buv, dst_c[3 * x + 3], dst_c[3 * x + 4], dst_c[3 * x + 5]);
            Yrgbuv2Rgb<MODE>(y10, ruv, guv, buv, dst_n[3 * x + 0], dst_n[3 * x + 1], dst_n[3 * x + 2]);
            Yrgbuv2Rgb<MODE>(y11, ruv, guv, buv, dst_n[3 * x + 3], dst_n[3 * x + 4], dst_n[3 * x + 5]);
        }
    }

    return Status::OK;
}

static Status CvtNv2RgbNoneImpl(const Mat &src_y, const Mat &src_uv, Mat &dst, MI_BOOL swapuv, CvtColorType type, MI_S32 start_row, MI_S32 end_row)
{
    Status ret = Status::ERROR;

    switch (type)
    {
        case CvtColorType::YUV2RGB_NV12:
        case CvtColorType::YUV2RGB_NV21:
        {
            ret = CvtNv2RgbNoneCore<0>(src_y, src_uv, dst, swapuv, start_row, end_row);
            break;
        }

        case CvtColorType::YUV2RGB_NV12_601:
        case CvtColorType::YUV2RGB_NV21_601:
        {
            ret = CvtNv2RgbNoneCore<1>(src_y, src_uv, dst, swapuv, start_row, end_row);
            break;
        }

        default:
        {
            ret = Status::ERROR;
        }
    }

    return ret;
}

Status CvtNv2RgbNone(Context *ctx, const Mat &src_y, const Mat &src_uv, Mat &dst, MI_BOOL swapuv, CvtColorType type, const OpTarget &target)
{
    Status ret = Status::ERROR;

    if ((src_y.GetSizes().m_width & 1) || (src_y.GetSizes().m_height & 1))
    {
        AURA_ADD_ERROR_STRING(ctx, "src and dst mats size only support even");
        return ret;
    }

    if (src_y.GetSizes().m_width != dst.GetSizes().m_width || src_y.GetSizes().m_height != dst.GetSizes().m_height ||
        src_uv.GetSizes().m_width != (src_y.GetSizes().m_width >> 1) || src_uv.GetSizes().m_height != (src_y.GetSizes().m_height >> 1) ||
        src_y.GetSizes().m_channel != 1 || src_uv.GetSizes().m_channel != 2 || dst.GetSizes().m_channel != 3)
    {
        AURA_ADD_ERROR_STRING(ctx, "src and dst mats params do not match");
        return ret;
    }

    MI_S32 height = src_uv.GetSizes().m_height;

    if (target.m_data.none.enable_mt)
    {
        WorkerPool *wp = ctx->GetWorkerPool();
        if (MI_NULL == wp)
        {
            AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
            return Status::ERROR;
        }

        ret = wp->ParallelFor((MI_S32)0, height, CvtNv2RgbNoneImpl, std::cref(src_y), std::cref(src_uv), std::ref(dst), swapuv, type);
    }
    else
    {
        ret = CvtNv2RgbNoneImpl(src_y, src_uv, dst, swapuv, type, 0, height);
    }

    AURA_RETURN(ctx, ret);
}

template <MI_U32 MODE>
static Status CvtY4202RgbNoneCore(const Mat &src_y, const Mat &src_u, const Mat &src_v, Mat &dst, MI_BOOL swapuv, MI_S32 start_row, MI_S32 end_row)
{
    MI_S32 width = dst.GetSizes().m_width;

    for (MI_S32 uv = start_row; uv < end_row; uv++)
    {
        MI_S32 y = uv * 2;

        const MI_U8 *src_y_c = src_y.Ptr<MI_U8>(y);
        const MI_U8 *src_y_n = src_y.Ptr<MI_U8>(y + 1);
        const MI_U8 *src_u_c = swapuv ? src_v.Ptr<MI_U8>(uv) : src_u.Ptr<MI_U8>(uv);
        const MI_U8 *src_v_c = swapuv ? src_u.Ptr<MI_U8>(uv) : src_v.Ptr<MI_U8>(uv);

        MI_U8 *dst_c = dst.Ptr<MI_U8>(y);
        MI_U8 *dst_n = dst.Ptr<MI_U8>(y + 1);

        for (MI_S32 x = 0; x < width; x += 2)
        {
            MI_U8  y00 = src_y_c[x], y01 = src_y_c[x + 1];
            MI_U8  y10 = src_y_n[x], y11 = src_y_n[x + 1];
            MI_U8  u00 = src_u_c[x >> 1], v00 = src_v_c[x >> 1];

            MI_S32 ruv, guv, buv;
            Uv2Rgbuv<MODE>(u00, v00, ruv, guv, buv);
            Yrgbuv2Rgb<MODE>(y00, ruv, guv, buv, dst_c[3 * x + 0], dst_c[3 * x + 1], dst_c[3 * x + 2]);
            Yrgbuv2Rgb<MODE>(y01, ruv, guv, buv, dst_c[3 * x + 3], dst_c[3 * x + 4], dst_c[3 * x + 5]);
            Yrgbuv2Rgb<MODE>(y10, ruv, guv, buv, dst_n[3 * x + 0], dst_n[3 * x + 1], dst_n[3 * x + 2]);
            Yrgbuv2Rgb<MODE>(y11, ruv, guv, buv, dst_n[3 * x + 3], dst_n[3 * x + 4], dst_n[3 * x + 5]);
        }
    }

    return Status::OK;
}

static Status CvtY4202RgbNoneImpl(const Mat &src_y, const Mat &src_u, const Mat &src_v, Mat &dst, MI_BOOL swapuv, CvtColorType type, MI_S32 start_row, MI_S32 end_row)
{
    Status ret = Status::ERROR;

    switch (type)
    {
        case CvtColorType::YUV2RGB_YU12:
        case CvtColorType::YUV2RGB_YV12:
        {
            ret = CvtY4202RgbNoneCore<0>(src_y, src_u, src_v, dst, swapuv, start_row, end_row);
            break;
        }

        case CvtColorType::YUV2RGB_YU12_601:
        case CvtColorType::YUV2RGB_YV12_601:
        {
            ret = CvtY4202RgbNoneCore<1>(src_y, src_u, src_v, dst, swapuv, start_row, end_row);
            break;
        }

        default:
        {
            ret = Status::ERROR;
        }
    }

    return ret;
}

Status CvtY4202RgbNone(Context *ctx, const Mat &src_y, const Mat &src_u, const Mat &src_v, Mat &dst, MI_BOOL swapuv, CvtColorType type, const OpTarget &target)
{
    Status ret = Status::ERROR;

    if ((src_y.GetSizes().m_width & 1) || (src_y.GetSizes().m_height & 1))
    {
        AURA_ADD_ERROR_STRING(ctx, "src and dst mats size only support even");
        return ret;
    }

    if (src_y.GetSizes().m_width != dst.GetSizes().m_width || src_y.GetSizes().m_height != dst.GetSizes().m_height ||
        src_u.GetSizes().m_width != src_v.GetSizes().m_width || src_u.GetSizes().m_height != src_v.GetSizes().m_height ||
        src_u.GetSizes().m_width != (src_y.GetSizes().m_width >> 1) || src_u.GetSizes().m_height != (src_y.GetSizes().m_height >> 1) ||
        src_y.GetSizes().m_channel != 1 || src_u.GetSizes().m_channel != 1 || src_v.GetSizes().m_channel != 1 || dst.GetSizes().m_channel != 3)
    {
        AURA_ADD_ERROR_STRING(ctx, "src and dst mats params do not match");
        return ret;
    }

    MI_S32 height = src_u.GetSizes().m_height;

    if (target.m_data.none.enable_mt)
    {
        WorkerPool *wp = ctx->GetWorkerPool();
        if (MI_NULL == wp)
        {
            AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
            return Status::ERROR;
        }

        ret = wp->ParallelFor((MI_S32)0, height, CvtY4202RgbNoneImpl, std::cref(src_y), std::cref(src_u), std::cref(src_v),
                              std::ref(dst), swapuv, type);
    }
    else
    {
        ret = CvtY4202RgbNoneImpl(src_y, src_u, src_v, dst, swapuv, type, 0, height);
    }

    AURA_RETURN(ctx, ret);
}

template <MI_U32 MODE, MI_U32 SWAPUV, MI_U32 SWAPY>
static Status CvtY4222RgbNoneCore(const Mat &src, Mat &dst, MI_S32 start_row, MI_S32 end_row)
{
    MI_S32 width = dst.GetSizes().m_width;

    const MI_S32 uidx = 1 - SWAPY + SWAPUV * 2;
    const MI_S32 vidx = (2 + uidx) % 4;

    for (MI_S32 y = start_row; y < end_row; y++)
    {
        const MI_U8 *src_c = src.Ptr<MI_U8>(y);
        MI_U8       *dst_c = dst.Ptr<MI_U8>(y);

        for (MI_S32 x = 0; x < width; x += 2)
        {
            MI_U8  y00 = src_c[2 * x + SWAPY], y01 = src_c[2 * x + SWAPY + 2];
            MI_U8  u = src_c[2 * x + uidx], v = src_c[2 * x + vidx];

            MI_S32 ruv, guv, buv;
            Uv2Rgbuv<MODE>(u, v, ruv, guv, buv);
            Yrgbuv2Rgb<MODE>(y00, ruv, guv, buv, dst_c[3 * x + 0], dst_c[3 * x + 1], dst_c[3 * x + 2]);
            Yrgbuv2Rgb<MODE>(y01, ruv, guv, buv, dst_c[3 * x + 3], dst_c[3 * x + 4], dst_c[3 * x + 5]);
        }
    }

    return Status::OK;
}

template <MI_U32 MODE>
static Status CvtY4222RgbNoneCore(const Mat &src, Mat &dst, MI_BOOL swapuv, MI_BOOL swapy, MI_S32 start_row, MI_S32 end_row)
{
    if (swapuv && swapy)
    {
        return CvtY4222RgbNoneCore<MODE, 1, 1>(src, dst, start_row, end_row);
    }
    else if (swapuv)
    {
        return CvtY4222RgbNoneCore<MODE, 1, 0>(src, dst, start_row, end_row);
    }
    else if (swapy)
    {
        return CvtY4222RgbNoneCore<MODE, 0, 1>(src, dst, start_row, end_row);
    }
    else
    {
        return CvtY4222RgbNoneCore<MODE, 0, 0>(src, dst, start_row, end_row);
    }
}

static Status CvtY4222RgbNoneImpl(const Mat &src, Mat &dst, MI_BOOL swapuv, MI_BOOL swapy, CvtColorType type, MI_S32 start_row, MI_S32 end_row)
{
    Status ret = Status::ERROR;

    switch (type)
    {
        case CvtColorType::YUV2RGB_Y422:
        case CvtColorType::YUV2RGB_YUYV:
        case CvtColorType::YUV2RGB_YVYU:
        {
            ret = CvtY4222RgbNoneCore<0>(src, dst, swapuv, swapy, start_row, end_row);
            break;
        }

        case CvtColorType::YUV2RGB_Y422_601:
        case CvtColorType::YUV2RGB_YUYV_601:
        case CvtColorType::YUV2RGB_YVYU_601:
        {
            ret = CvtY4222RgbNoneCore<1>(src, dst, swapuv, swapy, start_row, end_row);
            break;
        }

        default:
        {
            ret = Status::ERROR;
        }
    }

    return ret;
}

Status CvtY4222RgbNone(Context *ctx, const Mat &src, Mat &dst, MI_BOOL swapuv, MI_BOOL swapy, CvtColorType type, const OpTarget &target)
{
    Status ret = Status::ERROR;

    if ((src.GetSizes().m_width & 1))
    {
        AURA_ADD_ERROR_STRING(ctx, "src width only support even");
        return ret;
    }

    if ((src.GetSizes().m_width != dst.GetSizes().m_width) || (src.GetSizes().m_height != dst.GetSizes().m_height) ||
        (src.GetSizes().m_channel != 2) || (dst.GetSizes().m_channel != 3))
    {
        AURA_ADD_ERROR_STRING(ctx, "src and dst mats params do not match");
        return ret;
    }

    MI_S32 height = src.GetSizes().m_height;

    if (target.m_data.none.enable_mt)
    {
        WorkerPool *wp = ctx->GetWorkerPool();
        if (MI_NULL == wp)
        {
            AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
            return Status::ERROR;
        }

        ret = wp->ParallelFor((MI_S32)0, height, CvtY4222RgbNoneImpl, std::cref(src), std::ref(dst), swapuv, swapy, type);
    }
    else
    {
        ret = CvtY4222RgbNoneImpl(src, dst, swapuv, swapy, type, 0, height);
    }

    AURA_RETURN(ctx, ret);
}

template <MI_U32 MODE>
static Status CvtY4442RgbNoneCore(const Mat &src_y, const Mat &src_u, const Mat &src_v, Mat &dst, MI_S32 start_row, MI_S32 end_row)
{
    MI_S32 width = dst.GetSizes().m_width;

    for (MI_S32 y = start_row; y < end_row; y++)
    {
        const MI_U8 *src_y_c = src_y.Ptr<MI_U8>(y);
        const MI_U8 *src_u_c = src_u.Ptr<MI_U8>(y);
        const MI_U8 *src_v_c = src_v.Ptr<MI_U8>(y);

        MI_U8 *dst_c = dst.Ptr<MI_U8>(y);

        for (MI_S32 x = 0; x < width; x++)
        {
            MI_U8  y = src_y_c[x];
            MI_U8  u = src_u_c[x];
            MI_U8  v = src_v_c[x];

            MI_S32 ruv, guv, buv;
            Uv2Rgbuv<MODE>(u, v, ruv, guv, buv);
            Yrgbuv2Rgb<MODE>(y, ruv, guv, buv, dst_c[3 * x + 0], dst_c[3 * x + 1], dst_c[3 * x + 2]);
        }
    }

    return Status::OK;
}

static Status CvtY4442RgbNoneImpl(const Mat &src_y, const Mat &src_u, const Mat &src_v, Mat &dst, CvtColorType type, MI_S32 start_row, MI_S32 end_row)
{
    Status ret = Status::ERROR;

    switch (type)
    {
        case CvtColorType::YUV2RGB_Y444:
        {
            ret = CvtY4442RgbNoneCore<0>(src_y, src_u, src_v, dst, start_row, end_row);
            break;
        }

        case CvtColorType::YUV2RGB_Y444_601:
        {
            ret = CvtY4442RgbNoneCore<1>(src_y, src_u, src_v, dst, start_row, end_row);
            break;
        }

        default:
        {
            ret = Status::ERROR;
        }
    }

    return ret;
}

Status CvtY4442RgbNone(Context *ctx, const Mat &src_y, const Mat &src_u, const Mat &src_v, Mat &dst, CvtColorType type, const OpTarget &target)
{
    Status ret = Status::ERROR;

    if (src_y.GetSizes().m_width != dst.GetSizes().m_width || src_y.GetSizes().m_height != dst.GetSizes().m_height ||
        src_u.GetSizes().m_width != dst.GetSizes().m_width || src_u.GetSizes().m_height != dst.GetSizes().m_height ||
        src_v.GetSizes().m_width != dst.GetSizes().m_width || src_v.GetSizes().m_height != dst.GetSizes().m_height ||
        src_y.GetSizes().m_channel != 1 || src_u.GetSizes().m_channel != 1 || src_v.GetSizes().m_channel != 1 || dst.GetSizes().m_channel != 3)
    {
        AURA_ADD_ERROR_STRING(ctx, "src and dst mats params do not match");
        return ret;
    }

    MI_S32 height = dst.GetSizes().m_height;

    if (target.m_data.none.enable_mt)
    {
        WorkerPool *wp = ctx->GetWorkerPool();
        if (MI_NULL == wp)
        {
            AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
            return Status::ERROR;
        }

        ret = wp->ParallelFor((MI_S32)0, height, CvtY4442RgbNoneImpl, std::cref(src_y), std::cref(src_u), std::cref(src_v),
                              std::ref(dst), type);
    }
    else
    {
        ret = CvtY4442RgbNoneImpl(src_y, src_u, src_v, dst, type, 0, height);
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura
