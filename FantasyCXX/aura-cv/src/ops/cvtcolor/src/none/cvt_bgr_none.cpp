#include "cvtcolor_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

Status CvtBgr2BgraNoneImpl(const Mat &src, Mat &dst, MI_BOOL swapb, MI_S32 start_row, MI_S32 end_row)
{
    MI_S32 width    = src.GetSizes().m_width;
    MI_S32 ichannel = src.GetSizes().m_channel;
    MI_S32 ochannel = dst.GetSizes().m_channel;

    MI_S32 blue_idx = swapb ? 2 : 0;
    MI_S32 red_idx  = swapb ? 0 : 2;

    for (MI_S32 y = start_row; y < end_row; ++y)
    {
        const MI_U8 *src_row = src.Ptr<MI_U8>(y);
        MI_U8       *dst_row = dst.Ptr<MI_U8>(y);

        for (MI_S32 x = 0, ix = 0, ox = 0; x < width; x++, ix += ichannel, ox += ochannel)
        {
            dst_row[ox + blue_idx] = src_row[ix];
            dst_row[ox + 1]        = src_row[ix + 1];
            dst_row[ox + red_idx]  = src_row[ix + 2];
            if (4 == ochannel)
            {
                dst_row[ox + 3] = 255;
            }
        }
    }

    return Status::OK;
}

Status CvtBgr2BgraNone(Context *ctx, const Mat &src, Mat &dst, MI_BOOL swapb, const OpTarget &target)
{
    Status ret = Status::ERROR;

    if (src.GetSizes().m_height != dst.GetSizes().m_height || src.GetSizes().m_width != dst.GetSizes().m_width)
    {
        AURA_ADD_ERROR_STRING(ctx, "src and dst must have the same height and width");
        return ret;
    }

    if ((src.GetSizes().m_channel != 3 && src.GetSizes().m_channel != 4) ||
        (dst.GetSizes().m_channel != 3 && dst.GetSizes().m_channel != 4))
    {
        AURA_ADD_ERROR_STRING(ctx, "src and dst channel should be 3 or 4");
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

        ret = wp->ParallelFor((MI_S32)0, height, CvtBgr2BgraNoneImpl, std::cref(src), std::ref(dst), swapb);
    }
    else
    {
        ret = CvtBgr2BgraNoneImpl(src, dst, swapb, 0, height);
    }

    return ret;
}

Status CvtBgr2GrayNoneImpl(const Mat &src, Mat &dst, MI_BOOL swapb, MI_S32 start_row, MI_S32 end_row)
{
    MI_S32 width    = src.GetSizes().m_width;
    MI_S32 ichannel = src.GetSizes().m_channel;

    // Gray = R * 0.299 + G * 0.587 + B * 0.114
    // static constexpr MI_S32 BC = 3735;  // Round(0.114f  * (1 << 15));
    // static constexpr MI_S32 GC = 19235; // Round(0.587f  * (1 << 15));
    // static constexpr MI_S32 RC = 9798;  // Round(0.299f  * (1 << 15));
    MI_S32 b_coeff = Bgr2GrayParam::BC;
    MI_S32 g_coeff = Bgr2GrayParam::GC;
    MI_S32 r_coeff = Bgr2GrayParam::RC;

    if (swapb)
    {
        Swap(b_coeff, r_coeff);
    }
    /// 遍历每一行的数据
    for (MI_S32 y = start_row; y < end_row; y++)
    {
        /// 找到对应的
        const MI_U8 *src_row = src.Ptr<MI_U8>(y);
        MI_U8       *dst_row = dst.Ptr<MI_U8>(y);
        /// ix每次移动是哪个通道。output的每次移动一个通道
        for (MI_S32 ix = 0, ox = 0; ox < width; ix += ichannel, ox++)
        {
            MI_S32 b = src_row[ix];
            MI_S32 g = src_row[ix + 1];
            MI_S32 r = src_row[ix + 2];
            /// 移位回去，进行四舍五入计算
            dst_row[ox] = ShiftSatCast<MI_S32, MI_U8, 15>(b * b_coeff + g * g_coeff + r * r_coeff);
        }
    }

    return Status::OK;
}

Status CvtBgr2GrayNone(Context *ctx, const Mat &src, Mat &dst, MI_BOOL swapb, const OpTarget &target)
{
    Status ret = Status::ERROR;

    if (src.GetSizes().m_height != dst.GetSizes().m_height || src.GetSizes().m_width != dst.GetSizes().m_width)
    {
        AURA_ADD_ERROR_STRING(ctx, "src and dst must have the same height and width");
        return ret;
    }

    if ((src.GetSizes().m_channel != 3 && src.GetSizes().m_channel != 4) || dst.GetSizes().m_channel != 1)
    {
        AURA_ADD_ERROR_STRING(ctx, "src channel must be 3 or 4 adn dst channel must be 1");
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

        ret = wp->ParallelFor((MI_S32)0, height, CvtBgr2GrayNoneImpl, std::cref(src), std::ref(dst), swapb);
    }
    else
    {
        ret = CvtBgr2GrayNoneImpl(src, dst, swapb, 0, height);
    }

    return ret;
}

Status CvtGray2BgrNoneImpl(const Mat &src, Mat &dst, MI_S32 start_row, MI_S32 end_row)
{
    MI_S32 width    = src.GetSizes().m_width;
    MI_S32 ochannel = dst.GetSizes().m_channel;

    for (MI_S32 y = start_row; y < end_row; ++y)
    {
        const MI_U8 *src_row = src.Ptr<MI_U8>(y);
        MI_U8       *dst_row = dst.Ptr<MI_U8>(y);

        for (MI_S32 ix = 0, ox = 0; ix < width; ix++, ox += ochannel)
        {
            dst_row[ox]     = src_row[ix];
            dst_row[ox + 1] = src_row[ix];
            dst_row[ox + 2] = src_row[ix];
            if (4 == ochannel)
            {
                dst_row[ox + 3] = 255;
            }
        }
    }

    return Status::OK;
}

Status CvtGray2BgrNone(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    Status ret = Status::ERROR;

    if (src.GetSizes().m_height != dst.GetSizes().m_height || src.GetSizes().m_width != dst.GetSizes().m_width)
    {
        AURA_ADD_ERROR_STRING(ctx, "src and dst must have the same height and width");
        return ret;
    }

    if (src.GetSizes().m_channel != 1 || (dst.GetSizes().m_channel != 3 && dst.GetSizes().m_channel != 4))
    {
        AURA_ADD_ERROR_STRING(ctx, "src channel must be 1 and dst channel must be 3 or 4");
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

        ret = wp->ParallelFor((MI_S32)0, height, CvtGray2BgrNoneImpl, std::cref(src), std::ref(dst));
    }
    else
    {
        ret = CvtGray2BgrNoneImpl(src, dst, 0, height);
    }

    return ret;
}

} // namespace aura