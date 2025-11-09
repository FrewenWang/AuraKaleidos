#include "resize_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/thread_buffer.h"
#include "aura/runtime/logger.h"

namespace aura
{

template <typename Tp>
static Status ResizeAreaFastNoneImpl(Context *ctx, const Mat &src, Mat &dst, const DT_S32 *area_offset, const DT_S32 *x_offset,
                                     DT_S32 scale_x, DT_S32 scale_y)
{
    using FastSumType = typename ResizeAreaTraits<Tp>::FastSumType;
    if (DT_NULL == area_offset || DT_NULL == x_offset)
    {
        AURA_ADD_ERROR_STRING(ctx, "ResizeAreaFastNoneImpl NULL ptr");
        return Status::ERROR;
    }

    DT_S32 channel = src.GetSizes().m_channel;
    DT_S32 owidth  = dst.GetSizes().m_width;
    DT_S32 oheight = dst.GetSizes().m_height;

    DT_S32 area = scale_x * scale_y;
    DT_F32 scale = 1.f / area;
    DT_S32 owidth_x_cn = owidth * channel;

    for (DT_S32 dy = 0; dy < oheight; dy++)
    {
        Tp *dst_c = dst.Ptr<Tp>(dy);
        DT_S32 sy = dy * scale_y;

        for (DT_S32 dx = 0; dx < owidth_x_cn; dx++)
        {
            const Tp *src_c = src.Ptr<Tp>(sy) + x_offset[dx];

            FastSumType sum = static_cast<FastSumType>(0.f);
            for (DT_S32 k = 0; k < area; k++)
            {
                sum += static_cast<FastSumType>(src_c[area_offset[k]]);
            }
            dst_c[dx] = SaturateCast<Tp>(sum * scale);
        }
    }

    return Status::OK;
}

template <typename Tp>
static Status ResizeAreaNoneHelper(Context *ctx, const Mat &src, Mat &dst)
{
    DT_S32 iwidth  = src.GetSizes().m_width;
    DT_S32 iheight = src.GetSizes().m_height;
    DT_S32 istride = src.GetRowPitch();
    DT_S32 channel = src.GetSizes().m_channel;
    DT_S32 owidth  = dst.GetSizes().m_width;
    DT_S32 oheight = dst.GetSizes().m_height;

    DT_F32 scale_x = static_cast<DT_F32>(iwidth) / owidth;
    DT_F32 scale_y = static_cast<DT_F32>(iheight) / oheight;
    DT_S32 int_scale_x = SaturateCast<DT_S32>(scale_x);
    DT_S32 int_scale_y = SaturateCast<DT_S32>(scale_y);
    DT_S32 fast_flag   = (NearlyEqual(scale_x, int_scale_x) && NearlyEqual(scale_y, int_scale_y)) ? 1 : 0;

    Status ret = Status::ERROR;

    if (scale_x < 1.0 || scale_y < 1.0) /**> upscale using bilinear */
    {
        ret = ResizeBnNoneImpl<Tp>(ctx, src, dst, DT_TRUE);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "ResizeBnNoneImpl<Tp> failed");
        }
    }
    else if (fast_flag)
    {
        DT_S32 area = int_scale_x * int_scale_y;
        DT_S32 src_step = istride / sizeof(Tp);
        DT_S32 *buffer = static_cast<DT_S32*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, (channel * owidth + area) * sizeof(DT_S32), 0));
        if (DT_NULL == buffer)
        {
            AURA_ADD_ERROR_STRING(ctx, "AURA_ALLOC_PARAM failed");
            return Status::ERROR;
        }

        DT_S32 *area_offset = buffer;
        DT_S32 *x_offset = area_offset + area;

        for (DT_S32 sy = 0, k = 0; sy < int_scale_y; sy++)
        {
            for (DT_S32 sx = 0; sx < int_scale_x; sx++)
            {
                area_offset[k++] = sy * src_step + sx * channel;
            }
        }

        for (DT_S32 dx = 0; dx < owidth; dx++)
        {
            DT_S32 j = dx * channel;
            DT_S32 sx = int_scale_x * j;
            for (DT_S32 k = 0; k < channel; k++)
            {
                x_offset[j + k] = sx + k;
            }
        }

        ret = ResizeAreaFastNoneImpl<Tp>(ctx, src, dst, area_offset, x_offset, int_scale_x, int_scale_y);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "ResizeAreaFastNoneImpl<Tp> failed");
        }

        AURA_FREE(ctx, buffer);
    }
    else
    {
        AreaDecimateAlpha *x_tab = static_cast<AreaDecimateAlpha*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, 2 * iwidth * sizeof(AreaDecimateAlpha), 0));
        AreaDecimateAlpha *y_tab = static_cast<AreaDecimateAlpha*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, 2 * iheight * sizeof(AreaDecimateAlpha), 0));
        DT_S32 *tab_offset = static_cast<DT_S32*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, (oheight + 1) * sizeof(DT_S32), 0));
        if ((DT_NULL == x_tab) || (DT_NULL == y_tab) || (DT_NULL == tab_offset))
        {
            AURA_FREE(ctx, x_tab);
            AURA_FREE(ctx, y_tab);
            AURA_FREE(ctx, tab_offset);
            AURA_ADD_ERROR_STRING(ctx, "AURA_ALLOC_PARAM failed");
            return Status::ERROR;
        }

        DT_S32 x_tab_size = GetAreaOffset(iwidth, owidth, channel, scale_x, x_tab);
        DT_S32 y_tab_size = GetAreaOffset(iheight, oheight, 1, scale_y, y_tab);
        DT_S32 dy = 0;
        for (DT_S32 k = 0; k < y_tab_size; k++)
        {
            if ((0 == k) || y_tab[k].di != y_tab[k - 1].di)
            {
                tab_offset[dy++] = k;
            }
        }
        tab_offset[dy] = y_tab_size;

        ThreadBuffer thread_buffer(ctx, channel * owidth * 2 * sizeof(DT_F32));

        ret = ResizeAreaCommNoneImpl<Tp>(ctx, src, dst, thread_buffer, x_tab, x_tab_size, y_tab, tab_offset, 0, oheight);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "ResizeAreaCommNoneImpl<Tp> failed");
        }

        AURA_FREE(ctx, x_tab);
        AURA_FREE(ctx, y_tab);
        AURA_FREE(ctx, tab_offset);
    }

    AURA_RETURN(ctx, ret);
}

Status ResizeAreaNone(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    AURA_UNUSED(target);
    Status ret = Status::ERROR;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = ResizeAreaNoneHelper<DT_U8>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaCommNoneImpl failed, type: DT_U8");
            }
            break;
        }

        case ElemType::S8:
        {
            ret = ResizeAreaNoneHelper<DT_S8>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaCommNoneImpl failed, type: DT_S8");
            }
            break;
        }

        case ElemType::U16:
        {
            ret = ResizeAreaNoneHelper<DT_U16>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaCommNoneImpl failed, type: DT_U16");
            }
            break;
        }

        case ElemType::S16:
        {
            ret = ResizeAreaNoneHelper<DT_S16>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaCommNoneImpl failed, type: DT_S16");
            }
            break;
        }

#if defined(AURA_BUILD_HOST)
        case ElemType::F16:
        {
            ret = ResizeAreaNoneHelper<MI_F16>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaCommNoneImpl failed, type: MI_F16");
            }
            break;
        }

        case ElemType::F32:
        {
            ret = ResizeAreaNoneHelper<DT_F32>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaCommNoneImpl failed, type: DT_F32");
            }
            break;
        }
#endif // AURA_BUILD_HOST

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "elem type error");
            ret = Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura