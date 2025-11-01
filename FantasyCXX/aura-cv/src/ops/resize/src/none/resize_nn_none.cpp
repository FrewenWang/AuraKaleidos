#include "resize_impl.hpp"
#include "aura/ops/core.h"
#include "aura/runtime/mat.h"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

static AURA_VOID GetNnOffset(MI_S32 *offset, MI_S32 iwidth, MI_S32 owidth, MI_F64 xratio)
{
    MI_S32 x;
    MI_S32 t_max = static_cast<MI_S32>(Ceil((iwidth - 1) / xratio));
    MI_S32 xmax = Min(owidth, t_max);

    for (x = 0; x < xmax - 7; x += 8)
    {
        offset[x] = static_cast<MI_S32>(Floor(x * xratio));
        offset[x + 1] = static_cast<MI_S32>(Floor((x + 1) * xratio));
        offset[x + 2] = static_cast<MI_S32>(Floor((x + 2) * xratio));
        offset[x + 3] = static_cast<MI_S32>(Floor((x + 3) * xratio));
        offset[x + 4] = static_cast<MI_S32>(Floor((x + 4) * xratio));
        offset[x + 5] = static_cast<MI_S32>(Floor((x + 5) * xratio));
        offset[x + 6] = static_cast<MI_S32>(Floor((x + 6) * xratio));
        offset[x + 7] = static_cast<MI_S32>(Floor((x + 7) * xratio));
    }
    for (; x < xmax; x++)
    {
        offset[x] = static_cast<MI_S32>(Floor(x * xratio));
    }

    for (; x < owidth; x++)
    {
        offset[x] = iwidth - 1;
    }
}

// FIXME: row/col idx compute may has 1 error, eg: 6.99996 vs 7.0, use #pragma optimize on/off to solve it.
template <typename Tp>
static Status ResizeNnNoneImpl(const Mat &src, Mat &dst, MI_S32 *xofs, MI_S32 *yofs, MI_S32 start_row, MI_S32 end_row)
{
    if ((MI_NULL == xofs) || (MI_NULL == yofs))
    {
        return Status::ERROR;
    }

    MI_S32 channel = dst.GetSizes().m_channel;
    MI_S32 owidth  = dst.GetSizes().m_width;
    for(MI_S32 y = start_row; y < end_row; y++)
    {
        const Tp *src_c = src.Ptr<Tp>(yofs[y]);
        Tp *dst_c = dst.Ptr<Tp>(y);

        for (MI_S32 x = 0; x < owidth; x++)
        {
            for (MI_S32 c = 0; c < channel; c++)
            {
                dst_c[x * channel + c] = src_c[xofs[x] * channel + c];
            }
        }
    }

    return Status::OK;
}

Status ResizeNnNone(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    Status ret = Status::ERROR;

    MI_S32 iwidth  = src.GetSizes().m_width;
    MI_S32 iheight = src.GetSizes().m_height;
    MI_S32 owidth  = dst.GetSizes().m_width;
    MI_S32 oheight = dst.GetSizes().m_height;

    MI_F64 inv_scale_x = static_cast<MI_F64>(owidth) / iwidth;
    MI_F64 inv_scale_y = static_cast<MI_F64>(oheight) / iheight;
    MI_F64 scale_x = 1. / inv_scale_x;
    MI_F64 scale_y = 1. / inv_scale_y;

    MI_S32 *xofs = static_cast<MI_S32*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, owidth  * sizeof(MI_S32), 0));
    MI_S32 *yofs = static_cast<MI_S32*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, oheight * sizeof(MI_S32), 0));

    if ((MI_NULL == xofs) || (MI_NULL == yofs))
    {
        AURA_FREE(ctx, xofs);
        AURA_FREE(ctx, yofs);
        AURA_ADD_ERROR_STRING(ctx, "AURA_ALLOC_PARAM fail");
        return Status::ERROR;
    }

    GetNnOffset(xofs, iwidth,  owidth,  scale_x);
    GetNnOffset(yofs, iheight, oheight, scale_y);

#define RESIZE_NN_NONE_IMPL(type)                                                                                                       \
    if (target.m_data.none.enable_mt)                                                                                                   \
    {                                                                                                                                   \
        WorkerPool *wp = ctx->GetWorkerPool();                                                                                          \
        if (MI_NULL == wp)                                                                                                              \
        {                                                                                                                               \
            AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");                                                                         \
            return Status::ERROR;                                                                                                       \
        }                                                                                                                               \
        ret = wp->ParallelFor(static_cast<MI_S32>(0), oheight, ResizeNnNoneImpl<type>, std::cref(src), std::ref(dst), xofs, yofs);      \
    }                                                                                                                                   \
    else                                                                                                                                \
    {                                                                                                                                   \
        ret = ResizeNnNoneImpl<type>(src, dst, xofs, yofs, 0, oheight);                                                                 \
    }                                                                                                                                   \
    if (ret != Status::OK)                                                                                                              \
    {                                                                                                                                   \
        MI_CHAR error_msg[128];                                                                                                         \
        std::snprintf(error_msg, sizeof(error_msg), "ResizeNnNoneImpl<%s> failed", #type);                                              \
        AURA_ADD_ERROR_STRING(ctx, error_msg);                                                                                          \
    }

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            RESIZE_NN_NONE_IMPL(MI_U8);
            break;
        }

        case ElemType::S8:
        {
            RESIZE_NN_NONE_IMPL(MI_S8);
            break;
        }

        case ElemType::U16:
        {
            RESIZE_NN_NONE_IMPL(MI_U16);
            break;
        }

        case ElemType::S16:
        {
            RESIZE_NN_NONE_IMPL(MI_S16);
            break;
        }

#if defined(AURA_BUILD_HOST)
        case ElemType::F16:
        {
            RESIZE_NN_NONE_IMPL(MI_F16);
            break;
        }

        case ElemType::F32:
        {
            RESIZE_NN_NONE_IMPL(MI_F32);
            break;
        }
#endif // AURA_BUILD_HOST

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "elem type error");
            ret = Status::ERROR;
        }
    }

    AURA_FREE(ctx, xofs);
    AURA_FREE(ctx, yofs);
    AURA_RETURN(ctx, ret);
}

} // namespace aura