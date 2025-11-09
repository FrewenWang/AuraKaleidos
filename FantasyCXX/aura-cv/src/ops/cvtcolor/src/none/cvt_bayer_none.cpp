#include "aura/ops/core.h"
#include "aura/runtime/mat.h"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

template <typename Tp>
Status CvtBayer2BgrNoneImpl(const Mat &src, Mat &dst, DT_BOOL swapb, DT_BOOL swapg, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 width = src.GetSizes().m_width;

    const Tp *src_p  = DT_NULL;
    const Tp *src_c  = DT_NULL;
    const Tp *src_n0 = DT_NULL;
    const Tp *src_n1 = DT_NULL;

    Tp    *dst_c    = DT_NULL;
    Tp    *dst_n    = DT_NULL;
    DT_S32 offset   = 0;
    DT_S32 blue_idx = swapb ? -1 : 1;

    for (DT_S32 y = (start_row * 2); y < (end_row * 2); y += 2)
    {
        if (swapg)
        {
            src_p  = src.Ptr<Tp>(y + 3);
            src_c  = src.Ptr<Tp>(y + 2);
            src_n0 = src.Ptr<Tp>(y + 1);
            src_n1 = src.Ptr<Tp>(y);
            dst_c  = dst.Ptr<Tp>(y + 2);
            dst_n  = dst.Ptr<Tp>(y + 1);
        }
        else
        {
            src_p  = src.Ptr<Tp>(y);
            src_c  = src.Ptr<Tp>(y + 1);
            src_n0 = src.Ptr<Tp>(y + 2);
            src_n1 = src.Ptr<Tp>(y + 3);
            dst_c  = dst.Ptr<Tp>(y + 1);
            dst_n  = dst.Ptr<Tp>(y + 2);
        }

        for (DT_S32 x = 1; x < width - 1; x += 2)
        {
            offset                   = 3 * x + 1;
            dst_c[offset - blue_idx] = (src_c[x - 1] + src_c[x + 1] + 1) >> 1;
            dst_c[offset]            = src_c[x];
            dst_c[offset + blue_idx] = (src_p[x] + src_n0[x] + 1) >> 1;

            offset                   = 3 * x + 4;
            dst_c[offset - blue_idx] = src_c[x + 1];
            dst_c[offset]            = (src_p[x + 1] + src_c[x] + src_c[x + 2] + src_n0[x + 1] + 2) >> 2;
            dst_c[offset + blue_idx] = (src_p[x] + src_p[x + 2] + src_n0[x] + src_n0[x + 2] + 2) >> 2;

            offset                   = 3 * x + 1;
            dst_n[offset - blue_idx] = (src_c[x - 1] + src_c[x + 1] + src_n1[x - 1] + src_n1[x + 1] + 2) >> 2;
            dst_n[offset]            = (src_c[x] + src_n0[x - 1] + src_n0[x + 1] + src_n1[x] + 2) >> 2;
            dst_n[offset + blue_idx] = src_n0[x];

            offset                   = 3 * x + 4;
            dst_n[offset - blue_idx] = (src_c[x + 1] + src_n1[x + 1] + 1) >> 1;
            dst_n[offset]            = src_n0[x + 1];
            dst_n[offset + blue_idx] = (src_n0[x] + src_n0[x + 2] + 1) >> 1;
        }

        offset            = 3 * width;
        dst_c[0]          = dst_c[3];
        dst_c[1]          = dst_c[4];
        dst_c[2]          = dst_c[5];
        dst_c[offset - 3] = dst_c[offset - 6];
        dst_c[offset - 2] = dst_c[offset - 5];
        dst_c[offset - 1] = dst_c[offset - 4];

        dst_n[0]          = dst_n[3];
        dst_n[1]          = dst_n[4];
        dst_n[2]          = dst_n[5];
        dst_n[offset - 3] = dst_n[offset - 6];
        dst_n[offset - 2] = dst_n[offset - 5];
        dst_n[offset - 1] = dst_n[offset - 4];
    }

    return Status::OK;
}

static Status CvtBayer2BgrRemainNoneImpl(Mat &dst)
{
    DT_S32 height = dst.GetSizes().m_height;
    DT_S32 width  = dst.GetSizes().m_width;

    DT_VOID *dst_c = dst.Ptr<DT_VOID>(height - 1);
    DT_VOID *dst_n = dst.Ptr<DT_VOID>(height - 2);
    memcpy(dst_c, dst_n, 3 * width * ElemTypeSize(dst.GetElemType()));

    dst_c = dst.Ptr<DT_VOID>(0);
    dst_n = dst.Ptr<DT_VOID>(1);
    memcpy(dst_c, dst_n, 3 * width * ElemTypeSize(dst.GetElemType()));

    return Status::OK;
}

Status CvtBayer2BgrNone(Context *ctx, const Mat &src, Mat &dst, DT_BOOL swapb, DT_BOOL swapg, const OpTarget &target)
{
    Status ret = Status::ERROR;

    if (src.GetSizes().m_height & 1 || dst.GetSizes().m_width & 1)
    {
        AURA_ADD_ERROR_STRING(ctx, "src and dst size only support even");
        return Status::ERROR;
    }

    if (dst.GetSizes().m_height != src.GetSizes().m_height || dst.GetSizes().m_width != dst.GetSizes().m_width)
    {
        AURA_ADD_ERROR_STRING(ctx, "src and dst must have the same height and width");
        return Status::ERROR;
    }

    if (src.GetSizes().m_channel != 1 || dst.GetSizes().m_channel != 3)
    {
        AURA_ADD_ERROR_STRING(ctx, "src channel must be 1 and dst channel must be 3");
        return Status::ERROR;
    }

    DT_S32 worker_height = (src.GetSizes().m_height - 2) / 2;

#define CVT_BAYER_NONE_IMPL(type)                                                                                                 \
    if (target.m_data.none.enable_mt)                                                                                             \
    {                                                                                                                             \
        WorkerPool *wp = ctx->GetWorkerPool();                                                                                    \
        if (DT_NULL == wp)                                                                                                        \
        {                                                                                                                         \
            AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");                                                                   \
            return Status::ERROR;                                                                                                 \
        }                                                                                                                         \
        ret = wp->ParallelFor((DT_S32)0, worker_height, CvtBayer2BgrNoneImpl<type>, std::cref(src), std::ref(dst), swapb, swapg); \
    }                                                                                                                             \
    else                                                                                                                          \
    {                                                                                                                             \
        ret = CvtBayer2BgrNoneImpl<type>(src, dst, swapb, swapg, 0, worker_height);                                               \
    }

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            CVT_BAYER_NONE_IMPL(DT_U8)
            break;
        }

        case ElemType::U16:
        {
            CVT_BAYER_NONE_IMPL(DT_U16)
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "elem type error");
            ret = Status::ERROR;
        }
    }

    ret |= CvtBayer2BgrRemainNoneImpl(dst);

#undef CVT_BAYER_NONE_IMPL

    AURA_RETURN(ctx, ret);
} // namespace aura

} // namespace aura