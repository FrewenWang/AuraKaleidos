#include "calc_hist_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/thread_buffer.h"
#include "aura/runtime/logger.h"

namespace aura
{

template <typename Tp>
static Status CalcHistNoneImpl(Context *ctx, const Mat &src, const DT_S32 channel, const Mat &mask, DT_U32 *hist_local,
                               const DT_S32 max_size, ThreadBuffer &thread_buffer, std::mutex &mutex, DT_S32 start_row, DT_S32 end_row)
{
    const DT_S32 width        = src.GetSizes().m_width;
    const DT_S32 ichannel     = src.GetSizes().m_channel;
    const DT_S32 channel2     = ichannel * 2;
    const DT_S32 channel3     = ichannel * 3;
    const DT_S32 width_align4 = width & (-4);

    DT_U32 *hist = thread_buffer.GetThreadData<DT_U32>();

    if (!hist)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    memset(hist, 0, max_size * sizeof(DT_U32));

    if (mask.IsValid())
    {
        for (DT_S32 y = start_row; y < end_row; y++)
        {
            const Tp    *src_row  = src.Ptr<Tp>(y);
            const DT_U8 *mask_row = mask.Ptr<DT_U8>(y);

            DT_S32 x = 0;
            for (; x < width_align4; x += 4)
            {
                DT_S32 index = x * ichannel + channel;
                Tp t0, t1, t2, t3;
                t0 = src_row[index];
                t1 = src_row[index + ichannel];
                t2 = src_row[index + channel2];
                t3 = src_row[index + channel3];

                hist[t0] = mask_row[x + 0] ? hist[t0] + 1 : hist[t0];
                hist[t1] = mask_row[x + 1] ? hist[t1] + 1 : hist[t1];
                hist[t2] = mask_row[x + 2] ? hist[t2] + 1 : hist[t2];
                hist[t3] = mask_row[x + 3] ? hist[t3] + 1 : hist[t3];
            }
            for (; x < width; x++)
            {
                DT_S32 index = x * ichannel + channel;
                hist[src_row[index]] = mask_row[x] ? hist[src_row[index]] + 1 : hist[src_row[index]];
            }
        }
    }
    else
    {
        for (DT_S32 y = start_row; y < end_row; y++)
        {
            const Tp *src_row = src.Ptr<Tp>(y);

            DT_S32 x = 0;
            for (; x < width_align4; x += 4)
            {
                DT_S32 index = x * ichannel + channel;

                Tp t0, t1, t2, t3;
                t0 = src_row[index];
                t1 = src_row[index + ichannel];
                t2 = src_row[index + channel2];
                t3 = src_row[index + channel3];

                hist[t0]++;
                hist[t1]++;
                hist[t2]++;
                hist[t3]++;
            }
            for (; x < width; x++)
            {
                DT_S32 index = x * ichannel + channel;
                hist[src_row[index]]++;
            }
        }
    }

    std::lock_guard<std::mutex> guard(mutex);
    for (DT_S32 i = 0; i < max_size; i++)
    {
        hist_local[i] += hist[i];
    }

    return Status::OK;
}

template <typename Tp>
static Status CalcHistNoneHelper(Context *ctx, const Mat &src, DT_S32 channel, const Mat &mask, std::vector<DT_U32> &dst,
                                 DT_S32 hist_size, DT_S32 max_size, const Scalar &ranges, DT_BOOL accumulate, const OpTarget &target)
{
    Status ret = Status::ERROR;

    AURA_UNUSED(target);

    const DT_S32 height = src.GetSizes().m_height;

    DT_S32 hist_low  = ranges.m_val[0];
    DT_S32 hist_high = ranges.m_val[1];

    DT_U32 *hist_local = static_cast<DT_U32*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, max_size * sizeof(DT_U32), 0));
    if (DT_NULL == hist_local)
    {
        AURA_ADD_ERROR_STRING(ctx, "AURA_ALLOC_PARAM fail");
        return ret;
    }

    std::mutex mutex;

    if (target.m_data.none.enable_mt)
    {
        WorkerPool *wp = ctx->GetWorkerPool();
        if (DT_NULL == wp)
        {
            AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
            AURA_FREE(ctx, hist_local);
            return ret;
        }

        ThreadBuffer thread_buffer(ctx, max_size * sizeof(DT_U32));

        if (wp->ParallelFor(static_cast<DT_S32>(0), height, CalcHistNoneImpl<Tp>, ctx, std::cref(src), channel, std::cref(mask),
                            hist_local, max_size, std::ref(thread_buffer), std::ref(mutex)) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "ParallelFor run CalcHistNoneImpl failed");
            AURA_FREE(ctx, hist_local);
            return ret;
        }
    }
    else
    {
        ThreadBuffer thread_buffer(ctx, max_size * sizeof(DT_U32));

        if (CalcHistNoneImpl<Tp>(ctx, src, channel, mask, hist_local, max_size, thread_buffer, mutex,
                                 static_cast<DT_S32>(0), height) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "CalcHistNoneImpl run failed");
            AURA_FREE(ctx, hist_local);
            return ret;
        }
    }

    if (DT_FALSE == accumulate)
    {
        std::fill(dst.begin(), dst.end(), 0);
    }

    DT_F64 scale = (DT_F64)(hist_size) / (hist_high - hist_low);
    DT_S32 j = 0;
    for (DT_S32 i = hist_low; i < hist_high; i++)
    {
        DT_S32 idx = Floor(j * scale);
        dst[idx] += hist_local[i];
        j++;
    }

    AURA_FREE(ctx, hist_local);
    return Status::OK;
}

CalcHistNone::CalcHistNone(Context *ctx, const OpTarget &target) : CalcHistImpl(ctx, target)
{}

Status CalcHistNone::SetArgs(const Array *src, DT_S32 channel, std::vector<DT_U32> &hist, DT_S32 hist_size,
                             const Scalar &ranges, const Array *mask, DT_BOOL accumulate)
{
    Status ret = Status::ERROR;

    if (CalcHistImpl::SetArgs(src, channel, hist, hist_size, ranges, mask, accumulate) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "CalcHistImpl::SetArgs failed");
        return ret;
    }

    if (src->GetArrayType() != ArrayType::MAT)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src must be mat type");
        return ret;
    }

    if (src->GetElemType() != ElemType::U8 && src->GetElemType() != ElemType::U16)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src element type must be u8/u16");
        return ret;
    }

    return Status::OK;
}

Status CalcHistNone::Run()
{
    Status ret = Status::ERROR;

    const Mat *src = dynamic_cast<const Mat *>(m_src);
    const Mat *mask = dynamic_cast<const Mat *>(m_mask);
    if ((DT_NULL == src) || (DT_NULL == mask))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src mask is null");
        return ret;
    }

    switch (src->GetElemType())
    {
        case ElemType::U8:
        {
            ret = CalcHistNoneHelper<DT_U8>(m_ctx, *src, m_channel, *mask, *m_hist, m_hist_size, 256, m_ranges, m_accumulate, m_target);
            break;
        }

        case ElemType::U16:
        {
            ret = CalcHistNoneHelper<DT_U16>(m_ctx, *src, m_channel, *mask, *m_hist, m_hist_size, 65536, m_ranges, m_accumulate, m_target);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "elem type error");
            ret = Status::ERROR;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura