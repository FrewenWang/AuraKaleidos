#include "equalize_hist_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"

namespace aura
{

static Status EqualizeHistCalcHistNoneImpl(const Mat &src, MI_S32 *hist, std::mutex &hist_mutex, MI_S32 start_row, MI_S32 end_row)
{
    MI_S32 local_hist[256] = {0};
    MI_S32 width           = src.GetSizes().m_width;
    MI_S32 width_align     = width & (-4);

    for (MI_S32 y = start_row; y < end_row; y++)
    {
        const MI_U8 *src_row = src.Ptr<MI_U8>(y);

        MI_S32 x = 0;
        for (; x < width_align; x += 4)
        {
            MI_U8 val0, val1, val2, val3;

            val0 = src_row[x];
            val1 = src_row[x + 1];
            val2 = src_row[x + 2];
            val3 = src_row[x + 3];

            local_hist[val0]++;
            local_hist[val1]++;
            local_hist[val2]++;
            local_hist[val3]++;
        }

        for (; x < width; x++)
        {
            local_hist[src_row[x]]++;
        }
    }

    std::lock_guard<std::mutex> guard(hist_mutex);
    for (MI_S32 i = 0; i < 256; i++)
    {
        hist[i] += local_hist[i];
    }

    return Status::OK;
}

static Status EqualizeHistLutNoneImpl(const Mat &src, Mat &dst, MI_U8 *lut, MI_S32 start_row, MI_S32 end_row)
{
    MI_S32 width       = src.GetSizes().m_width;
    MI_S32 width_align = width & (-4);
    for (MI_S32 y = start_row; y < end_row; y++)
    {
        const MI_U8 *src_row = src.Ptr<MI_U8>(y);
        MI_U8 *dst_row       = dst.Ptr<MI_U8>(y);

        MI_S32 x = 0;
        for (; x < width_align; x += 4)
        {
            MI_U8 val0, val1, val2, val3;
            MI_U8 lut0, lut1, lut2, lut3;

            val0 = src_row[x];
            val1 = src_row[x + 1];
            val2 = src_row[x + 2];
            val3 = src_row[x + 3];

            lut0 = lut[val0];
            lut1 = lut[val1];
            lut2 = lut[val2];
            lut3 = lut[val3];

            dst_row[x]     = lut0;
            dst_row[x + 1] = lut1;
            dst_row[x + 2] = lut2;
            dst_row[x + 3] = lut3;
        }

        for (; x < width; x++)
        {
            dst_row[x] = lut[src_row[x]];
        }
    }

    return Status::OK;
}

EqualizeHistNone::EqualizeHistNone(Context *ctx, const OpTarget &target) : EqualizeHistImpl(ctx, target)
{}

Status EqualizeHistNone::SetArgs(const Array *src, Array *dst)
{
    Status ret = Status::ERROR;

    if (EqualizeHistImpl::SetArgs(src, dst) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "EqualizeHistImpl::SetArgs failed");
        return ret;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return ret;
    }

    return Status::OK;
}

Status EqualizeHistNone::Run()
{
    Status ret = Status::ERROR;

    const Mat *src = dynamic_cast<const Mat *>(m_src);
    Mat *dst = dynamic_cast<Mat *>(m_dst);

    if ((MI_NULL == src) || (MI_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst is null");
        return ret;
    }

    MI_S32 height = src->GetSizes().m_height;
    MI_S32 width  = src->GetSizes().m_width;
    MI_S32 total  = height * width;

    MI_S32 hist[256] = {0};
    MI_U8 lut[256]   = {0};

    std::mutex hist_mutex;
    if (m_target.m_data.none.enable_mt)
    {
        WorkerPool *wp = m_ctx->GetWorkerPool();
        if (MI_NULL == wp)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "GetWorkerPool failed");
            return ret;
        }

        if (wp->ParallelFor(static_cast<MI_S32>(0), height, EqualizeHistCalcHistNoneImpl, std::cref(*src), hist, std::ref(hist_mutex)) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "ParallelFor run EqualizeHistCalcHist failed");
            return ret;
        }
    }
    else
    {
        if (EqualizeHistCalcHistNoneImpl(*src, hist, hist_mutex, static_cast<MI_S32>(0), height) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "ParallelFor run EqualizeHistCalcHist failed");
            return ret;
        }
    }


    MI_S32 idx = 0;
    while (!hist[idx])
    {
        idx++;
    }

    if (total == hist[idx])
    {
        memset(dst->GetData(), idx, dst->GetTotalBytes());
        return Status::OK;
    }

    MI_F32 scale = (256 - 1.f) / (total - hist[idx]);
    MI_S32 sum   = 0;
    for (lut[idx++] = 0; idx < 256; idx++)
    {
        sum     += hist[idx];
        lut[idx] = SaturateCast<MI_U8>(sum * scale);
    }

    if (m_target.m_data.none.enable_mt)
    {
        WorkerPool *wp = m_ctx->GetWorkerPool();
        if (MI_NULL == wp)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "GetWorkerPool failed");
            return ret;
        }

        if (wp->ParallelFor(static_cast<MI_S32>(0), height, EqualizeHistLutNoneImpl, std::cref(*src), std::ref(*dst), lut) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "ParallelFor run EqualizeHistLutNoneImpl failed");
            return ret;
        }
    }
    else
    {
        if (EqualizeHistLutNoneImpl(*src, *dst, lut, static_cast<MI_S32>(0), height) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "EqualizeHistLutNoneImpl run failed");
            return ret;
        }
    }

    return Status::OK;
}

} // namespace aura