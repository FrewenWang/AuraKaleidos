#include "aura/ops/hist/calc_hist.hpp"
#include "calc_hist_impl.hpp"
#include "hist_comm.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/thread_buffer.h"
#include "aura/runtime/logger.h"

#if defined(__cplusplus)
extern "C" {
#endif // __cplusplus

AURA_VOID HistogramPernRow(MI_U8 *src, MI_S32 stride, MI_S32 width, MI_S32 height, MI_S32 *hist);
AURA_VOID HistogramBorderPernRow(MI_U8 *src, MI_S32 stride, MI_S32 width, MI_S32 height, MI_S32 *hist, MI_S32 border);
AURA_VOID HistogramBorderMaskPernRow(MI_U8 *src, MI_S32 stride, MI_S32 width, MI_S32 height,
                                   MI_S32 *hist, MI_S32 border, MI_U8 *mask, MI_S32 mask_stride);
AURA_VOID ClearHistogram(MI_S32 *hist);

#if defined(__cplusplus)
}
#endif // __cplusplus

namespace aura
{

template <MI_BOOL IS_MASK>
Status CalcHistU8HvxImpl(Context *ctx, const Mat &src, MI_U32 *dst, const Mat &mask,
                         const MI_S32 max_size, ThreadBuffer &thread_buffer, std::mutex &mutex,
                         MI_S32 start_row, MI_S32 end_row)
{
    MI_S32 *hist = thread_buffer.GetThreadData<MI_S32>();

    if (!hist)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    MI_S32 width  = src.GetSizes().m_width;
    MI_S32 istride = src.GetStrides().m_width;

    memset(hist, 0, max_size * sizeof(MI_U32));

    ClearHistogram(hist);

    MI_S32 i, k, n;
    n = Max((MI_S32)(8192 / istride), (MI_S32)1);

    if (IS_MASK)
    {
        MI_S32 mask_stride = mask.GetStrides().m_width;

        MI_U64 l2fetch_src_size  = L2PfParam(istride, width, n, 1);
        MI_U64 l2fetch_mask_size = L2PfParam(mask_stride, width, n, 1);

        // HVX-optimized implementation
        for (i = start_row; i < end_row; i += n)
        {
            k = (end_row - i) > n ? n : (end_row - i);
            // fetch next row
            if (i + n < end_row)
            {
                if (i + 2 * n >= end_row)
                {
                    l2fetch_src_size  = L2PfParam(istride, width, end_row - i - n, 1);
                    l2fetch_mask_size = L2PfParam(mask_stride, width, end_row - i - n, 1);
                }

                L2Fetch(reinterpret_cast<MI_U32>(src.Ptr<MI_U8>(i + n)), l2fetch_src_size);
                L2Fetch(reinterpret_cast<MI_U32>(mask.Ptr<MI_U8>(i + n)), l2fetch_mask_size);
            }

            MI_U8 *src_data = const_cast<MI_U8 *>(src.Ptr<MI_U8>(i));
            MI_U8 *mask_data = const_cast<MI_U8 *>(mask.Ptr<MI_U8>(i));
            HistogramBorderMaskPernRow(src_data, istride, width, k, hist, 0, mask_data, mask_stride);
        }
    }
    else
    {
        MI_U64 l2fetch_src_size = L2PfParam(istride, width, n, 1);
        // HVX-optimized implementation
        for (i = start_row; i < end_row; i += n)
        {
            k = (end_row - i) > n ? n : (end_row - i);
            // fetch next row
            if (i + n < end_row)
            {
                if (i + 2 * n >= end_row)
                {
                    l2fetch_src_size = L2PfParam(istride, width, end_row - i - n, 1);
                }

                L2Fetch(reinterpret_cast<MI_U32>(src.Ptr<MI_U8>(i + n)), l2fetch_src_size);
            }

            MI_U8 *src_data = const_cast<MI_U8 *>(src.Ptr<MI_U8>(i));
            HistogramPernRow(src_data, istride, width, k, hist);
        }
    }

    std::lock_guard<std::mutex> guard(mutex);
    for (MI_S32 i = 0; i < max_size; i++)
    {
        dst[i] += hist[i];
    }

    return Status::OK;
}

template <MI_BOOL IS_MASK>
static Status CalcHistU8HvxHelper(Context *ctx, const Mat &src, MI_U32 *dst, MI_S32 max_size, const Mat &mask)
{
    Status ret = Status::ERROR;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
        return ret;
    }

    MI_S32 width  = src.GetSizes().m_width;
    MI_S32 height = src.GetSizes().m_height;
    MI_S32 stride = src.GetStrides().m_width;

    Mat src_align, mask_align;
    if (stride % 128)
    {
        MI_S32 align_width  = width / 128 * 128;
        MI_S32 align_height = width * height / align_width;
        src_align  = Mat(ctx, ElemType::U8, Sizes3(align_height, align_width, 1), src.GetBuffer());
        mask_align = Mat(ctx, ElemType::U8, Sizes3(align_height, align_width, 1), mask.GetBuffer());
    }
    else
    {
        src_align = src;
        mask_align = mask;
    }

    ThreadBuffer thread_buffer(ctx, max_size * sizeof(MI_U32));
    std::mutex mutex;

    ret = wp->ParallelFor((MI_S32)0, src_align.GetSizes().m_height, CalcHistU8HvxImpl<IS_MASK>, ctx, src_align,
                          dst, mask_align, max_size, std::ref(thread_buffer), std::ref(mutex));

    if (stride % 128)
    {
        MI_S32 begin    = src_align.GetTotalBytes();
        MI_S32 end      = src.GetTotalBytes();
        MI_U8 *src_data = (MI_U8 *)src.GetData();

        if (IS_MASK)
        {
            MI_U8 *mask_data = (MI_U8 *)mask.GetData();
            for (MI_S32 x = begin; x < end; x++)
            {
                dst[src_data[x]] = mask_data[x] ? dst[src_data[x]] + 1 : dst[src_data[x]];
            }
        }
        else
        {
            for (MI_S32 x = begin; x < end; x++)
            {
                dst[src_data[x]]++;
            }
        }
    }

    AURA_RETURN(ctx, ret);
}

static Status CalcHistU8Hvx(Context *ctx, const Mat &src, std::vector<MI_U32> &dst, MI_S32 hist_size,
                            const Scalar &ranges, const Mat &mask, MI_BOOL accumulate)
{
    Status ret = Status::ERROR;

    if (dst.size() < (MI_U32)hist_size)
    {
        dst.resize(hist_size);
    }

    if (src.GetElemType() != ElemType::U8)
    {
        AURA_ADD_ERROR_STRING(ctx, "src element type must be u8");
        return ret;
    }

    if (src.GetSizes().m_channel != 1)
    {
        AURA_ADD_ERROR_STRING(ctx, "hist hvx only support channel 1");
        return ret;
    }

    MI_U32 max_size  = 256;
    MI_S32 hist_low  = ranges.m_val[0];
    MI_S32 hist_high = ranges.m_val[1];

    MI_U32 *hist_local = static_cast<MI_U32*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, max_size * sizeof(MI_U32), 0));
    if (MI_NULL == hist_local)
    {
        AURA_ADD_ERROR_STRING(ctx, "AURA_ALLOC_PARAM fail");
        return ret;
    }

    memset(hist_local, 0, max_size * sizeof(MI_U32));

    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
        AURA_FREE(ctx, hist_local);
        return Status::ERROR;
    }

    if (mask.IsValid())
    {
        ret = CalcHistU8HvxHelper<MI_TRUE>(ctx, src, hist_local, max_size, mask);
    }
    else
    {
        ret = CalcHistU8HvxHelper<MI_FALSE>(ctx, src, hist_local, max_size, mask);
    }

    if (MI_FALSE == accumulate)
    {
        std::fill(dst.begin(), dst.end(), 0);
    }

    MI_F64 scale = (MI_F64)(hist_size) / (hist_high - hist_low);
    MI_S32 j = 0;
    for (MI_S32 i = hist_low; i < hist_high; i++)
    {
        MI_S32 idx = Floor(j * scale);
        dst[idx] += hist_local[i];
        j++;
    }

    AURA_FREE(ctx, hist_local);

    AURA_RETURN(ctx, ret);
}

CalcHistHvx::CalcHistHvx(Context *ctx, const OpTarget &target) : CalcHistImpl(ctx, target)
{}

Status CalcHistHvx::SetArgs(const Array *src, MI_S32 channel, std::vector<MI_U32> &hist, MI_S32 hist_size,
                            const Scalar &ranges, const Array *mask, MI_BOOL accumulate)
{
    if (CalcHistImpl::SetArgs(src, channel, hist, hist_size, ranges, mask, accumulate) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "CalcHistImpl::SetArgs failed");
        return Status::ERROR;
    }

    if (src->GetArrayType() != ArrayType::MAT)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src must be mat type");
        return Status::ERROR;
    }

    if (src->GetElemType() != ElemType::U8)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src element type must be u8");
        return Status::ERROR;
    }

    if (src->GetSizes().m_channel != 1)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "hist hvx only support channel 1");
        return Status::ERROR;
    }

    if (0 != (src->GetStrides().m_width & 0x7F) && src->GetSizes().m_width != src->GetStrides().m_width)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src must be 128 aligned or width must be equal to stride");
        return Status::ERROR;
    }

    if (mask->IsValid())
    {
        if (mask->GetStrides() != src->GetStrides())
        {
            AURA_ADD_ERROR_STRING(m_ctx, "mask stride must be equal to src stride");
            return Status::ERROR;
        }
    }

    return Status::OK;
}

Status CalcHistHvx::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    const Mat *mask = dynamic_cast<const Mat*>(m_mask);

    if ((MI_NULL == src) || (MI_NULL == mask))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or mask is nullptr");
        return Status::ERROR;
    }

    Status ret = Status::OK;

    switch (src->GetElemType())
    {
        case ElemType::U8:
        {
            ret = CalcHistU8Hvx(m_ctx, *src, *m_hist, m_hist_size, m_ranges, *mask, m_accumulate);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "CalcHistU8Hvx failed");
            }
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

std::string CalcHistHvx::ToString() const
{
    return CalcHistImpl::ToString();
}

Status CalcHistRpc(Context *ctx, HexagonRpcParam &rpc_param)
{
    Mat src;
    Mat mask;
    MI_S32 channel;
    MI_S32 hist_size;
    Scalar ranges;
    std::vector<MI_U32> hist;
    MI_BOOL accumulate;

    CalcHistInParam in_param(ctx, rpc_param);
    Status ret = in_param.Get(src, channel, hist, hist_size, ranges, mask, accumulate);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get failed");
        return Status::ERROR;
    }

    CalcHist calchist(ctx, OpTarget::Hvx());

    if (OpCall(ctx, calchist, &src, channel, hist, hist_size, ranges, &mask, accumulate) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "OpCall failed");
        return Status::ERROR;
    }

    CalcHistOutParam out_param(ctx, rpc_param);
    ret = out_param.Set(hist);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "out_param set failed");
        return Status::ERROR;
    }

    return Status::OK;
}

AURA_HEXAGON_RPC_FUNC_REGISTER(AURA_OPS_HIST_PACKAGE_NAME, AURA_OPS_HIST_CALCHIST_OP_NAME, CalcHistRpc);

} // namespace aura