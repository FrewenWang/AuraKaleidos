#include "split_impl.hpp"
#include "aura/ops/core.h"
#include "aura/runtime/mat.h"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

template <typename Tp>
AURA_INLINE DT_VOID SplitRow(DT_S32 width, const Tp *src_row, DT_S32 src_ch, const std::vector<DT_S32> &ch_offsets,
                             const std::vector<Tp*> &dst_rows, const std::vector<DT_S32> &ochannels)
{
    DT_S32 dst_count = dst_rows.size();

    for (DT_S32 x = 0; x < width; ++x)
    {
        for (DT_S32 n = 0; n < dst_count; ++n)
        {
            const Tp *src_ptr = src_row + x * src_ch + ch_offsets[n];
            Tp *dst_ptr       = dst_rows[n] + x * ochannels[n];

            for (DT_S32 ch = 0; ch < ochannels[n]; ++ch)
            {
                dst_ptr[ch] = src_ptr[ch];
            }
        }
    }
}

template <typename Tp>
static Status SplitNoneImpl(const Mat &src, std::vector<Mat*> &dst, const std::vector<DT_S32> &ochannels,
                            const std::vector<DT_S32> &ch_offsets, DT_S32 dst_count, DT_S32 start_row, DT_S32 end_row)
{
    Sizes3 src_sz = src.GetSizes();

    const DT_S32 width    = src_sz.m_width;
    const DT_S32 ichannel = src_sz.m_channel;

    std::vector<Tp*> dst_rows(dst_count, DT_NULL);

    for (DT_S32 y = start_row; y < end_row; ++y)
    {
        const Tp *src_ptr = src.Ptr<Tp>(y);

        for (DT_S32 n = 0; n < dst_count; ++n)
        {
            dst_rows[n] = dst[n]->Ptr<Tp>(y);
        }

        SplitRow(width, src_ptr, ichannel, ch_offsets, dst_rows, ochannels);
    }

    return Status::OK;
}

template <typename Tp>
static Status SplitNoneHelper(Context *ctx, const Mat &src, std::vector<Mat*> &dst, OpTarget &target)
{
    DT_S32 dst_count = dst.size();

    if (1 == dst_count)
    {
        return src.CopyTo(*(dst[0]));
    }

    std::vector<DT_S32> ochannels;

    for (DT_S32 n = 0; n < dst_count; ++n)
    {
        ochannels.emplace_back(dst[n]->GetSizes().m_channel);
    }

    std::vector<DT_S32> ch_offsets(dst_count, 0);

    for (DT_S32 n = 1; n < dst_count; ++n)
    {
        ch_offsets[n] = ch_offsets[n - 1] + ochannels[n - 1];
    }

    Status ret = Status::ERROR;

    DT_S32 height = src.GetSizes().m_height;

    if (target.m_data.none.enable_mt)
    {
        WorkerPool *wp = ctx->GetWorkerPool();
        if (DT_NULL == wp)
        {
            AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
            return Status::ERROR;
        }

        ret = wp->ParallelFor(static_cast<DT_S32>(0), height, SplitNoneImpl<Tp>, src, dst,
                              ochannels, ch_offsets, dst_count);
    }
    else
    {
        ret = SplitNoneImpl<Tp>(src, dst, ochannels, ch_offsets, dst_count, static_cast<DT_S32>(0), height);
    }

    AURA_RETURN(ctx, ret);
}

SplitNone::SplitNone(Context *ctx, const OpTarget &target) : SplitImpl(ctx, target)
{}

Status SplitNone::SetArgs(const Array *src, const std::vector<Array*> &dst)
{
    if (SplitImpl::SetArgs(src, dst) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SplitImpl::SetArgs failed");
        return Status::ERROR;
    }

    if (src->GetArrayType() != ArrayType::MAT)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src must be mat type");
        return Status::ERROR;
    }

    for (auto& mat:dst)
    {
        if (mat->GetArrayType() != ArrayType::MAT)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "dst must be mat type");
            return Status::ERROR;
        }
    }

    return Status::OK;
}

Status SplitNone::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    std::vector<Mat*> dst;
    dst.reserve(m_dst.size());
    for (auto& mat:m_dst)
    {
        Mat *dst_mat = dynamic_cast<Mat*>(mat);
        if (DT_NULL == dst_mat)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "dynamic_cast dst failed.");
            return Status::ERROR;
        }
        dst.push_back(dst_mat);
    }

    if (DT_NULL == src)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src is null");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    switch (src->GetElemType())
    {
        case ElemType::U8:
        case ElemType::S8:
        {
            ret = SplitNoneHelper<DT_U8>(m_ctx, *src, dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "SplitNoneHelper<DT_U8> failed.");
            }
            break;
        }

        case ElemType::U16:
        case ElemType::S16:
#if defined(AURA_BUILD_HOST)
        case ElemType::F16:
#endif
        {
            ret = SplitNoneHelper<DT_U16>(m_ctx, *src, dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "SplitNoneHelper<DT_U16> failed.");
            }
            break;
        }

        case ElemType::U32:
        case ElemType::S32:
#if defined(AURA_BUILD_HOST)
        case ElemType::F32:
#endif
        {
            ret = SplitNoneHelper<DT_U32>(m_ctx, *src, dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "SplitNoneHelper<DT_U32> failed.");
            }
            break;
        }

        default:
        {
            ret = Status::ERROR;
            AURA_ADD_ERROR_STRING(m_ctx, "SplitNone with invalid ElemType.");
            break;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura