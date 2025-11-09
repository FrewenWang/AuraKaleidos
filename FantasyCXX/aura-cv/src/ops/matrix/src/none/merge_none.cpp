#include "merge_impl.hpp"
#include "aura/ops/core.h"
#include "aura/runtime/mat.h"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

template <typename Tp>
AURA_INLINE DT_VOID MergeRow(DT_S32 width, const std::vector<const Tp*> &src_rows, const std::vector<DT_S32> &src_chs,
                             Tp *dst_row, DT_S32 dst_ch, const std::vector<DT_S32> &dst_ch_offsets)
{
    DT_S32 src_count = src_rows.size();

    for (DT_S32 x = 0; x < width; ++x)
    {
        for (DT_S32 n = 0; n < src_count; ++n)
        {
            const Tp *src_ptr = src_rows[n] + x * src_chs[n];
            Tp *dst_ptr       = dst_row + x * dst_ch + dst_ch_offsets[n];

            for (DT_S32 ch = 0; ch < src_chs[n]; ++ch)
            {
                dst_ptr[ch] = src_ptr[ch];
            }
        }
    }
}

template <typename Tp>
static Status MergeNoneImpl(Context *ctx, std::vector<const Array*> &src, Mat &dst, const std::vector<DT_S32> &src_chs,
                            const std::vector<DT_S32> &dst_ch_offsets, DT_S32 src_count, DT_S32 start_row, DT_S32 end_row)
{
    std::vector<const Tp*> src_rows(src_count, DT_NULL);

    const Sizes3 dst_sz = dst.GetSizes();
    const DT_S32 width  = dst_sz.m_width;
    const DT_S32 dst_ch = dst_sz.m_channel;

    for (DT_S32 y = start_row; y < end_row; ++y)
    {
        for (DT_S32 n = 0; n < src_count; ++n)
        {
            const Mat *src_mat = dynamic_cast<const Mat*>(src[n]);
            if (NULL == src_mat)
            {
                AURA_ADD_ERROR_STRING(ctx, "dynamic_cast src[n] failed.");
                return Status::ERROR;
            }
            src_rows[n] = src_mat->Ptr<Tp>(y);
        }

        Tp *dst_row = dst.Ptr<Tp>(y);

        MergeRow(width, src_rows, src_chs, dst_row, dst_ch, dst_ch_offsets);
    }

    return Status::OK;
}

template<typename Tp>
static Status MergeNoneHelper(Context *ctx, std::vector<const Array*> &src, Mat &dst, OpTarget &target)
{
    DT_S32 src_count = src.size();

    if (1 == src_count)
    {
        const Mat *src_mat = dynamic_cast<const Mat*>(src[0]);
        if (NULL == src_mat)
        {
            AURA_ADD_ERROR_STRING(ctx, "dynamic_cast src[0] failed.");
            return Status::ERROR;
        }

        return src_mat->CopyTo(dst);
    }


    std::vector<DT_S32> src_chs;

    for (DT_S32 n = 0; n < src_count; ++n)
    {
        src_chs.emplace_back(src[n]->GetSizes().m_channel);
    }

    std::vector<DT_S32> dst_ch_offsets(src_count, 0);
    for (DT_S32 n = 1; n < src_count; ++n)
    {
        dst_ch_offsets[n] = dst_ch_offsets[n - 1] + src_chs[n - 1];
    }

    Status ret = Status::ERROR;

    const DT_S32 height = dst.GetSizes().m_height;

    if (target.m_data.none.enable_mt)
    {
        WorkerPool *wp = ctx->GetWorkerPool();
        if (DT_NULL == wp)
        {
            AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
            return Status::ERROR;
        }

        ret = wp->ParallelFor(static_cast<DT_S32>(0), height, MergeNoneImpl<Tp>, ctx, src, dst,
                              src_chs, dst_ch_offsets, src_count);
    }
    else
    {
        ret = MergeNoneImpl<Tp>(ctx, src, dst, src_chs, dst_ch_offsets, src_count, static_cast<DT_S32>(0), height);
    }

    AURA_RETURN(ctx, ret);
}

MergeNone::MergeNone(Context *ctx, const OpTarget &target) : MergeImpl(ctx, target)
{}

Status MergeNone::SetArgs(const std::vector<const Array*> &src, Array *dst)
{
    if (MergeImpl::SetArgs(src, dst) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "MergeImpl::SetArgs failed");
        return Status::ERROR;
    }

    if (dst->GetArrayType() != ArrayType::MAT)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "dst must be mat type");
        return Status::ERROR;
    }

    for (auto &mat : src)
    {
        if (mat->GetArrayType() != ArrayType::MAT)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "src must be mat type");
            return Status::ERROR;
        }
    }

    return Status::OK;
}

Status MergeNone::Run()
{
    Mat *dst = dynamic_cast<Mat*>(m_dst);

    if (DT_NULL == dst)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "dst is null");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    switch (dst->GetElemType())
    {
        case ElemType::U8:
        case ElemType::S8:
        {
            ret = MergeNoneHelper<DT_U8>(m_ctx, m_src, *dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "MergeNoneHelper<DT_U8> failed.");
            }
            break;
        }

        case ElemType::U16:
        case ElemType::S16:
#if defined(AURA_BUILD_HOST)
        case ElemType::F16:
#endif
        {
            ret = MergeNoneHelper<DT_U16>(m_ctx, m_src, *dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "MergeNoneHelper<DT_U16> failed.");
            }
            break;
        }

        case ElemType::U32:
        case ElemType::S32:
#if defined(AURA_BUILD_HOST)
        case ElemType::F32:
#endif
        {
            ret = MergeNoneHelper<DT_U32>(m_ctx, m_src, *dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "MergeNoneHelper<DT_U32> failed.");
            }
            break;
        }

        default:
        {
            ret = Status::ERROR;
            AURA_ADD_ERROR_STRING(m_ctx, "MergeNone with invalid ElemType.");
            break;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura