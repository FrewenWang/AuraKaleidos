#include "split_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

template <typename Tp, MI_S32 C>
static Status SplitNeonImpl(const Mat &src, std::vector<Mat*> &dst, MI_S32 start_row, MI_S32 end_row)
{
    using MVType = typename neon::MQVector<Tp, C>::MVType;
    constexpr MI_S32 ELEM_COUNTS = 16 / sizeof(Tp);

    MI_S32 width       = src.GetSizes().m_width;
    MI_S32 width_align = width & (-ELEM_COUNTS);

    MVType mv_src;
    std::vector<Tp*> dst_rows(C, MI_NULL);

    for (MI_S32 y = start_row; y < end_row; y++)
    {
        const Tp *src_row = src.Ptr<Tp>(y);
        for (MI_S32 c = 0; c < C; c++)
        {
            dst_rows[c] = dst[c]->Ptr<Tp>(y);
        }

        MI_S32 x = 0;
        for (; x < width_align; x += ELEM_COUNTS)
        {
            neon::vload(src_row + x * C, mv_src);
            for (MI_S32 c = 0; c < C; c++)
            {
                neon::vstore(dst_rows[c] + x, mv_src.val[c]);
            }
        }

        for (; x < width; x++)
        {
            for (MI_S32 c = 0; c < C; c++)
            {
                dst_rows[c][x]  = src_row[x * C + c];
            }
        }
    }

    return Status::OK;
}

template <typename Tp>
static Status SplitNeonHelper(Context *ctx, const Mat &src, std::vector<Mat*> &dst)
{
    if (1 == dst.size())
    {
        return src.CopyTo(*(dst[0]));
    }

    Status ret = Status::ERROR;

    MI_S32 height  = src.GetSizes().m_height;
    MI_S32 channel = src.GetSizes().m_channel;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
        return Status::ERROR;
    }

    switch (channel)
    {
        case 1:
        {
            ret = wp->ParallelFor(static_cast<MI_S32>(0), height, SplitNeonImpl<Tp, 1>, src, dst);
            break;
        }
        case 2:
        {
            ret = wp->ParallelFor(static_cast<MI_S32>(0), height, SplitNeonImpl<Tp, 2>, src, dst);
            break;
        }
        case 3:
        {
            ret = wp->ParallelFor(static_cast<MI_S32>(0), height, SplitNeonImpl<Tp, 3>, src, dst);
            break;
        }
        case 4:
        {
            ret = wp->ParallelFor(static_cast<MI_S32>(0), height, SplitNeonImpl<Tp, 4>, src, dst);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "invalid channel");
            break;
        }
    }

    AURA_RETURN(ctx, ret);
}

SplitNeon::SplitNeon(Context *ctx, const OpTarget &target) : SplitImpl(ctx, target)
{}

Status SplitNeon::SetArgs(const Array *src, const std::vector<Array*> &dst)
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

    for (auto &mat : dst)
    {
        if (mat->GetArrayType() != ArrayType::MAT)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "dst must be mat type");
            return Status::ERROR;
        }

        if (mat->GetSizes().m_channel != 1)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "the channel of dst must be equal to 1");
            return Status::ERROR;
        }
    }

    if (src->GetSizes().m_channel != static_cast<MI_S32>(dst.size()))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "the channel of src and dst do not match");
        return Status::ERROR;
    }

    return Status::OK;
}

Status SplitNeon::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    if (MI_NULL == src)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src is null");
        return Status::ERROR;
    }

    std::vector<Mat*> dst;
    dst.reserve(m_dst.size());
    for (auto &mat : m_dst)
    {
        Mat *dst_mat = dynamic_cast<Mat*>(mat);
        if (MI_NULL == dst_mat)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "dst is null");
            return Status::ERROR;
        }
        dst.push_back(dst_mat);
    }

    Status ret = Status::ERROR;

    switch (src->GetElemType())
    {
        case ElemType::U8:
        case ElemType::S8:
        {
            ret = SplitNeonHelper<MI_U8>(m_ctx, *src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "SplitNeonHelper<MI_U8> failed");
            }
            break;
        }

        case ElemType::U16:
        case ElemType::S16:
#if defined(AURA_ENABLE_NEON_FP16)
        case ElemType::F16:
#endif
        {
            ret = SplitNeonHelper<MI_U16>(m_ctx, *src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "SplitNeonHelper<MI_U16> failed");
            }
            break;
        }

        case ElemType::U32:
        case ElemType::S32:
#if defined(AURA_BUILD_HOST)
        case ElemType::F32:
#endif
        {
            ret = SplitNeonHelper<MI_U32>(m_ctx, *src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "SplitNeonHelper<MI_U32> failed");
            }
            break;
        }

        default:
        {
            ret = Status::ERROR;
            AURA_ADD_ERROR_STRING(m_ctx, "invalid ElemType");
            break;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura