#include "transpose_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

#define TRN_ROW_PTR(T, data, row, stride)  reinterpret_cast<T*>(reinterpret_cast<DT_SPTR_T>(data) + (row) * (stride))

namespace aura
{

template <typename Tp>
static DT_VOID TransposeBlock(const Tp *src, Tp *dst, DT_S32 block_size_x, DT_S32 block_size_y, DT_S32 channel, DT_S32 ipitch, DT_S32 opitch)
{
    for (DT_S32 y = 0; y < block_size_y; ++y)
    {
        Tp *dst_row = TRN_ROW_PTR(Tp, dst, y, opitch);

        for (DT_S32 x = 0; x < block_size_x; ++x)
        {
            const Tp *src_row = TRN_ROW_PTR(Tp, src, x, ipitch);

            for (DT_S32 ch = 0; ch < channel; ++ch)
            {
                dst_row[x * channel + ch] = src_row[y * channel + ch];
            }
        }
    }
}

template <typename Tp>
static DT_VOID TransposeCommNoneImpl(const Mat &src, Mat &dst)
{
    Sizes3 src_sz = src.GetSizes();
    Sizes3 dst_sz = dst.GetSizes();

    const DT_S32 block_size = 8;
    const DT_S32 channel = src_sz.m_channel;

    const DT_S32 blk_count_x = (dst_sz.m_width  + block_size - 1) / block_size;
    const DT_S32 blk_count_y = (dst_sz.m_height + block_size - 1) / block_size;

    const DT_S32 src_pitch = src.GetRowPitch();
    const DT_S32 dst_pitch = dst.GetRowPitch();

    for (DT_S32 by = 0; by < blk_count_y; ++by)
    {
        DT_S32 block_size_y = (blk_count_y - 1 == by) ? (dst_sz.m_height - block_size * (blk_count_y - 1)) : block_size;
        DT_S32 bx = 0;
        for (; bx < blk_count_x - 1; ++bx)
        {
            const Tp *src_c = &src.At<Tp>(bx * block_size, by * block_size, 0);
            Tp *dst_c       = &dst.At<Tp>(by * block_size, bx * block_size, 0);
            TransposeBlock(src_c, dst_c, block_size, block_size_y, channel, src_pitch, dst_pitch);
        }

        const Tp *src_c   = &src.At<Tp>(bx * block_size, by * block_size, 0);
        Tp *dst_c         = &dst.At<Tp>(by * block_size, bx * block_size, 0);
        DT_S32 block_size_x = dst_sz.m_width - bx * block_size;
        TransposeBlock(src_c, dst_c, block_size_x, block_size_y, channel, src_pitch, dst_pitch);
    }
}

template <typename Tp>
static Status TransposeNoneHelper(Context *ctx, const Mat &src, Mat &dst, OpTarget &target)
{
    Status ret = Status::OK;
    DT_S32 height = dst.GetSizes().m_height;

#define TRANSPOSE_NONE_FUNCTOR(channel)                                                                                 \
    TransposeNoneFunctor<Tp, channel> op;                                                                               \
    if (target.m_data.none.enable_mt)                                                                                   \
    {                                                                                                                   \
        WorkerPool *wp = ctx->GetWorkerPool();                                                                          \
        if (DT_NULL == wp)                                                                                              \
        {                                                                                                               \
            AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");                                                         \
            return Status::ERROR;                                                                                       \
        }                                                                                                               \
        ret = wp->ParallelFor(static_cast<DT_S32>(0), AURA_ALIGN(height, 16) / 16, op, std::cref(src), std::ref(dst));  \
    }                                                                                                                   \
    else                                                                                                                \
    {                                                                                                                   \
        ret = op(src, dst, 0, AURA_ALIGN(height, 16) / 16);                                                             \
    }                                                                                                                   \
                                                                                                                        \
    if (ret != Status::OK)                                                                                              \
    {                                                                                                                   \
        DT_CHAR error_msg[128];                                                                                         \
        std::snprintf(error_msg, sizeof(error_msg), "TransposeNoneFunctor channel:%s failed", #channel);                \
        AURA_ADD_ERROR_STRING(ctx, error_msg);                                                                          \
    }

    switch (dst.GetSizes().m_channel)
    {
        case 1:
        {
            TRANSPOSE_NONE_FUNCTOR(1);
            break;
        }
        case 2:
        {
            TRANSPOSE_NONE_FUNCTOR(2);
            break;
        }
        case 3:
        {
            TRANSPOSE_NONE_FUNCTOR(3);
            break;
        }
        case 4:
        {
            TRANSPOSE_NONE_FUNCTOR(4);
            break;
        }
        default:
        {
            TransposeCommNoneImpl<Tp>(src, dst);
            break;
        }
    }

    AURA_RETURN(ctx, ret);
}

TransposeNone::TransposeNone(Context *ctx, const OpTarget &target) : TransposeImpl(ctx, target)
{}

Status TransposeNone::SetArgs(const Array *src, Array *dst)
{
    if (TransposeImpl::SetArgs(src, dst) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "TransposeImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status TransposeNone::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst       = dynamic_cast<Mat*>(m_dst);

    if ((DT_NULL == src) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is null");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    switch (src->GetElemType())
    {
        case ElemType::U8:
        case ElemType::S8:
        {
            ret = TransposeNoneHelper<DT_U8>(m_ctx, *src, *dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "TransposeNone Elem8 failed.");
            }
            break;
        }
        case ElemType::U16:
        case ElemType::S16:
#if defined(AURA_BUILD_HOST)
        case ElemType::F16:
#endif
        {
            ret = TransposeNoneHelper<DT_U16>(m_ctx, *src, *dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "TransposeNone Elem16 failed.");
            }
            break;
        }
        case ElemType::U32:
        case ElemType::S32:
#if defined(AURA_BUILD_HOST)
        case ElemType::F32:
#endif
        {
            ret = TransposeNoneHelper<DT_U32>(m_ctx, *src, *dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "TransposeNone Elem32 failed.");
            }
            break;
        }
        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "TransposeNone with invalid ElemType.");
            ret = Status::ERROR;
            break;
        }
    }

    AURA_RETURN(m_ctx, ret);
}
} // namespace aura