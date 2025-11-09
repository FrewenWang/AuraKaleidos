#include "flip_impl.hpp"

#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

static Status FlipVerticalInplaceNeonImpl(const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    const DT_S32 width     = src.GetSizes().m_width;
    const DT_S32 height    = src.GetSizes().m_height;
    const DT_S32 channel   = src.GetSizes().m_channel;
    const DT_S32 row_bytes = ElemTypeSize(src.GetElemType()) * channel * width;

    for (DT_S32 y = start_row; y < end_row; ++y)
    {
        const DT_U8 *src_top = src.Ptr<DT_U8>(y);
        const DT_U8 *src_bot = src.Ptr<DT_U8>(height - 1 - y);

        DT_U8 *dst_top = dst.Ptr<DT_U8>(y);
        DT_U8 *dst_bot = dst.Ptr<DT_U8>(height - 1 - y);

        DT_S32 x = 0;

        for (; x <= row_bytes - 16; x += 16)
        {
            const DT_S32 *src_top_cur = reinterpret_cast<const DT_S32 *>(src_top + x);
            const DT_S32 *src_bot_cur = reinterpret_cast<const DT_S32 *>(src_bot + x);

            DT_S32 *dst_top_cur = reinterpret_cast<DT_S32 *>(dst_top + x);
            DT_S32 *dst_bot_cur = reinterpret_cast<DT_S32 *>(dst_bot + x);

            int32x4_t vqs32_x0 = neon::vload1q(src_top_cur);
            int32x4_t vqs32_x1 = neon::vload1q(src_bot_cur);

            neon::vstore(dst_bot_cur, vqs32_x0);
            neon::vstore(dst_top_cur, vqs32_x1);
        }

        for (; x <= row_bytes - 4; x += 4)
        {
            DT_S32 v0 = (reinterpret_cast<const DT_S32 *>(src_top + x))[0];
            DT_S32 v1 = (reinterpret_cast<const DT_S32 *>(src_bot + x))[0];

            (reinterpret_cast<DT_S32 *>(dst_top + x))[0] = v1;
            (reinterpret_cast<DT_S32 *>(dst_bot + x))[0] = v0;
        }

        for (; x < row_bytes; ++x)
        {
            DT_U8 v0 = src_top[x];
            DT_U8 v1 = src_bot[x];

            dst_top[x] = v1;
            dst_bot[x] = v0;
        }
    }

    return Status::OK;
}

static Status FlipVerticalRowCopy(const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    const DT_S32 width     = src.GetSizes().m_width;
    const DT_S32 height    = src.GetSizes().m_height;
    const DT_S32 channel   = src.GetSizes().m_channel;
    const DT_S32 row_bytes = ElemTypeSize(src.GetElemType()) * channel * width;

    for (DT_S32 y = start_row; y < end_row; ++y)
    {
        memcpy(dst.Ptr<DT_U8>(height - y - 1), src.Ptr<DT_U8>(y), row_bytes);
    }

    return Status::OK;
}

static Status FlipVerticalNeonHelper(const Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    AURA_UNUSED(target);

    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return Status::ERROR;
    }

    Status ret = Status::OK;

    DT_S32 height  = src.GetSizes().m_height;

    if (src.GetData() == dst.GetData())
    {
        ret = wp->ParallelFor(0, height / 2, FlipVerticalInplaceNeonImpl, std::cref(src), std::ref(dst));
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "ParallelFor run failed");
            return Status::ERROR;
        }
    }
    else
    {
        ret = wp->ParallelFor(0, height, FlipVerticalRowCopy, std::cref(src), std::ref(dst));
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "ParallelFor run failed");
            return Status::ERROR;
        }
    }

    return Status::OK;
}

template <typename Tp, DT_S32 C>
static Status FlipHorizonalInplaceNeonImpl(const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    using MVType = typename neon::MQVector<Tp, C>::MVType;

    constexpr DT_S32  VEC_LEN = static_cast<DT_S32>(16 / sizeof(Tp));
    constexpr DT_S32 MVEC_LEN = VEC_LEN * C;

    DT_S32 width    = src.GetSizes().m_width;
    DT_S32 row_half  = ((width + 1) >> 1) * C;
    DT_S32 row_total = width * C;

    struct FlipVec
    {
        Tp val[C];
    };

    for (DT_S32 y = start_row; y < end_row; ++y)
    {
        Tp *src_row = const_cast<Tp*>(src.Ptr<Tp>(y));
        Tp *dst_row = dst.Ptr<Tp>(y);

        DT_S32 x = 0;
        for (; x <= row_half - MVEC_LEN; x += MVEC_LEN)
        {
            DT_S32 idx_left  = x;
            DT_S32 idx_right = row_total - MVEC_LEN - x;

            MVType vsrc_left, vsrc_right;
            neon::vload(src_row + idx_left,  vsrc_left);
            neon::vload(src_row + idx_right, vsrc_right);

            for (DT_S32 i = 0; i < C; i++)
            {
                // reverse elements in every half vector
                vsrc_left.val[i]  = neon::vrev64(vsrc_left.val[i]);
                vsrc_right.val[i] = neon::vrev64(vsrc_right.val[i]);
                // exchange high and low half vectors
                vsrc_left.val[i]  = neon::vcombine(neon::vgethigh(vsrc_left.val[i]),  neon::vgetlow(vsrc_left.val[i]));
                vsrc_right.val[i] = neon::vcombine(neon::vgethigh(vsrc_right.val[i]), neon::vgetlow(vsrc_right.val[i]));
            }

            neon::vstore(dst_row + idx_right, vsrc_left);
            neon::vstore(dst_row + idx_left,  vsrc_right);
        }

        for (; x < row_half; x += C)
        {
            DT_S32 idx_left  = x;
            DT_S32 idx_right = row_total - C - x;

            FlipVec src_left  = *((FlipVec*)(src_row + idx_left));
            FlipVec src_right = *((FlipVec*)(src_row + idx_right));

            *((FlipVec*)(dst_row + idx_right)) = src_left;
            *((FlipVec*)(dst_row + idx_left))  = src_right;
        }
    }

    return Status::OK;
}

static Status FlipHorizonalNeonHelper(const Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    AURA_UNUSED(target);

    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    DT_S32 height = src.GetSizes().m_height;
    DT_S32 channel_bytes = src.GetSizes().m_channel * ElemTypeSize(src.GetElemType());
    switch (channel_bytes)
    {
        case 1:
        {
            ret = wp->ParallelFor(0, height, FlipHorizonalInplaceNeonImpl<DT_U8,  1>, src, dst);
            break;
        }
        case 2:
        {
            ret = wp->ParallelFor(0, height, FlipHorizonalInplaceNeonImpl<DT_U16, 1>, src, dst);
            break;
        }
        case 3:
        {
            ret = wp->ParallelFor(0, height, FlipHorizonalInplaceNeonImpl<DT_U8,  3>, src, dst);
            break;
        }
        case 4:
        {
            ret = wp->ParallelFor(0, height, FlipHorizonalInplaceNeonImpl<DT_U32, 1>, src, dst);
            break;
        }
        case 6:
        {
            ret = wp->ParallelFor(0, height, FlipHorizonalInplaceNeonImpl<DT_U16, 3>, src, dst);
            break;
        }
        case 8:
        {
            ret = wp->ParallelFor(0, height, FlipHorizonalInplaceNeonImpl<DT_U32, 2>, src, dst);
            break;
        }
        case 12:
        {
            ret = wp->ParallelFor(0, height, FlipHorizonalInplaceNeonImpl<DT_U32, 3>, src, dst);
            break;
        }
        case 16:
        {
            ret = wp->ParallelFor(0, height, FlipHorizonalInplaceNeonImpl<DT_U32, 4>, src, dst);
            break;
        }
        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "FlipHorizonalNeonHelper with unsupported channel&element type.");
            break;
        }
    }

    AURA_RETURN(ctx, ret);
}

FlipNeon::FlipNeon(Context *ctx, const OpTarget &target) : FlipImpl(ctx, target)
{}

Status FlipNeon::SetArgs(const Array *src, Array *dst, FlipType type)
{
    if (FlipImpl::SetArgs(src, dst, type) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "FlipImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    if (src->GetSizes().m_channel > 4)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "FlipNeon with channel > 4 is not supported.");
        return Status::ERROR;
    }

    return Status::OK;
}

Status FlipNeon::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst       = dynamic_cast<Mat*>(m_dst);

    if ((DT_NULL == src) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is null");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;
    
    switch (m_type)
    {
        case FlipType::VERTICAL:
        {
            ret = FlipVerticalNeonHelper(m_ctx, *src, *dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "FlipNeon(Vertical) failed.");
            }
            break;
        }

        case FlipType::HORIZONTAL:
        {
            ret = FlipHorizonalNeonHelper(m_ctx, *src, *dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "FlipNeon(Horizonal) failed.");
            }
            break;
        }

        case FlipType::BOTH:
        {
            ret  = FlipVerticalNeonHelper(m_ctx, *src, *dst, m_target);
            ret |= FlipHorizonalNeonHelper(m_ctx, *dst, *dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "FlipNeon(Both) failed.");
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "FlipNeonHelper with unsupported FlipType.");
        }
    }

    AURA_RETURN(m_ctx, ret);
}

}  // namespace aura