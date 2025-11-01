#include "flip_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

template <typename Tp>
static Status FlipVerticalInplaceImpl(const Mat &src, Mat &dst, MI_S32 start_row, MI_S32 end_row)
{
    const Sizes3 sz        = src.GetSizes();
    const MI_S32 row_bytes = sizeof(Tp) * sz.m_channel * sz.m_width;

    for (MI_S32 dy = start_row; dy < end_row; ++dy)
    {
        const MI_U8 *src_top = src.Ptr<MI_U8>(dy);
        const MI_U8 *src_bot = src.Ptr<MI_U8>(sz.m_height - 1 - dy);

        MI_U8 *dst_top = dst.Ptr<MI_U8>(dy);
        MI_U8 *dst_bot = dst.Ptr<MI_U8>(sz.m_height - 1 - dy);

        MI_S32 x = 0;

        if (!(reinterpret_cast<MI_UPTR_T>(src_top) & 0x3) && !(reinterpret_cast<MI_UPTR_T>(src_bot) & 0x3) &&
            !(reinterpret_cast<MI_UPTR_T>(dst_top) & 0x3) && !(reinterpret_cast<MI_UPTR_T>(dst_bot) & 0x3))
        {
            for (; x < row_bytes - 16; x += 16)
            {
                const MI_S32 *src_top_cur = reinterpret_cast<const MI_S32*>(src_top + x);
                const MI_S32 *src_bot_cur = reinterpret_cast<const MI_S32*>(src_bot + x);

                MI_S32 *dst_top_cur = reinterpret_cast<MI_S32*>(dst_top + x);
                MI_S32 *dst_bot_cur = reinterpret_cast<MI_S32*>(dst_bot + x);

                MI_S32 v0 = src_top_cur[0];
                MI_S32 v1 = src_bot_cur[0];

                dst_top_cur[0] = v1;
                dst_bot_cur[0] = v0;

                v0 = src_top_cur[1];
                v1 = src_bot_cur[1];

                dst_top_cur[1] = v1;
                dst_bot_cur[1] = v0;

                v0 = src_top_cur[2];
                v1 = src_bot_cur[2];

                dst_top_cur[2] = v1;
                dst_bot_cur[2] = v0;

                v0 = src_top_cur[3];
                v1 = src_bot_cur[3];

                dst_top_cur[3] = v1;
                dst_bot_cur[3] = v0;
            }

            for (; x <= row_bytes - 4; x +=4)
            {
                MI_S32 v0 = reinterpret_cast<const MI_S32*>(src_top + x)[0];
                MI_S32 v1 = reinterpret_cast<const MI_S32*>(src_bot + x)[0];

                reinterpret_cast<MI_S32*>(dst_top + x)[0] = v1;
                reinterpret_cast<MI_S32*>(dst_bot + x)[0] = v0;
            }
        }

        for (; x < row_bytes; ++x)
        {
            MI_U8 v0 = src_top[x];
            MI_U8 v1 = src_bot[x];

            dst_top[x] = v1;
            dst_bot[x] = v0;
        }
    }

    return Status::OK;
}

template <typename Tp>
static Status FlipVerticalCopyImpl(const Mat &src, Mat &dst, MI_S32 start_row, MI_S32 end_row)
{
    const Sizes3 sz = src.GetSizes();
    const MI_S32 row_bytes = sizeof(Tp) * sz.m_channel * sz.m_width;

    for (MI_S32 y = start_row; y < end_row; ++y)
    {
        memcpy(dst.Ptr<Tp>(sz.m_height - y - 1), src.Ptr<Tp>(y), row_bytes);
    }

    return Status::OK;
}

template <typename Tp>
static Status FlipVerticalNoneImpl(Context *ctx, const Mat &src, Mat &dst, OpTarget &target)
{
    Status ret = Status::ERROR;

    const MI_S32 height = src.GetSizes().m_height;

    if (src.GetData() == dst.GetData())
    {
        if (target.m_data.none.enable_mt)
        {
            WorkerPool *wp = ctx->GetWorkerPool();
            if (MI_NULL == wp)
            {
                AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
                return Status::ERROR;
            }

            ret = wp->ParallelFor(static_cast<MI_S32>(0), (height / 2), FlipVerticalInplaceImpl<Tp>,
                                  std::cref(src), std::ref(dst));
        }
        else
        {
            ret = FlipVerticalInplaceImpl<Tp>(src, dst, static_cast<MI_S32>(0), (height / 2));
        }
    }
    else
    {
        if (target.m_data.none.enable_mt)
        {
            WorkerPool *wp = ctx->GetWorkerPool();
            if (MI_NULL == wp)
            {
                AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
                return Status::ERROR;
            }

            ret = wp->ParallelFor(static_cast<MI_S32>(0), height, FlipVerticalCopyImpl<Tp>, std::cref(src), std::ref(dst));
        }
        else
        {
            ret = FlipVerticalCopyImpl<Tp>(src, dst, static_cast<MI_S32>(0), height);
        }

    }

    AURA_RETURN(ctx, ret);
}

template <typename Tp>
static Status FlipHorizonalNoneCore(const Mat &src, Mat &dst, const std::vector<MI_S32> &idx_table, MI_S32 start_row, MI_S32 end_row)
{
    MI_S32 width   = dst.GetSizes().m_width;
    MI_S32 channel = dst.GetSizes().m_channel;

    for (MI_S32 y = start_row; y < end_row; ++y)
    {
        const Tp *src_row = src.Ptr<Tp>(y);
        Tp *dst_row       = dst.Ptr<Tp>(y);

        for (MI_S32 x = 0; x < (width + 1) / 2 * channel; ++x)
        {
            MI_S32 idx = idx_table[x];

            Tp v0 = src_row[x];
            Tp v1 = src_row[idx];

            dst_row[x]   = v1;
            dst_row[idx] = v0;
        }
    }

    return Status::OK;
}

template <typename Tp>
static Status FlipHorizonalNoneImpl(Context *ctx, const Mat &src, Mat &dst, OpTarget &target)
{
    Status ret = Status::ERROR;

    const Sizes3 sz = src.GetSizes();

    std::vector<MI_S32> idx_table((sz.m_width + 1) / 2 * sz.m_channel, 0);

    for (MI_S32 i = 0; i < (sz.m_width + 1) / 2; ++i)
    {
        for (MI_S32 j = 0; j < sz.m_channel; ++j)
        {
            idx_table[i * sz.m_channel + j] = (sz.m_width - i - 1) * sz.m_channel + j;
        }
    }

    if (target.m_data.none.enable_mt)
    {
        WorkerPool *wp = ctx->GetWorkerPool();
        if (MI_NULL == wp)
        {
            AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
            return Status::ERROR;
        }

        ret = wp->ParallelFor(static_cast<MI_S32>(0), sz.m_height, FlipHorizonalNoneCore<Tp>, std::cref(src),
                              std::ref(dst), std::cref(idx_table));
    }
    else
    {
        ret = FlipHorizonalNoneCore<Tp>(src, dst, idx_table, static_cast<MI_S32>(0), sz.m_height);
    }

    AURA_RETURN(ctx, ret);
}

template <typename Tp>
static Status FlipNoneHelper(Context *ctx, const Mat &src, Mat &dst, FlipType type, OpTarget &target)
{
    Status ret = Status::ERROR;

    switch (type)
    {
        case FlipType::VERTICAL:
        {
            ret = FlipVerticalNoneImpl<Tp>(ctx, src, dst, target);
            break;
        }
        case FlipType::HORIZONTAL:
        {
            ret = FlipHorizonalNoneImpl<Tp>(ctx, src, dst, target);
            break;
        }
        case FlipType::BOTH:
        {
            ret = FlipVerticalNoneImpl<Tp>(ctx, src, dst, target);
            ret |= FlipHorizonalNoneImpl<Tp>(ctx, dst, dst, target);
            break;
        }
        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "FlipNoneHelper with unsupported FlipType");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

FlipNone::FlipNone(Context *ctx, const OpTarget &target) : FlipImpl(ctx, target)
{}

Status FlipNone::SetArgs(const Array *src, Array *dst, FlipType type)
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

    return Status::OK;
}

Status FlipNone::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst       = dynamic_cast<Mat*>(m_dst);

    if ((MI_NULL == src) || (MI_NULL == dst))
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
            ret = FlipNoneHelper<MI_U8>(m_ctx, *src, *dst, m_type, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "FlipNoneHelper<MI_U8> failed.");
            }
            break;
        }
        case ElemType::U16:
        case ElemType::S16:
#if defined(AURA_BUILD_HOST)
        case ElemType::F16:
#endif
        {
            ret = FlipNoneHelper<MI_U16>(m_ctx, *src, *dst, m_type, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "FlipNoneHelper<MI_U16> failed.");
            }
            break;
        }
        case ElemType::U32:
        case ElemType::S32:
#if defined(AURA_BUILD_HOST)
        case ElemType::F32:
#endif
        {
            ret = FlipNoneHelper<MI_U32>(m_ctx, *src, *dst, m_type, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "FlipNoneHelper<MI_U32> failed.");
            }
            break;
        }
        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "FlipNone with invalid ElemType.");
            ret = Status::ERROR;
            break;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura