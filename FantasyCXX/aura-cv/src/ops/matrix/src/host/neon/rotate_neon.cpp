#include "rotate_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"

namespace aura
{

template <typename Tp, RotateType Rt, DT_S32 C> struct RotateBorderNoneFunctor;

template <typename Tp, DT_S32 C>
struct RotateBorderNoneFunctor<Tp, RotateType::ROTATE_90, C>
{
    using BLOCK = struct { Tp val[C]; };

    DT_VOID operator()(const Mat &src, Mat &dst, DT_S32 start_blk, DT_S32 end_blk)
    {
        DT_S32 width  = src.GetSizes().m_width;
        DT_S32 height = src.GetSizes().m_height;

        DT_S32 y = Max(0, start_blk);
        DT_S32 h = Min(height, end_blk);

        for (; y < h; y++)
        {
            DT_S32 dx = height - y - 1;
            const BLOCK *src_row = src.Ptr<BLOCK>(y);
            for (DT_S32 x = 0; x < width; x++)
            {
                BLOCK *dst_row = dst.Ptr<BLOCK>(x);
                dst_row[dx] = src_row[x];
            }
        }
    }
};

template <typename Tp, DT_S32 C>
struct RotateBorderNoneFunctor<Tp, RotateType::ROTATE_180, C>
{
    using BLOCK = struct { Tp val[C]; };

    DT_VOID operator()(const Mat &src, Mat &dst, DT_S32 start_blk, DT_S32 end_blk)
    {
        DT_S32 width  = src.GetSizes().m_width;
        DT_S32 height = src.GetSizes().m_height;

        DT_S32 y = Max(0, start_blk);
        DT_S32 h = Min(height, end_blk);

        for (; y < h; y++)
        {
            const BLOCK *src_row = src.Ptr<BLOCK>(y);
            BLOCK *dst_row = dst.Ptr<BLOCK>(height - y - 1);
            DT_S32 x_src = 0;
            DT_S32 x_dst = width - 1;

            for (DT_S32 x = 0; x < width; x++)
            {
                dst_row[x_dst--] = src_row[x_src++];
            }
        }
    }
};

template <typename Tp, DT_S32 C>
struct RotateBorderNoneFunctor<Tp, RotateType::ROTATE_270, C>
{
    using BLOCK = struct { Tp val[C]; };

    DT_VOID operator()(const Mat &src, Mat &dst, DT_S32 start_blk, DT_S32 end_blk)
    {
        DT_S32 width  = src.GetSizes().m_width;
        DT_S32 height = src.GetSizes().m_height;

        DT_S32 y = Max(0, start_blk);
        DT_S32 h = Min(height, end_blk);

        for (; y < h; y++)
        {
            const BLOCK *src_row = src.Ptr<BLOCK>(y);
            for (DT_S32 x = 0; x < width; x++)
            {
                BLOCK *dst_row = dst.Ptr<BLOCK>(width - x - 1);
                dst_row[y] = src_row[x];
            }
        }
    }
};

template <typename Tp, RotateType Rt>
AURA_INLINE Status RotateCommNeonHelper(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    AURA_UNUSED(target);
    Status ret = Status::ERROR;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return Status::ERROR;
    }

    DT_S32 height  = dst.GetSizes().m_height;
    DT_S32 channel = dst.GetSizes().m_channel;

    switch (channel)
    {
        case 1:
        {
            RotateNoneFunctor<Tp, Rt, 1> op;
            ret = wp->ParallelFor(0, AURA_ALIGN(height, op.BLOCK_SIZE) / op.BLOCK_SIZE,
                                  op, std::cref(src), std::ref(dst));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "RotateNoneFunctor with C1 failed.");
            }
            break;
        }
        case 2:
        {
            RotateNoneFunctor<Tp, Rt, 2> op;
            ret = wp->ParallelFor(0, AURA_ALIGN(height, op.BLOCK_SIZE) / op.BLOCK_SIZE,
                                  op, std::cref(src), std::ref(dst));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "RotateNoneFunctor with C2 failed.");
            }
            break;
        }
        case 3:
        {
            RotateNoneFunctor<Tp, Rt, 3> op;
            ret = wp->ParallelFor(0, AURA_ALIGN(height, op.BLOCK_SIZE) / op.BLOCK_SIZE,
                                  op, std::cref(src), std::ref(dst));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "RotateNoneFunctor with C3 failed.");
            }
            break;
        }
        case 4:
        {
            RotateNoneFunctor<Tp, Rt, 4> op;
            ret = wp->ParallelFor(0, AURA_ALIGN(height, op.BLOCK_SIZE) / op.BLOCK_SIZE,
                                  op, std::cref(src), std::ref(dst));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "RotateNoneFunctor with C4 failed.");
            }
            break;
        }
        default:
        {
            ret = Status::ERROR;
            AURA_ADD_ERROR_STRING(ctx, "channel should be <= 4");
            break;
        }
    }

    AURA_RETURN(ctx, ret);
}

template <typename Tp, RotateType Rt, DT_S32 C>
AURA_INLINE Status RotateNeonImpl(const Mat &src, Mat &dst, DT_S32 start_blk, DT_S32 end_blk)
{
    using Functor = RotateNeonFunctor<Tp, Rt, C>;
    using SType = typename Functor::SType;
    constexpr DT_S32 BLOCK_SIZE = Functor::BLOCK_SIZE;
    Functor op;

    SType *src_data = (SType *)src.GetData();
    SType *dst_data = (SType *)dst.GetData();

    DT_U32 src_step = src.GetRowPitch() / sizeof(SType);
    DT_U32 dst_step = dst.GetRowPitch() / sizeof(SType);

    DT_S32 w = src.GetSizes().m_width;
    DT_S32 h = src.GetSizes().m_height;

    start_blk = start_blk * BLOCK_SIZE;
    end_blk   = Min(end_blk * BLOCK_SIZE, h);

    DT_S32 x_align = (w & (-BLOCK_SIZE));
    DT_S32 y_align = (end_blk & (-BLOCK_SIZE));

    DT_S32 x = 0;
    DT_S32 y = Max(0, start_blk);

    for (; y < y_align; y += BLOCK_SIZE)
    {
        x = 0;
        for (; x < x_align; x += BLOCK_SIZE)
        {
            op(src_data, dst_data, src_step, dst_step, w, h, x, y);
        }
        if (x < w)
        {
            x = w - BLOCK_SIZE;
            op(src_data, dst_data, src_step, dst_step, w, h, x, y);
        }
    }
    if (y < end_blk)
    {
        RotateBorderNoneFunctor<Tp, Rt, C> op_none;
        op_none(src, dst, y, end_blk);
    }
    return Status::OK;
}

template <typename Tp, RotateType Rt>
struct RotateParallelNeonFunctor
{
    Status operator()(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
    {
        Status ret = RotateCommNeonHelper<Tp, Rt>(ctx, src, dst, target);
        AURA_RETURN(ctx, ret);
    }
};

template <RotateType Rt>
struct RotateParallelNeonFunctor<DT_U8, Rt>
{
    using Tp = DT_U8;
    Status operator()(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
    {
        Status ret = Status::ERROR;

        WorkerPool *wp = ctx->GetWorkerPool();
        if (DT_NULL == wp)
        {
            AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
            return Status::ERROR;
        }

        DT_S32 height  = src.GetSizes().m_height;
        DT_S32 channel = src.GetSizes().m_channel;

        switch (channel)
        {
            case 1:
            {
                RotateNeonFunctor<Tp, Rt, 1> op;
                ret = wp->ParallelFor(0, AURA_ALIGN(height, op.BLOCK_SIZE) / op.BLOCK_SIZE,
                                      RotateNeonImpl<Tp, Rt, 1>, std::cref(src), std::ref(dst));
                if (ret != Status::OK)
                {
                    AURA_ADD_ERROR_STRING(ctx, "RotateNeon U8C1 failed.");
                }
                break;
            }
            case 2:
            {
                RotateNeonFunctor<Tp, Rt, 2> op;
                ret = wp->ParallelFor(0, AURA_ALIGN(height, op.BLOCK_SIZE) / op.BLOCK_SIZE,
                                      RotateNeonImpl<Tp, Rt, 2>, std::cref(src), std::ref(dst));
                if (ret != Status::OK)
                {
                    AURA_ADD_ERROR_STRING(ctx, "RotateNeon U8C2 failed.");
                }
                break;
            }
            case 3:
            {
                RotateNeonFunctor<Tp, Rt, 3> op;
                ret = wp->ParallelFor(0, AURA_ALIGN(height, op.BLOCK_SIZE) / op.BLOCK_SIZE,
                                      RotateNeonImpl<Tp, Rt, 3>, std::cref(src), std::ref(dst));
                if (ret != Status::OK)
                {
                    AURA_ADD_ERROR_STRING(ctx, "RotateNeon U8C3 failed.");
                }
                break;
            }
            case 4:
            {
                RotateNeonFunctor<Tp, Rt, 4> op;
                ret = wp->ParallelFor(0, AURA_ALIGN(height, op.BLOCK_SIZE) / op.BLOCK_SIZE,
                                      RotateNeonImpl<Tp, Rt, 4>, std::cref(src), std::ref(dst));
                if (ret != Status::OK)
                {
                    AURA_ADD_ERROR_STRING(ctx, "RotateNeon U8C4 failed.");
                }
                break;
            }
            default:
            {
                ret = RotateCommNeonHelper<Tp, Rt>(ctx, src, dst, target);
                if (ret != Status::OK)
                {
                    AURA_ADD_ERROR_STRING(ctx, "Rotate with Elem8 failed.");
                }
                break;
            }
        }

        AURA_RETURN(ctx, ret);
    }
};

template <RotateType Rt>
struct RotateParallelNeonFunctor<DT_U16, Rt>
{
    using Tp = DT_U16;
    Status operator()(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
    {
        Status ret = Status::ERROR;

        WorkerPool *wp = ctx->GetWorkerPool();
        if (DT_NULL == wp)
        {
            AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
            return Status::ERROR;
        }

        DT_S32 height  = src.GetSizes().m_height;
        DT_S32 channel = src.GetSizes().m_channel;

        switch (channel)
        {
            case 1:
            {
                // process U16C1 as U8C2
                RotateNeonFunctor<DT_U8, Rt, 2> op;
                ret = wp->ParallelFor(0, AURA_ALIGN(height, op.BLOCK_SIZE) / op.BLOCK_SIZE,
                                      RotateNeonImpl<DT_U8, Rt, 2>, std::cref(src), std::ref(dst));
                if (ret != Status::OK)
                {
                    AURA_ADD_ERROR_STRING(ctx, "RotateNeon U16C1 failed.");
                }
                break;
            }
            case 2:
            {
                // process U16C2 as U8C4
                RotateNeonFunctor<DT_U8, Rt, 4> op;
                ret = wp->ParallelFor(0, AURA_ALIGN(height, op.BLOCK_SIZE) / op.BLOCK_SIZE,
                                      RotateNeonImpl<DT_U8, Rt, 4>, std::cref(src), std::ref(dst));
                if (ret != Status::OK)
                {
                    AURA_ADD_ERROR_STRING(ctx, "RotateNeon U16C1 failed.");
                }
                break;
            }
            default:
            {
                ret = RotateCommNeonHelper<Tp, Rt>(ctx, src, dst, target);
                if (ret != Status::OK)
                {
                    AURA_ADD_ERROR_STRING(ctx, "Rotate with Elem16 failed.");
                }
                break;
            }
        }

        AURA_RETURN(ctx, ret);
    }
};

template <RotateType Rt>
struct RotateParallelNeonFunctor<DT_U32, Rt>
{
    using Tp = DT_U32;
    Status operator()(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
    {
        Status ret = Status::ERROR;

        WorkerPool *wp = ctx->GetWorkerPool();
        if (DT_NULL == wp)
        {
            AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
            return Status::ERROR;
        }

        DT_S32 height  = src.GetSizes().m_height;
        DT_S32 channel = src.GetSizes().m_channel;

        switch (channel)
        {
            case 1:
            {
                // process U32C1 as U8C4
                ret = wp->ParallelFor(0, height,
                                      RotateNeonImpl<DT_U8, Rt, 4>, std::cref(src), std::ref(dst));
                if (ret != Status::OK)
                {
                    AURA_ADD_ERROR_STRING(ctx, "RotateNeon U32C1 failed.");
                }
                break;
            }
            default:
            {
                ret = RotateCommNeonHelper<Tp, Rt>(ctx, src, dst, target);
                if (ret != Status::OK)
                {
                    AURA_ADD_ERROR_STRING(ctx, "Rotate with Elem32 failed.");
                }
                break;
            }
        }

        AURA_RETURN(ctx, ret);
    }
};

template <typename Tp>
AURA_INLINE Status RotateNeonHelper(Context *ctx, const Mat &src, Mat &dst, RotateType type, const OpTarget &target)
{
    Status ret = Status::ERROR;

    switch (type)
    {
        case RotateType::ROTATE_90:
        {
            ret = RotateParallelNeonFunctor<Tp, RotateType::ROTATE_90>()(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "RotateNeon with ROTATE_90 failed.");
            }
            break;
        }
        case RotateType::ROTATE_180:
        {
            ret = RotateParallelNeonFunctor<Tp, RotateType::ROTATE_180>()(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "RotateNeon with ROTATE_180 failed.");
            }
            break;
        }
        case RotateType::ROTATE_270:
        {
            ret = RotateParallelNeonFunctor<Tp, RotateType::ROTATE_270>()(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "RotateNeon with ROTATE_270 failed.");
            }
            break;
        }
        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "RotateNeon call with invalid RotateType.");
            ret = Status::ERROR;
            break;
        }
    }

    AURA_RETURN(ctx, ret);
}

RotateNeon::RotateNeon(Context *ctx, const OpTarget &target) : RotateImpl(ctx, target)
{}

Status RotateNeon::SetArgs(const Array *src, Array *dst, RotateType type)
{
    if (RotateImpl::SetArgs(src, dst, type) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "RotateImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status RotateNeon::Run()
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
            ret = RotateNeonHelper<DT_U8>(m_ctx, *src, *dst, m_type, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "RotateNeon Elem8 failed.");
            }
            break;
        }
        case ElemType::U16:
        case ElemType::S16:
#if defined(AURA_ENABLE_NEON_FP16)
        case ElemType::F16:
#endif
        {
            ret = RotateNeonHelper<DT_U16>(m_ctx, *src, *dst, m_type, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "RotateNeon Elem16 failed.");
            }
            break;
        }
        case ElemType::U32:
        case ElemType::S32:
        case ElemType::F32:
        {
            ret = RotateNeonHelper<DT_U32>(m_ctx, *src, *dst, m_type, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "RotateNeon Elem32 failed.");
            }
            break;
        }
        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "RotateNeon call with invalid ElemType.");
            ret = Status::ERROR;
            break;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura
