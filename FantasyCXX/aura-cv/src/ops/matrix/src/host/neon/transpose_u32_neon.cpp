#include "transpose_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"

namespace aura
{

Status TransposeU32Neon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    AURA_UNUSED(target);
    Status ret = Status::ERROR;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return Status::ERROR;
    }

    MI_S32 channel = src.GetSizes().m_channel;

    switch (channel)
    {
        case 1:
        {
            TransposeNoneFunctor<MI_U32, 1> op;
            ret = wp->ParallelFor(0, dst.GetSizes().m_height, op, std::cref(src), std::ref(dst));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "TransposeU32C1 failed.");
            }
            break;
        }
        case 2:
        {
            TransposeNoneFunctor<MI_U32, 2> op;
            ret = wp->ParallelFor(0, dst.GetSizes().m_height, op, std::cref(src), std::ref(dst));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "TransposeU32C2 failed.");
            }
            break;
        }
        case 3:
        {
            TransposeNoneFunctor<MI_U32, 3> op;
            ret = wp->ParallelFor(0, dst.GetSizes().m_height, op, std::cref(src), std::ref(dst));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "TransposeU32C3 failed.");
            }
            break;
        }
        case 4:
        {
            TransposeNoneFunctor<MI_U32, 4> op;
            ret = wp->ParallelFor(0, dst.GetSizes().m_height, op, std::cref(src), std::ref(dst));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "TransposeU32C4 failed.");
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

} // namespace aura
