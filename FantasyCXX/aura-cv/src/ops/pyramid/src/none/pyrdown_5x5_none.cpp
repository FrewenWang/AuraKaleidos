#include "pyrdown_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

template <typename Tp, BorderType BORDER_TYPE, typename Kt>
static AURA_VOID PyrDown5x5Row(const Tp *src_p1, const Tp *src_p0, const Tp *src_c, const Tp *src_n0,
                             const Tp *src_n1, MI_S32 iwidth, Tp *dst, MI_S32 owidth, const Kt *kernel)
{
    using SumType = typename Promote<Kt>::Type;

    MI_S32 sx = 0;
    MI_S32 src_idx_l1 = 0, src_idx_l0 = 0, src_idx_c = 0, src_idx_n0 = 0, src_idx_n1 = 0;
    SumType sum[5] = {0}, result = 0;
    for (MI_S32 dx = 0; dx < owidth; dx++)
    {
        sx = dx << 1;
        src_idx_l1 = GetBorderIdx<BORDER_TYPE>(sx - 2, iwidth);
        src_idx_l0 = GetBorderIdx<BORDER_TYPE>(sx - 1, iwidth);
        src_idx_c  = sx;
        src_idx_n0 = GetBorderIdx<BORDER_TYPE>(sx + 1, iwidth);
        src_idx_n1 = GetBorderIdx<BORDER_TYPE>(sx + 2, iwidth);

        // vertical
        sum[0] = (src_p1[src_idx_l1] + src_n1[src_idx_l1]) * kernel[0] + (src_p0[src_idx_l1] + src_n0[src_idx_l1]) * kernel[1] +
                  src_c[src_idx_l1] * kernel[2];
        sum[1] = (src_p1[src_idx_l0] + src_n1[src_idx_l0]) * kernel[0] + (src_p0[src_idx_l0] + src_n0[src_idx_l0]) * kernel[1] +
                  src_c[src_idx_l0] * kernel[2];
        sum[2] = (src_p1[src_idx_c] + src_n1[src_idx_c]) * kernel[0] + (src_p0[src_idx_c] + src_n0[src_idx_c]) * kernel[1] +
                  src_c[src_idx_c] * kernel[2];
        sum[3] = (src_p1[src_idx_n0] + src_n1[src_idx_n0]) * kernel[0] + (src_p0[src_idx_n0] + src_n0[src_idx_n0]) * kernel[1] +
                  src_c[src_idx_n0] * kernel[2];
        sum[4] = (src_p1[src_idx_n1] + src_n1[src_idx_n1]) * kernel[0] + (src_p0[src_idx_n1] + src_n0[src_idx_n1]) * kernel[1] +
                  src_c[src_idx_n1] * kernel[2];

        // horizon
        result = (sum[0] + sum[4]) * kernel[0] + (sum[1] + sum[3]) * kernel[1] + sum[2] * kernel[2];

        dst[dx] = ShiftSatCast<SumType, Tp, PyrDownTraits<Tp>::Q << 1>(result);
    }
}

template <typename Tp, BorderType BORDER_TYPE>
static Status PyrDown5x5NoneImpl(Context *ctx, const Mat &src, Mat &dst, const Mat &kmat, MI_S32 start_row, MI_S32 end_row)
{
    AURA_UNUSED(ctx);

    using Kt = typename PyrDownTraits<Tp>::KernelType;
    const Kt *kernel = kmat.Ptr<Kt>(0);

    const MI_S32 iwidth = src.GetSizes().m_width;
    const MI_S32 owidth = dst.GetSizes().m_width;
    MI_S32 sy = 0;

    const Tp *src_p1  = src.Ptr<Tp, BORDER_TYPE>(start_row - 2, MI_NULL);
    const Tp *src_p0  = src.Ptr<Tp, BORDER_TYPE>(start_row - 1, MI_NULL);
    const Tp *src_c   = src.Ptr<Tp>(start_row);
    const Tp *src_n0  = src.Ptr<Tp, BORDER_TYPE>(start_row + 1, MI_NULL);
    const Tp *src_n1  = src.Ptr<Tp, BORDER_TYPE>(start_row + 2, MI_NULL);

    for (MI_S32 dy = start_row; dy < end_row; dy++)
    {
        sy          = dy << 1;
        Tp *dst_row = dst.Ptr<Tp>(dy);

        PyrDown5x5Row<Tp, BORDER_TYPE, Kt>(src_p1, src_p0, src_c, src_n0, src_n1, iwidth, dst_row, owidth, kernel);

        src_p1 = src_c;
        src_p0 = src_n0;
        src_c  = src_n1;
        src_n0 = src.Ptr<Tp, BORDER_TYPE>(sy + 3, MI_NULL);
        src_n1 = src.Ptr<Tp, BORDER_TYPE>(sy + 4, MI_NULL);
    }

    return Status::OK;
}

template <typename Tp>
static Status PyrDown5x5NoneHelper(Context *ctx, const Mat &src, Mat &dst, const Mat &kmat,
                                   BorderType &border_type, const OpTarget &target)
{
    Status ret = Status::ERROR;
    MI_S32 oheight = dst.GetSizes().m_height;

    switch (border_type)
    {
        case BorderType::REPLICATE:
        {
            if (target.m_data.none.enable_mt)
            {
                WorkerPool *wp = ctx->GetWorkerPool();
                if (MI_NULL == wp)
                {
                    AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
                    return Status::ERROR;
                }

                ret = wp->ParallelFor(static_cast<MI_S32>(0), oheight,
                                      PyrDown5x5NoneImpl<Tp, BorderType::REPLICATE>, ctx,
                                      std::cref(src), std::ref(dst), std::cref(kmat));
            }
            else
            {
                ret = PyrDown5x5NoneImpl<Tp, BorderType::REPLICATE>(ctx, src, dst, kmat, 0, oheight);
            }
            break;
        }

        case BorderType::REFLECT_101:
        {
            if (target.m_data.none.enable_mt)
            {
                WorkerPool *wp = ctx->GetWorkerPool();
                if (MI_NULL == wp)
                {
                    AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
                    return Status::ERROR;
                }

                ret = wp->ParallelFor(static_cast<MI_S32>(0), oheight,
                                      PyrDown5x5NoneImpl<Tp, BorderType::REFLECT_101>, ctx,
                                      std::cref(src), std::ref(dst), std::cref(kmat));
            }
            else
            {
                ret = PyrDown5x5NoneImpl<Tp, BorderType::REFLECT_101>(ctx, src, dst, kmat, 0, oheight);
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "Unsupport border_type");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

Status PyrDown5x5None(Context *ctx, const Mat &src, Mat &dst, const Mat &kmat, BorderType border_type, const OpTarget &target)
{
    Status ret = Status::ERROR;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = PyrDown5x5NoneHelper<MI_U8>(ctx, src, dst, kmat, border_type, target);
            break;
        }

        case ElemType::U16:
        {
            ret = PyrDown5x5NoneHelper<MI_U16>(ctx, src, dst, kmat, border_type, target);
            break;
        }

        case ElemType::S16:
        {
            ret = PyrDown5x5NoneHelper<MI_S16>(ctx, src, dst, kmat, border_type, target);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "Unsupported source format");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura
