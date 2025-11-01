#include "pyrup_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"
namespace aura
{

template <typename Tp, BorderType BORDER_TYPE, typename Kt>
static AURA_VOID PyrUp5x5TwoRow(const Tp *src_p, const Tp *src_c, const Tp *src_n,
                              MI_S32 iwidth, Tp *dst_c0, Tp *dst_c1, const Kt *kernel)
{
    using SumType = typename Promote<Kt>::Type;

    MI_S32 dx = 0;
    MI_S32 src_idx_l = 0, src_idx_c = 0, src_idx_n = 0;
    SumType e_sum[3] = {0}, o_sum[3] = {0}, result[4] = {0};
    for (MI_S32 sx = 0; sx < iwidth; sx++)
    {
        dx = sx << 1;
        src_idx_l = GetBorderIdx<BORDER_TYPE>(sx - 1, iwidth);
        src_idx_c = sx;
        src_idx_n = GetBorderIdx<BorderType::REPLICATE>(sx + 1, iwidth);

        // vertical
        e_sum[0] = (src_p[src_idx_l] + src_n[src_idx_l]) * kernel[0] + src_c[src_idx_l] * kernel[2];
        e_sum[1] = (src_p[src_idx_c] + src_n[src_idx_c]) * kernel[0] + src_c[src_idx_c] * kernel[2];
        e_sum[2] = (src_p[src_idx_n] + src_n[src_idx_n]) * kernel[0] + src_c[src_idx_n] * kernel[2];

        o_sum[0] = (src_c[src_idx_l] + src_n[src_idx_l]) * kernel[1];
        o_sum[1] = (src_c[src_idx_c] + src_n[src_idx_c]) * kernel[1];
        o_sum[2] = (src_c[src_idx_n] + src_n[src_idx_n]) * kernel[1];

        // horizon
        result[0] = (e_sum[0] + e_sum[2]) * kernel[0] + e_sum[1] * kernel[2];
        result[1] = (e_sum[1] + e_sum[2]) * kernel[1];
        result[2] = (o_sum[0] + o_sum[2]) * kernel[0] + o_sum[1] * kernel[2];
        result[3] = (o_sum[1] + o_sum[2]) * kernel[1];

        dst_c0[dx]     = ShiftSatCast<SumType, Tp, PyrUpTraits<Tp>::Q << 1>(result[0]);
        dst_c0[dx + 1] = ShiftSatCast<SumType, Tp, PyrUpTraits<Tp>::Q << 1>(result[1]);
        dst_c1[dx]     = ShiftSatCast<SumType, Tp, PyrUpTraits<Tp>::Q << 1>(result[2]);
        dst_c1[dx + 1] = ShiftSatCast<SumType, Tp, PyrUpTraits<Tp>::Q << 1>(result[3]);
    }
}

template <typename Tp, BorderType BORDER_TYPE>
static Status PyrUp5x5NoneImpl(Context *ctx, const Mat &src, Mat &dst, const Mat &kmat, MI_S32 start_row, MI_S32 end_row)
{
    AURA_UNUSED(ctx);

    using Kt = typename PyrUpTraits<Tp>::KernelType;
    const Kt *kernel = kmat.Ptr<Kt>(0);

    const MI_S32 iwidth  = src.GetSizes().m_width;
    MI_S32 dy = 0;

    const Tp *src_p = src.Ptr<Tp, BORDER_TYPE>(start_row - 1, MI_NULL);
    const Tp *src_c = src.Ptr<Tp>(start_row);
    const Tp *src_n = src.Ptr<Tp, BorderType::REPLICATE>(start_row + 1, MI_NULL);

    for (MI_S32 sy = start_row; sy < end_row; sy++)
    {
        dy = sy << 1;

        Tp *dst_c0 = dst.Ptr<Tp>(dy);
        Tp *dst_c1 = dst.Ptr<Tp>(dy + 1);

        PyrUp5x5TwoRow<Tp, BORDER_TYPE, Kt>(src_p, src_c, src_n, iwidth, dst_c0, dst_c1, kernel);

        src_p = src_c;
        src_c = src_n;
        src_n = src.Ptr<Tp, BorderType::REPLICATE>(sy + 2, MI_NULL);
    }

    return Status::OK;
}


template <typename Tp>
static Status PyrUp5x5NoneHelper(Context *ctx, const Mat &src, Mat &dst, const Mat &kmat,
                                 BorderType &border_type, const OpTarget &target)
{
    Status ret = Status::ERROR;
    MI_S32 iheight = src.GetSizes().m_height;

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

                ret = wp->ParallelFor(static_cast<MI_S32>(0), iheight,
                                      PyrUp5x5NoneImpl<Tp, BorderType::REPLICATE>, ctx,
                                      std::cref(src), std::ref(dst), std::cref(kmat));
            }
            else
            {
                ret = PyrUp5x5NoneImpl<Tp, BorderType::REPLICATE>(ctx, src, dst, kmat, 0, iheight);
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

                ret = wp->ParallelFor(static_cast<MI_S32>(0), iheight,
                                      PyrUp5x5NoneImpl<Tp, BorderType::REFLECT_101>, ctx,
                                      std::cref(src), std::ref(dst), std::cref(kmat));
            }
            else
            {
                ret = PyrUp5x5NoneImpl<Tp, BorderType::REFLECT_101>(ctx, src, dst, kmat, 0, iheight);
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

Status PyrUp5x5None(Context *ctx, const Mat &src, Mat &dst, const Mat &kmat, BorderType border_type, const OpTarget &target)
{
    Status ret = Status::ERROR;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = PyrUp5x5NoneHelper<MI_U8>(ctx, src, dst, kmat, border_type, target);
            break;
        }

        case ElemType::U16:
        {
            ret = PyrUp5x5NoneHelper<MI_U16>(ctx, src, dst, kmat, border_type, target);
            break;
        }

        case ElemType::S16:
        {
            ret = PyrUp5x5NoneHelper<MI_S16>(ctx, src, dst, kmat, border_type, target);
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
