#include "guidefilter_impl.hpp"
#include "aura/ops/matrix.h"
#include "aura/ops/resize.h"
#include "aura/ops/filter.h"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/thread_buffer.h"
#include "aura/runtime/core.h"

namespace aura
{

template <typename Tp, BorderType BORDER_TYPE, DT_S32 C>
static Status GuideFilterNormalNeonImpl(Context *ctx, const Mat &src0, const Mat &src1, Mat &dst, DT_S32 ksize,
                                        DT_F32 eps, const Scalar &border_value, const OpTarget &target)
{
    AURA_UNUSED(target);
    Status ret = Status::ERROR;

    using SumType   = typename Promote<Tp>::Type;
    using SqSumType = typename Promote<SumType>::Type;

    Sizes3 sz = src0.GetSizes();

    Mat a = Mat(ctx, ElemType::F32, src0.GetSizes());
    Mat b = Mat(ctx, ElemType::F32, src0.GetSizes());
    if (!a.IsValid() || !b.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "Mat a or b is invalid");
        return Status::ERROR;
    }

    const DT_S32 height = src0.GetSizes().m_height;
    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return Status::ERROR;
    }

    DT_S32 width       = sz.m_width;
    DT_S32 width_align = AURA_ALIGN((width + ksize) * C, 64);

    // 0. prepare buffer
    ThreadBuffer row_a_buffer(ctx,   width_align * sizeof(Tp)      * ksize); // to store data with expanded border
    ThreadBuffer row_b_buffer(ctx,   width_align * sizeof(Tp)      * ksize); // to store data with expanded border
    ThreadBuffer row_aa_buffer(ctx,  width_align * sizeof(SumType) * ksize); // a * a
    ThreadBuffer row_ab_buffer(ctx,  width_align * sizeof(SumType) * ksize); // a * b
    ThreadBuffer sum_a_buffer(ctx,   width_align * sizeof(SumType));         // sum of a
    ThreadBuffer sum_b_buffer(ctx,   width_align * sizeof(SumType));         // sum of b
    ThreadBuffer sum_aa_buffer (ctx, width_align * sizeof(SqSumType));       // sum of a * a
    ThreadBuffer sum_ab_buffer (ctx, width_align * sizeof(SqSumType));       // sum of a * b
    ThreadBuffer arr_ptr_buffer(ctx, 4 * ksize * sizeof(DT_VOID*));

    ret = wp->ParallelFor(0, height, GuideFilterCalcABNeonImpl<Tp, SumType, SqSumType, BORDER_TYPE, C>,
                        std::cref(src0), std::cref(src1), std::ref(a), std::ref(b), std::ref(row_a_buffer),
                        std::ref(row_b_buffer), std::ref(row_aa_buffer), std::ref(row_ab_buffer), std::ref(sum_a_buffer),
                        std::ref(sum_b_buffer), std::ref(sum_aa_buffer), std::ref(sum_ab_buffer),
                        std::ref(arr_ptr_buffer), ksize, eps, border_value);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "ParallelFor GuideFilterCalcABNeonImpl run failed");
        return Status::ERROR;
    }

    Mat mean_a = Mat(ctx, ElemType::F32, src0.GetSizes());
    Mat mean_b = Mat(ctx, ElemType::F32, src0.GetSizes());
    if (!mean_a.IsValid() || !mean_b.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "Mat mean_sub_a or mean_sub_b is invalid");
        return Status::ERROR;
    }

    ret =  IBoxfilter(ctx, a, mean_a, ksize, BORDER_TYPE, border_value, target);
    ret |= IBoxfilter(ctx, b, mean_b, ksize, BORDER_TYPE, border_value, target);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "BoxFilterNeon failed");
        return Status::ERROR;
    }

    ret = wp->ParallelFor(static_cast<DT_S32>(0), mean_a.GetSizes().m_height, GuideFilterLinearTransNeonImpl<Tp, SumType, SqSumType, C>,
                          std::cref(src0), std::cref(mean_a), std::cref(mean_b), std::ref(dst));
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "GuideFilterLinearTransImpl failed");
        return Status::ERROR;
    }

    return Status::OK;
}

template <typename Tp, BorderType BORDER_TYPE, DT_S32 C>
static Status GuideFilterNormalNeonImpl(Context *ctx, const Mat &src, Mat &dst, DT_S32 ksize, DT_F32 eps,
                                        const Scalar &border_value, const OpTarget &target)
{
    AURA_UNUSED(target);
    Status ret = Status::ERROR;

    using SumType   = typename Promote<Tp>::Type;
    using SqSumType = typename Promote<SumType>::Type;

    Sizes3 sz = src.GetSizes();

    Mat a = Mat(ctx, ElemType::F32, src.GetSizes());
    Mat b = Mat(ctx, ElemType::F32, src.GetSizes());
    if (!a.IsValid() || !b.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "Mat a or b is invalid");
        return Status::ERROR;
    }

    const DT_S32 height = src.GetSizes().m_height;
    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return Status::ERROR;
    }

    DT_S32 width       = sz.m_width;
    DT_S32 width_align = AURA_ALIGN((width + ksize) * C, 64);

    // 0. prepare buffer
    ThreadBuffer row_a_buffer(ctx,   width_align * sizeof(Tp)      * ksize); // to store data with expanded border
    ThreadBuffer row_aa_buffer(ctx,  width_align * sizeof(SumType) * ksize); // a * a
    ThreadBuffer sum_a_buffer(ctx,   width_align * sizeof(SumType));         // sum of a
    ThreadBuffer sum_aa_buffer(ctx,  width_align * sizeof(SqSumType));       // sum of a * a
    ThreadBuffer arr_ptr_buffer(ctx, 2 * ksize * sizeof(DT_VOID*));

    ret = wp->ParallelFor(0, height, GuideFilterCalcABSameSrcNeonImpl<Tp, SumType, SqSumType, BORDER_TYPE, C>,
                          std::cref(src), std::ref(a), std::ref(b), std::ref(row_a_buffer),
                          std::ref(row_aa_buffer), std::ref(sum_a_buffer), std::ref(sum_aa_buffer),
                          std::ref(arr_ptr_buffer), ksize, eps, border_value);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "ParallelFor GuideFilterCalcABNeonImpl run failed");
        return Status::ERROR;
    }

    Mat mean_a = Mat(ctx, ElemType::F32, src.GetSizes());
    Mat mean_b = Mat(ctx, ElemType::F32, src.GetSizes());
    if (!mean_a.IsValid() || !mean_b.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "Mat mean_sub_a or mean_sub_b is invalid");
        return Status::ERROR;
    }

    ret =  IBoxfilter(ctx, a, mean_a, ksize, BORDER_TYPE, border_value, target);
    ret |= IBoxfilter(ctx, b, mean_b, ksize, BORDER_TYPE, border_value, target);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "BoxFilterNeon failed");
        return Status::ERROR;
    }

    ret = wp->ParallelFor(static_cast<DT_S32>(0), mean_a.GetSizes().m_height, GuideFilterLinearTransNeonImpl<Tp, SumType, SqSumType, C>,
                          std::cref(src), std::cref(mean_a), std::cref(mean_b), std::ref(dst));
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "GuideFilterLinearTransImpl failed");
        return Status::ERROR;
    }

    return Status::OK;
}

template <typename Tp, BorderType BORDER_TYPE>
Status GuideFilterNormalNeonHelper(Context *ctx, const Mat &src, Mat &dst, DT_S32 ksize, DT_F32 eps,
                                   const Scalar &border_value, const OpTarget &target)
{
    Status ret = Status::ERROR;

    switch (src.GetSizes().m_channel)
    {
        case 1:
        {
            ret = GuideFilterNormalNeonImpl<Tp, BORDER_TYPE, 1>(ctx, src, dst, ksize, eps, border_value, target);
            break;
        }
        case 2:
        {
            ret = GuideFilterNormalNeonImpl<Tp, BORDER_TYPE, 2>(ctx, src, dst, ksize, eps, border_value, target);
            break;
        }
        case 3:
        {
            ret = GuideFilterNormalNeonImpl<Tp, BORDER_TYPE, 3>(ctx, src, dst, ksize, eps, border_value, target);
            break;
        }
        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "channel is only suppose 1/2/3");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

template <typename Tp, BorderType BORDER_TYPE>
Status GuideFilterNormalNeonHelper(Context *ctx, const Mat &src0, const Mat &src1, Mat &dst, DT_S32 ksize, DT_F32 eps,
                                   const Scalar &border_value, const OpTarget &target)
{
    Status ret = Status::ERROR;

    switch (src0.GetSizes().m_channel)
    {
        case 1:
        {
            ret = GuideFilterNormalNeonImpl<Tp, BORDER_TYPE, 1>(ctx, src0, src1, dst, ksize, eps, border_value, target);
            break;
        }
        case 2:
        {
            ret = GuideFilterNormalNeonImpl<Tp, BORDER_TYPE, 2>(ctx, src0, src1, dst, ksize, eps, border_value, target);
            break;
        }
        case 3:
        {
            ret = GuideFilterNormalNeonImpl<Tp, BORDER_TYPE, 3>(ctx, src0, src1, dst, ksize, eps, border_value, target);
            break;
        }
        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "channel is only suppose 1/2/3");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

template <typename Tp>
Status GuideFilterNormalNeonHelper(Context *ctx, const Mat &src, Mat &dst, DT_S32 ksize, DT_F32 eps,
                                   BorderType border_type, const Scalar &border_value, const OpTarget &target)
{
    Status ret = Status::ERROR;

    switch (border_type)
    {
        case BorderType::CONSTANT:
        {
            ret = GuideFilterNormalNeonHelper<Tp, BorderType::CONSTANT>(ctx, src, dst, ksize, eps, border_value, target);
            break;
        }
        case BorderType::REPLICATE:
        {
            ret = GuideFilterNormalNeonHelper<Tp, BorderType::REPLICATE>(ctx, src, dst, ksize, eps, border_value, target);
            break;
        }
        case BorderType::REFLECT_101:
        {
            ret = GuideFilterNormalNeonHelper<Tp, BorderType::REFLECT_101>(ctx, src, dst, ksize, eps, border_value, target);
            break;
        }
        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "border_type is not supported");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

template <typename Tp>
Status GuideFilterNormalNeonHelper(Context *ctx, const Mat &src0, const Mat &src1, Mat &dst, DT_S32 ksize, DT_F32 eps,
                                   BorderType border_type, const Scalar &border_value, const OpTarget &target)
{
    Status ret = Status::ERROR;

    switch (border_type)
    {
        case BorderType::CONSTANT:
        {
            ret = GuideFilterNormalNeonHelper<Tp, BorderType::CONSTANT>(ctx, src0, src1, dst, ksize, eps, border_value, target);
            break;
        }
        case BorderType::REPLICATE:
        {
            ret = GuideFilterNormalNeonHelper<Tp, BorderType::REPLICATE>(ctx, src0, src1, dst, ksize, eps, border_value, target);
            break;
        }
        case BorderType::REFLECT_101:
        {
            ret = GuideFilterNormalNeonHelper<Tp, BorderType::REFLECT_101>(ctx, src0, src1, dst, ksize, eps, border_value, target);
            break;
        }
        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "border_type is not supported");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

Status GuideFilterNormalNeon(Context *ctx, const Mat &src0, const Mat &src1, Mat &dst, const DT_S32 &ksize, const DT_F32 &eps,
                             BorderType &border_type, const Scalar &border_value, const OpTarget &target)
{
    Status ret = Status::ERROR;

    if (src0.GetData() == src1.GetData())
    {
        switch (src0.GetElemType())
        {
            case ElemType::U8:
            {
                ret = GuideFilterNormalNeonHelper<DT_U8>(ctx, src0, dst, ksize, eps, border_type, border_value, target);
                break;
            }

            case ElemType::S8:
            {
                ret = GuideFilterNormalNeonHelper<DT_S8>(ctx, src0, dst, ksize, eps, border_type, border_value, target);
                break;
            }

            case ElemType::U16:
            {
                ret = GuideFilterNormalNeonHelper<DT_U16>(ctx, src0, dst, ksize, eps, border_type, border_value, target);
                break;
            }

            case ElemType::S16:
            {
                ret = GuideFilterNormalNeonHelper<DT_S16>(ctx, src0, dst, ksize, eps, border_type, border_value, target);
                break;
            }
#if defined(AURA_ENABLE_NEON_FP16)
            case ElemType::F16:
            {
                ret = GuideFilterNormalNeonHelper<MI_F16>(ctx, src0, dst, ksize, eps, border_type, border_value, target);
                break;
            }
#endif // AURA_ENABLE_NEON_FP16

            case ElemType::F32:
            {
                ret = GuideFilterNormalNeonHelper<DT_F32>(ctx, src0, dst, ksize, eps, border_type, border_value, target);
                break;
            }

            default:
            {
                AURA_ADD_ERROR_STRING(ctx, "Unsupported source format");
                return Status::ERROR;
            }
        }
    }
    else
    {
        switch (src0.GetElemType())
        {
            case ElemType::U8:
            {
                ret = GuideFilterNormalNeonHelper<DT_U8>(ctx, src0, src1, dst, ksize, eps, border_type, border_value, target);
                break;
            }

            case ElemType::S8:
            {
                ret = GuideFilterNormalNeonHelper<DT_S8>(ctx, src0, src1, dst, ksize, eps, border_type, border_value, target);
                break;
            }

            case ElemType::U16:
            {
                ret = GuideFilterNormalNeonHelper<DT_U16>(ctx, src0, src1, dst, ksize, eps, border_type, border_value, target);
                break;
            }

            case ElemType::S16:
            {
                ret = GuideFilterNormalNeonHelper<DT_S16>(ctx, src0, src1, dst, ksize, eps, border_type, border_value, target);
                break;
            }

#if defined(AURA_ENABLE_NEON_FP16)
            case ElemType::F16:
            {
                ret = GuideFilterNormalNeonHelper<MI_F16>(ctx, src0, src1, dst, ksize, eps, border_type, border_value, target);
                break;
            }
#endif // AURA_ENABLE_NEON_FP16

            case ElemType::F32:
            {
                ret = GuideFilterNormalNeonHelper<DT_F32>(ctx, src0, src1, dst, ksize, eps, border_type, border_value, target);
                break;
            }

            default:
            {
                AURA_ADD_ERROR_STRING(ctx, "Unsupported source format");
                return Status::ERROR;
            }
        }
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura