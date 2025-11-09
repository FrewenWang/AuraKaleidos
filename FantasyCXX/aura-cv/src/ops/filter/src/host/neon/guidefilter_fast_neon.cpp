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

/*
Possibilities: (ksize < 256)

| Tp  | SumRow  | SumKernel |  Sq  | SqSumRow | SqSumKernel |
-------------------------------------------------------------
| U8  |   U16   |  U16/U32  |  U16 |   U32    |     U32     |    #SumKernel can be U16 when ksize < 16
| S8  |   S16   |  S16/S32  |  S16 |   S32    |     S32     |    #SumKernel can be S16 when ksize < 16
| U16 |   U32   |  U32      |  U32 |   U64    |     U64     |
| S16 |   S32   |  S32      |  S32 |   S64    |     S64     |
| F16 |   F32   |  F32      |  F32 |   F32    |     F32     |
| F32 |   F32   |  F32      |  F32 |   F32    |     F32     |

# SumRow = Sq = SumKernel = Promote<Tp>::Type   (when ksize < 16)
# SqSumRow = SqSumKernel = Promote<SumRow>::Type
*/

template <typename Tp, BorderType BORDER_TYPE, DT_S32 C>
static Status GuideFilterFastNeonImpl(Context *ctx, const Mat &src0, const Mat &src1, Mat &dst, DT_S32 ksize,
                                      DT_F32 eps, const Scalar &border_value, const OpTarget &target)
{
    AURA_UNUSED(target);

    using SumType       = typename Promote<Tp>::Type;
    using SumKernelType = typename Promote<Tp>::Type;
    using SqSumType     = typename Promote<SumType>::Type;

    const DT_S32 fast_ksize = GetFastKsize(ksize);

    Sizes3 sub_sizes = {src0.GetSizes().m_height >> 1, src0.GetSizes().m_width >> 1, src0.GetSizes().m_channel};
    Mat    sub_src0  = Mat(ctx, src0.GetElemType(), sub_sizes, AURA_MEM_DEFAULT);
    Mat    sub_src1  = Mat(ctx, src1.GetElemType(), sub_sizes, AURA_MEM_DEFAULT);
    if ((!sub_src0.IsValid()) || (!sub_src1.IsValid()))
    {
        AURA_ADD_ERROR_STRING(ctx, "Mat sub_src0 or sub_src1 is invalid");
        return Status::ERROR;
    }

    // resize src0 and src1
    Status ret = Status::OK;
    ret |= IResize(ctx, src0, sub_src0, InterpType::LINEAR, target);
    ret |= IResize(ctx, src1, sub_src1, InterpType::LINEAR, target);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "IResize failed");
        return Status::ERROR;
    }

    Mat sub_a = Mat(ctx, ElemType::F32, sub_sizes);
    Mat sub_b = Mat(ctx, ElemType::F32, sub_sizes);
    if ((!sub_a.IsValid()) || (!sub_b.IsValid()))
    {
        AURA_ADD_ERROR_STRING(ctx, "Mat sub_a or sub_b is invalid");
        return Status::ERROR;
    }

    const DT_S32 height = sub_src0.GetSizes().m_height;
    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return Status::ERROR;
    }

    const Sizes3 sz    = sub_src0.GetSizes();
    DT_S32 width       = sz.m_width;
    DT_S32 width_align = AURA_ALIGN((width + fast_ksize) * C, 64);

    // 0. prepare buffer
    ThreadBuffer row_a_buffer(ctx,  width_align * sizeof(Tp)      * ksize); // to store data with expanded border
    ThreadBuffer row_b_buffer(ctx,  width_align * sizeof(Tp)      * ksize); // to store data with expanded border
    ThreadBuffer row_aa_buffer(ctx, width_align * sizeof(SumType) * ksize); // a * a
    ThreadBuffer row_ab_buffer(ctx, width_align * sizeof(SumType) * ksize); // a * b

    ThreadBuffer sum_a_buffer(ctx,  width_align * sizeof(SumKernelType));   // sum of a
    ThreadBuffer sum_b_buffer(ctx,  width_align * sizeof(SumKernelType));   // sum of b
    ThreadBuffer sum_aa_buffer(ctx, width_align * sizeof(SqSumType));       // sum of a * a
    ThreadBuffer sum_ab_buffer(ctx, width_align * sizeof(SqSumType));       // sum of a * b

    ThreadBuffer arr_ptr_buffer(ctx, 4 * ksize * sizeof(DT_VOID*));

    ret = wp->ParallelFor(0, height, GuideFilterCalcABNeonImpl<Tp, SumType, SqSumType, BORDER_TYPE, C>,
                          std::cref(sub_src0), std::cref(sub_src1), std::ref(sub_a), std::ref(sub_b),
                          std::ref(row_a_buffer), std::ref(row_b_buffer), std::ref(row_aa_buffer), std::ref(row_ab_buffer),
                          std::ref(sum_a_buffer), std::ref(sum_b_buffer), std::ref(sum_aa_buffer), std::ref(sum_ab_buffer),
                          std::ref(arr_ptr_buffer), fast_ksize, eps, border_value);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "ParallelFor run failed");
        return Status::ERROR;
    }

    Mat mean_sub_a = Mat(ctx, ElemType::F32, sub_sizes);
    Mat mean_sub_b = Mat(ctx, ElemType::F32, sub_sizes);
    if (!mean_sub_a.IsValid() || !mean_sub_b.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "Mat mean_sub_a or mean_sub_b is invalid");
        return Status::ERROR;
    }

    ret |= IBoxfilter(ctx, sub_a, mean_sub_a, fast_ksize, BORDER_TYPE, border_value, target);
    ret |= IBoxfilter(ctx, sub_b, mean_sub_b, fast_ksize, BORDER_TYPE, border_value, target);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "BoxFilterNeon failed");
        return Status::ERROR;
    }

    Mat mean_a = Mat(ctx, ElemType::F32, src0.GetSizes());
    Mat mean_b = Mat(ctx, ElemType::F32, src0.GetSizes());
    if (!mean_a.IsValid() || !mean_b.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "Mat mean_a or mean_b is invalid");
        return Status::ERROR;
    }

    ret |= IResize(ctx, mean_sub_a, mean_a, InterpType::LINEAR, target);
    ret |= IResize(ctx, mean_sub_b, mean_b, InterpType::LINEAR, target);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "IResize failed");
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
static Status GuideFilterFastNeonImpl(Context *ctx, const Mat &src0, Mat &dst, DT_S32 ksize,
                                      DT_F32 eps, const Scalar &border_value, const OpTarget &target)
{
    AURA_UNUSED(target);

    using SumType       = typename Promote<Tp>::Type;
    using SumKernelType = typename Promote<Tp>::Type;
    using SqSumType     = typename Promote<SumType>::Type;

    const DT_S32 fast_ksize = GetFastKsize(ksize);

    Sizes3 sub_sizes = {src0.GetSizes().m_height >> 1, src0.GetSizes().m_width >> 1, src0.GetSizes().m_channel};
    Mat    sub_src0  = Mat(ctx, src0.GetElemType(), sub_sizes, AURA_MEM_DEFAULT);
    if (!sub_src0.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "Mat sub_src0 or sub_src1 is invalid");
        return Status::ERROR;
    }

    // resize src0 and src1
    Status ret = Status::ERROR;
    ret = IResize(ctx, src0, sub_src0, InterpType::LINEAR, target);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "IResize failed");
        return Status::ERROR;
    }

    Mat sub_a = Mat(ctx, ElemType::F32, sub_sizes);
    Mat sub_b = Mat(ctx, ElemType::F32, sub_sizes);
    if ((!sub_a.IsValid()) || (!sub_b.IsValid()))
    {
        AURA_ADD_ERROR_STRING(ctx, "Mat sub_a or sub_b is invalid");
        return Status::ERROR;
    }

    const DT_S32 height = sub_src0.GetSizes().m_height;
    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return Status::ERROR;
    }

    const Sizes3 sz    = sub_src0.GetSizes();
    DT_S32 width       = sz.m_width;
    DT_S32 width_align = AURA_ALIGN((width + fast_ksize) * C, 64);

    // 0. prepare buffer
    ThreadBuffer row_a_buffer(ctx,  width_align * sizeof(Tp)      * ksize); // to store data with expanded border
    ThreadBuffer row_aa_buffer(ctx, width_align * sizeof(SumType) * ksize); // a * a

    ThreadBuffer sum_a_buffer(ctx,  width_align * sizeof(SumKernelType));   // sum of a
    ThreadBuffer sum_aa_buffer(ctx, width_align * sizeof(SqSumType));       // sum of a * a

    ThreadBuffer arr_ptr_buffer(ctx, 2 * ksize * sizeof(DT_VOID*));

    ret = wp->ParallelFor(0, height, GuideFilterCalcABSameSrcNeonImpl<Tp, SumType, SqSumType, BORDER_TYPE, C>,
                          std::cref(sub_src0), std::ref(sub_a), std::ref(sub_b),
                          std::ref(row_a_buffer), std::ref(row_aa_buffer),
                          std::ref(sum_a_buffer), std::ref(sum_aa_buffer),
                          std::ref(arr_ptr_buffer), fast_ksize, eps, border_value);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "ParallelFor run failed");
        return Status::ERROR;
    }

    Mat mean_sub_a = Mat(ctx, ElemType::F32, sub_sizes);
    Mat mean_sub_b = Mat(ctx, ElemType::F32, sub_sizes);
    if (!mean_sub_a.IsValid() || !mean_sub_b.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "Mat mean_sub_a or mean_sub_b is invalid");
        return Status::ERROR;
    }

    ret |= IBoxfilter(ctx, sub_a, mean_sub_a, fast_ksize, BORDER_TYPE, border_value, target);
    ret |= IBoxfilter(ctx, sub_b, mean_sub_b, fast_ksize, BORDER_TYPE, border_value, target);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "BoxFilterNeon failed");
        return Status::ERROR;
    }

    Mat mean_a = Mat(ctx, ElemType::F32, src0.GetSizes());
    Mat mean_b = Mat(ctx, ElemType::F32, src0.GetSizes());
    if (!mean_a.IsValid() || !mean_b.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "Mat mean_a or mean_b is invalid");
        return Status::ERROR;
    }

    ret |= IResize(ctx, mean_sub_a, mean_a, InterpType::LINEAR, target);
    ret |= IResize(ctx, mean_sub_b, mean_b, InterpType::LINEAR, target);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "IResize failed");
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

template <typename Tp, BorderType BORDER_TYPE>
Status GuideFilterFastNeonHelper(Context *ctx, const Mat &src0, const Mat &src1, Mat &dst, DT_S32 ksize, DT_F32 eps,
                                 const Scalar &border_value, const OpTarget &target)
{
    Status ret = Status::ERROR;

    switch (src0.GetSizes().m_channel)
    {
        case 1:
        {
            ret = GuideFilterFastNeonImpl<Tp, BORDER_TYPE, 1>(ctx, src0, src1, dst, ksize, eps, border_value, target);
            break;
        }
        case 2:
        {
            ret = GuideFilterFastNeonImpl<Tp, BORDER_TYPE, 2>(ctx, src0, src1, dst, ksize, eps, border_value, target);
            break;
        }
        case 3:
        {
            ret = GuideFilterFastNeonImpl<Tp, BORDER_TYPE, 3>(ctx, src0, src1, dst, ksize, eps, border_value, target);
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
Status GuideFilterFastNeonHelper(Context *ctx, const Mat &src, Mat &dst, DT_S32 ksize, DT_F32 eps,
                                 const Scalar &border_value, const OpTarget &target)
{
    Status ret = Status::ERROR;

    switch (src.GetSizes().m_channel)
    {
        case 1:
        {
            ret = GuideFilterFastNeonImpl<Tp, BORDER_TYPE, 1>(ctx, src, dst, ksize, eps, border_value, target);
            break;
        }
        case 2:
        {
            ret = GuideFilterFastNeonImpl<Tp, BORDER_TYPE, 2>(ctx, src, dst, ksize, eps, border_value, target);
            break;
        }
        case 3:
        {
            ret = GuideFilterFastNeonImpl<Tp, BORDER_TYPE, 3>(ctx, src, dst, ksize, eps, border_value, target);
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
Status GuideFilterFastNeonHelper(Context *ctx, const Mat &src0, const Mat &src1, Mat &dst, DT_S32 ksize, DT_F32 eps,
                                 BorderType border_type, const Scalar &border_value, const OpTarget &target)
{
    Status ret = Status::ERROR;

    switch (border_type)
    {
        case BorderType::CONSTANT:
        {
            ret = GuideFilterFastNeonHelper<Tp, BorderType::CONSTANT>(ctx, src0, src1, dst, ksize, eps, border_value, target);
            break;
        }
        case BorderType::REPLICATE:
        {
            ret = GuideFilterFastNeonHelper<Tp, BorderType::REPLICATE>(ctx, src0, src1, dst, ksize, eps, border_value, target);
            break;
        }
        case BorderType::REFLECT_101:
        {
            ret = GuideFilterFastNeonHelper<Tp, BorderType::REFLECT_101>(ctx, src0, src1, dst, ksize, eps, border_value, target);
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
Status GuideFilterFastNeonHelper(Context *ctx, const Mat &src, Mat &dst, DT_S32 ksize, DT_F32 eps, BorderType border_type,
                                 const Scalar &border_value, const OpTarget &target)
{
    Status ret = Status::ERROR;

    switch (border_type)
    {
        case BorderType::CONSTANT:
        {
            ret = GuideFilterFastNeonHelper<Tp, BorderType::CONSTANT>(ctx, src, dst, ksize, eps, border_value, target);
            break;
        }
        case BorderType::REPLICATE:
        {
            ret = GuideFilterFastNeonHelper<Tp, BorderType::REPLICATE>(ctx, src, dst, ksize, eps, border_value, target);
            break;
        }
        case BorderType::REFLECT_101:
        {
            ret = GuideFilterFastNeonHelper<Tp, BorderType::REFLECT_101>(ctx, src, dst, ksize, eps, border_value, target);
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

Status GuideFilterFastNeon(Context *ctx, const Mat &src0, const Mat &src1, Mat &dst, const DT_S32 &ksize, const DT_F32 &eps,
                           BorderType &border_type, const Scalar &border_value, const OpTarget &target)
{
    Status ret = Status::ERROR;
    if (src0.GetData() == src1.GetData())
    {
        switch (src0.GetElemType())
        {
            case ElemType::U8:
            {
                ret = GuideFilterFastNeonHelper<DT_U8>(ctx, src0, dst, ksize, eps, border_type, border_value, target);
                break;
            }

            case ElemType::S8:
            {
                ret = GuideFilterFastNeonHelper<DT_S8>(ctx, src0, dst, ksize, eps, border_type, border_value, target);
                break;
            }

            case ElemType::U16:
            {
                ret = GuideFilterFastNeonHelper<DT_U16>(ctx, src0, dst, ksize, eps, border_type, border_value, target);
                break;
            }

            case ElemType::S16:
            {
                ret = GuideFilterFastNeonHelper<DT_S16>(ctx, src0, dst, ksize, eps, border_type, border_value, target);
                break;
            }

#if defined(AURA_ENABLE_NEON_FP16)
            case ElemType::F16:
            {
                ret = GuideFilterFastNeonHelper<MI_F16>(ctx, src0, dst, ksize, eps, border_type, border_value, target);
                break;
            }
#endif // AURA_ENABLE_NEON_FP16

            case ElemType::F32:
            {
                ret = GuideFilterFastNeonHelper<DT_F32>(ctx, src0, dst, ksize, eps, border_type, border_value, target);
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
                ret = GuideFilterFastNeonHelper<DT_U8>(ctx, src0, src1, dst, ksize, eps, border_type, border_value, target);
                break;
            }

            case ElemType::S8:
            {
                ret = GuideFilterFastNeonHelper<DT_S8>(ctx, src0, src1, dst, ksize, eps, border_type, border_value, target);
                break;
            }

            case ElemType::U16:
            {
                ret = GuideFilterFastNeonHelper<DT_U16>(ctx, src0, src1, dst, ksize, eps, border_type, border_value, target);
                break;
            }

            case ElemType::S16:
            {
                ret = GuideFilterFastNeonHelper<DT_S16>(ctx, src0, src1, dst, ksize, eps, border_type, border_value, target);
                break;
            }

#if defined(AURA_ENABLE_NEON_FP16)
            case ElemType::F16:
            {
                ret = GuideFilterFastNeonHelper<MI_F16>(ctx, src0, src1, dst, ksize, eps, border_type, border_value, target);
                break;
            }
#endif // AURA_ENABLE_NEON_FP16

            case ElemType::F32:
            {
                ret = GuideFilterFastNeonHelper<DT_F32>(ctx, src0, src1, dst, ksize, eps, border_type, border_value, target);
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