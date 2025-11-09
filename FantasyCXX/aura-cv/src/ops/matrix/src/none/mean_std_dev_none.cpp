#include "mean_std_dev_impl.hpp"
#include "aura/ops/core.h"
#include "aura/runtime/mat.h"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

template <typename Tp>
struct MeanStdDevTraits
{
    static_assert(is_arithmetic<Tp>::value, "MeanStdDevTraits type must be arithmetic type.");

    using CastType = typename std::conditional<sizeof(Tp) < 4 && is_integral<Tp>::value,
                              typename std::conditional<is_signed<Tp>::value, DT_S32, DT_U32>::type,
                              DT_F64>::type;
    using SumType = typename std::conditional<sizeof(Tp) < 4 && is_integral<Tp>::value,
                             typename std::conditional<is_signed<Tp>::value, DT_S32, DT_U32>::type,
                             DT_F64>::type;

    using SqSumType = typename std::conditional<sizeof(Tp) < 4 && is_integral<Tp>::value,
                               DT_U64, DT_F64>::type;
};

template <typename Tp0, typename CastType, typename SumType, typename SqSumType, DT_S32 C>
static DT_VOID MeanStdPerRow(const Tp0 *data, DT_S32 width, SumType *sum, SqSumType *sq_sum)
{
    for (DT_S32 x = 0; x < width; ++x)
    {
        for (DT_S32 ch = 0; ch < C; ++ch)
        {
            CastType value = static_cast<CastType>(data[x * C + ch]);
            sum[ch] += value;
            sq_sum[ch] += value * value;
        }
    }
}

template <typename Tp, typename CastType, typename SumType, typename SqSumType>
static Status MeanStdDevSumImpl(Context *ctx, const Mat &mat, std::vector<std::vector<DT_F64>> &sum, std::vector<std::vector<DT_F64>> &sq_sum,
                                DT_S32 start_row, DT_S32 end_row)
{
    Sizes3 sz      = mat.GetSizes();
    DT_S32 width   = sz.m_width;
    DT_S32 channel = sz.m_channel;

    DT_S32 thread_idx = ctx->GetWorkerPool() ? ctx->GetWorkerPool()->GetComputeThreadIdx() : 0;

    for (DT_S32 y = start_row; y < end_row; ++y)
    {
        SumType row_sum[4] = {0, 0, 0, 0};
        SqSumType row_sq_sum[4] = {0, 0, 0, 0};

        const Tp *src_c = mat.Ptr<Tp>(y);

        switch (channel)
        {
            case 1:
            {
                MeanStdPerRow<Tp, CastType, SumType, SqSumType, 1>(src_c, width, row_sum, row_sq_sum);
                break;
            }
            case 2:
            {
                MeanStdPerRow<Tp, CastType, SumType, SqSumType, 2>(src_c, width, row_sum, row_sq_sum);
                break;
            }
            case 3:
            {
                MeanStdPerRow<Tp, CastType, SumType, SqSumType, 3>(src_c, width, row_sum, row_sq_sum);
                break;
            }
            default:
            {
                AURA_ADD_ERROR_STRING(ctx, "MeanStdDev with unsupported channels.");
                return Status::ERROR;
            }
        }

        for (DT_S32 i = 0; i < 4; ++i)
        {
            sum[thread_idx][i] += row_sum[i];
            sq_sum[thread_idx][i] += row_sq_sum[i];
        }
    }

    return Status::OK;
}

template <typename Tp>
static Status MeanStdDevNoneHelper(Context *ctx, const Mat &mat, Scalar &mean, Scalar &std_dev, OpTarget &target)
{
    using CastType  = typename MeanStdDevTraits<Tp>::CastType;
    using SumType   = typename MeanStdDevTraits<Tp>::SumType;
    using SqSumType = typename MeanStdDevTraits<Tp>::SqSumType;

    Status ret = Status::OK;

    Sizes3 sz     = mat.GetSizes();
    DT_S32 width  = sz.m_width;
    DT_S32 height = sz.m_height;

    SumType max_sum_elem = std::numeric_limits<SumType>::max() / std::numeric_limits<Tp>::max();
    SqSumType max_sq_sum_elem = std::numeric_limits<SqSumType>::max() / std::numeric_limits<Tp>::max() / std::numeric_limits<Tp>::max();

    DT_S32 enable_mt  = target.m_data.none.enable_mt;
    DT_S32 thread_num = (enable_mt && ctx->GetWorkerPool()) ? ctx->GetWorkerPool()->GetComputeThreadNum() : 1;

    std::vector<std::vector<DT_F64>> sum(thread_num, std::vector<DT_F64>(4, 0.f));
    std::vector<std::vector<DT_F64>> sq_sum(thread_num, std::vector<DT_F64>(4, 0.f));

    if (static_cast<SumType>(width) > max_sum_elem || static_cast<SqSumType>(width) > max_sq_sum_elem)
    {
        if (enable_mt)
        {
            WorkerPool *wp = ctx->GetWorkerPool();
            if (DT_NULL == wp)
            {
                AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
                return Status::ERROR;
            }

            ret = wp->ParallelFor(static_cast<DT_S32>(0), height, MeanStdDevSumImpl<Tp, CastType, DT_F64, DT_F64>, ctx, std::cref(mat), std::ref(sum), std::ref(sq_sum));
        }
        else
        {
            ret = MeanStdDevSumImpl<Tp, CastType, DT_F64, DT_F64>(ctx, mat, sum, sq_sum, static_cast<DT_S32>(0), height);
        }
    }
    else
    {
        if (enable_mt)
        {
            WorkerPool *wp = ctx->GetWorkerPool();
            if (DT_NULL == wp)
            {
                AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
                return Status::ERROR;
            }

            ret = wp->ParallelFor(static_cast<DT_S32>(0), height, MeanStdDevSumImpl<Tp, CastType, SumType, SqSumType>, ctx, std::cref(mat), std::ref(sum), std::ref(sq_sum));
        }
        else
        {
            ret = MeanStdDevSumImpl<Tp, CastType, SumType, SqSumType>(ctx, mat, sum, sq_sum, static_cast<DT_S32>(0), height);
        }
    }

    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "MeanStdDev compute sum and sq_sum failed.");
        return Status::ERROR;
    }

    DT_F64 total_elem_count = static_cast<DT_F64>(width) * height;

    for (DT_S32 thread_idx = 1; thread_idx < (DT_S32)sum.size(); thread_idx++)
    {
        sum[0][0] += sum[thread_idx][0];
        sum[0][1] += sum[thread_idx][1];
        sum[0][2] += sum[thread_idx][2];
        sum[0][3] += sum[thread_idx][3];

        sq_sum[0][0] += sq_sum[thread_idx][0];
        sq_sum[0][1] += sq_sum[thread_idx][1];
        sq_sum[0][2] += sq_sum[thread_idx][2];
        sq_sum[0][3] += sq_sum[thread_idx][3];
    }

    for (DT_S32 i = 0; i < 4; ++i)
    {
        mean.m_val[i] = sum[0][i] / total_elem_count;
    }

    for (DT_S32 i = 0; i < 4; ++i)
    {
        std_dev.m_val[i] = Sqrt(sq_sum[0][i] / total_elem_count - mean.m_val[i] * mean.m_val[i]);
    }

    return Status::OK;
}

MeanStdDevNone::MeanStdDevNone(Context *ctx, const OpTarget &target) : MeanStdDevImpl(ctx, target)
{}

Status MeanStdDevNone::SetArgs(const Array *src, Scalar *mean, Scalar *std_dev)
{
    if (MeanStdDevImpl::SetArgs(src, mean, std_dev) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "MeanStdDevImpl::SetArgs failed");
        return Status::ERROR;
    }

    if (src->GetArrayType() != ArrayType::MAT)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status MeanStdDevNone::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);

    if (DT_NULL == src)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src is null");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    switch (src->GetElemType())
    {
        case ElemType::U8:
        {
            ret = MeanStdDevNoneHelper<DT_U8>(m_ctx, *src, *m_mean, *m_std_dev, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "MeanStdDevNoneHelper<DT_U8> failed.");
            }
            break;
        }
        case ElemType::S8:
        {
            ret = MeanStdDevNoneHelper<DT_S8>(m_ctx, *src, *m_mean, *m_std_dev, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "MeanStdDevNoneHelper<DT_S8> failed.");
            }
            break;
        }
        case ElemType::U16:
        {
            ret = MeanStdDevNoneHelper<DT_U16>(m_ctx, *src, *m_mean, *m_std_dev, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "MeanStdDevNoneHelper<DT_U16> failed.");
            }
            break;
        }
        case ElemType::S16:
        {
            ret = MeanStdDevNoneHelper<DT_S16>(m_ctx, *src, *m_mean, *m_std_dev, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "MeanStdDevNoneHelper<DT_S16> failed.");
            }
            break;
        }
        case ElemType::U32:
        {
            ret = MeanStdDevNoneHelper<DT_U32>(m_ctx, *src, *m_mean, *m_std_dev, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "MeanStdDevNoneHelper<DT_U32> failed.");
            }
            break;
        }
        case ElemType::S32:
        {
            ret = MeanStdDevNoneHelper<DT_S32>(m_ctx, *src, *m_mean, *m_std_dev, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "MeanStdDevNoneHelper<DT_S32> failed.");
            }
            break;
        }
#if defined(AURA_BUILD_HOST)
        case ElemType::F32:
        {
            ret = MeanStdDevNoneHelper<DT_F32>(m_ctx, *src, *m_mean, *m_std_dev, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "MeanStdDevNoneHelper<DT_F32> failed.");
            }
            break;
        }
#endif
        default:
        {
            ret = Status::ERROR;
            AURA_ADD_ERROR_STRING(m_ctx, "MeanStdDevNone unsupported element type.");
            break;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura