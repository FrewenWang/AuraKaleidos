#include "sum_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

#define SUM_BLK   (8)

namespace aura
{

template <typename Tp>
struct SumNoneTraits
{
    using SumType = MI_F64;
    static constexpr MI_S32 BLOCK_SIZE = (1 << 20);
};

template <>
struct SumNoneTraits<MI_U8>
{
    using SumType = MI_U16;
    static constexpr MI_S32 BLOCK_SIZE = (1 << 8);
};

template <>
struct SumNoneTraits<MI_S8>
{
    using SumType = MI_S16;
    static constexpr MI_S32 BLOCK_SIZE = (1 << 8);
};

template <>
struct SumNoneTraits<MI_U16>
{
    using SumType = MI_U32;
    static constexpr MI_S32 BLOCK_SIZE = (1 << 16);
};

template <>
struct SumNoneTraits<MI_S16>
{
    using SumType = MI_S32;
    static constexpr MI_S32 BLOCK_SIZE = (1 << 16);
};

#if defined(AURA_BUILD_HOST)
template <>
struct SumNoneTraits<MI_F16>
{
    using SumType = MI_F32;
    static constexpr MI_S32 BLOCK_SIZE = (1 << 16);
};
#endif

template <typename SrcType, typename SumType, MI_S32 C, MI_S32 BLOCK_SIZE>
static typename std::enable_if<C <= 3, Status>::type
SumNoneImpl(const Mat &mat, std::vector<Scalar> &task_result, MI_S32 start_blk, MI_S32 end_blk)
{
    const MI_S32 width  = mat.GetSizes().m_width;
    Scalar result = Scalar(0, 0, 0, 0);

    MI_S32 start_row = start_blk * SUM_BLK;
    MI_S32 end_row   = Min(end_blk * SUM_BLK, mat.GetSizes().m_height);

    for (MI_S32 y = start_row; y < end_row; y++)
    {
        const SrcType *data = mat.Ptr<SrcType>(y);

        MI_S32 x = 0;
        for (; x + BLOCK_SIZE <= width; x += BLOCK_SIZE)
        {
            SumType sum_row[4] = {0};
            for (MI_S32 i_block = 0; i_block < BLOCK_SIZE; i_block++)
            {
                for (MI_S32 ch = 0; ch < C; ch++)
                {
                    sum_row[ch] += data[(x + i_block) * C + ch];
                }
            }
            result += Scalar(sum_row[0], sum_row[1], sum_row[2], sum_row[3]);
        }

        SumType sum_row[4] = {0};
        for (; x < width; x++)
        {
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                sum_row[ch] += data[x * C + ch];
            }
        }
        result += Scalar(sum_row[0], sum_row[1], sum_row[2], sum_row[3]);
    }

    MI_S32 idx = start_blk;
    task_result[idx] = result;

    return Status::OK;
}

template <typename Tp>
static Status SumNoneHelper(Context *ctx, const Mat &mat, Scalar &sum, const OpTarget &target)
{
    const MI_S32 channel = mat.GetSizes().m_channel;
    const MI_S32 height = mat.GetSizes().m_height;

    Status ret = Status::ERROR;

    MI_S32 blk_nums  = (height + SUM_BLK - 1) / SUM_BLK;
    MI_S32 task_nums = target.m_data.none.enable_mt ? blk_nums : 1;
    std::vector<Scalar> task_result(task_nums, Scalar::All(0.0));

    using SumType = typename SumNoneTraits<Tp>::SumType;
    constexpr MI_S32 BLOCK_SIZE = SumNoneTraits<Tp>::BLOCK_SIZE;

#define AURA_SUM_NONE_IMPL(channel)                                                                                               \
    if (target.m_data.none.enable_mt)                                                                                             \
    {                                                                                                                             \
        WorkerPool *wp = ctx->GetWorkerPool();                                                                                    \
        if (MI_NULL == wp)                                                                                                        \
        {                                                                                                                         \
            AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");                                                                   \
            return Status::ERROR;                                                                                                 \
        }                                                                                                                         \
                                                                                                                                  \
        ret = wp->ParallelFor(static_cast<MI_S32>(0), blk_nums, SumNoneImpl<Tp, SumType, channel, BLOCK_SIZE>,                    \
                              std::cref(mat), std::ref(task_result));                                                             \
    }                                                                                                                             \
    else                                                                                                                          \
    {                                                                                                                             \
        ret = SumNoneImpl<Tp, SumType, channel, BLOCK_SIZE>(mat, task_result, 0, blk_nums);                                       \
    }                                                                                                                             \
    if (ret != Status::OK)                                                                                                        \
    {                                                                                                                             \
        MI_CHAR error_msg[128];                                                                                                   \
        std::snprintf(error_msg, sizeof(error_msg), "SumNoneImpl failed (channel %s)", #channel);                                 \
        AURA_ADD_ERROR_STRING(ctx, error_msg);                                                                                    \
    }

    switch (channel)
    {
        case 1:
        {
            AURA_SUM_NONE_IMPL(1);
            break;
        }
        case 2:
        {
            AURA_SUM_NONE_IMPL(2);
            break;
        }
        case 3:
        {
            AURA_SUM_NONE_IMPL(3);
            break;
        }
        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "channel number must be less than 4");
            return Status::ERROR;
        }
    }

    sum = Scalar::All(0.0);
    for (const auto &val : task_result)
    {
        sum += val;
    }

    AURA_RETURN(ctx, ret);
}

SumNone::SumNone(Context *ctx, const OpTarget &target) : SumImpl(ctx, target)
{}

Status SumNone::SetArgs(const Array *src, Scalar *result)
{
    if (SumImpl::SetArgs(src, result) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SumImpl::SetArgs failed");
        return Status::ERROR;
    }

    if (src->GetArrayType() != ArrayType::MAT)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status SumNone::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);

    if (MI_NULL == src)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src is null");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    switch (src->GetElemType())
    {
        case ElemType::U8:
        {
            ret = SumNoneHelper<MI_U8>(m_ctx, *src, *m_result, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "SumNoneHelper<MI_U8> failed.");
            }
            break;
        }
        case ElemType::S8:
        {
            ret = SumNoneHelper<MI_S8>(m_ctx, *src, *m_result, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "SumNoneHelper<MI_S8> failed.");
            }
            break;
        }
        case ElemType::U16:
        {
            ret = SumNoneHelper<MI_U16>(m_ctx, *src, *m_result, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "SumNoneHelper<MI_U16> failed.");
            }
            break;
        }
        case ElemType::S16:
        {
            ret = SumNoneHelper<MI_S16>(m_ctx, *src, *m_result, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "SumNoneHelper<MI_S16> failed.");
            }
            break;
        }
        case ElemType::U32:
        {
            ret = SumNoneHelper<MI_U32>(m_ctx, *src, *m_result, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "SumNoneHelper<MI_U32> failed.");
            }
            break;
        }
        case ElemType::S32:
        {
            ret = SumNoneHelper<MI_S32>(m_ctx, *src, *m_result, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "SumNoneHelper<MI_S32> failed.");
            }
            break;
        }
#if defined(AURA_BUILD_HOST)
        case ElemType::F16:
        {
            ret = SumNoneHelper<MI_F16>(m_ctx, *src, *m_result, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "SumNoneHelper<MI_F16> failed.");
            }
            break;
        }
        case ElemType::F32:
        {
            ret = SumNoneHelper<MI_F32>(m_ctx, *src, *m_result, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "SumNoneHelper<MI_F32> failed.");
            }
            break;
        }
#endif

        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "unsupported element type");
            return Status::ERROR;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

MeanNone::MeanNone(Context *ctx, const OpTarget &target) : SumNone(ctx, target)
{}

Status MeanNone::Run()
{
    Status ret = SumNone::Run();
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SumNone run failed.");
        return Status::ERROR;
    }

    const MI_S32 height = m_src->GetSizes().m_height;
    const MI_S32 width  = m_src->GetSizes().m_width;
    *m_result           = (*m_result) / static_cast<MI_F64>(height * width);

    return Status::OK;
}

} // namespace aura