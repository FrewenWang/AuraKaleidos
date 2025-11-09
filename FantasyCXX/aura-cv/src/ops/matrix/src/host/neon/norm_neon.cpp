#include "norm_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"

namespace aura
{

template <typename Tp>
struct NormInfTraits
{
    using VType = typename neon::QVector<Tp>::VType;
    using SType = Tp;
};

template <>
struct NormInfTraits<DT_S8>
{
    using VType = uint8x16_t;
    using SType = DT_U8;
};

template <>
struct NormInfTraits<DT_S16>
{
    using VType = uint16x8_t;
    using SType = DT_U16;
};

template <typename Tp>
static Status NormInfNeonImpl(const Mat &mat, std::vector<DT_F64> &task_result, DT_S32 start_row, DT_S32 end_row)
{
    constexpr DT_S32 VEC_SIZE = 16 / sizeof(Tp);
    using VType = typename NormInfTraits<Tp>::VType;
    using SType = typename NormInfTraits<Tp>::SType;

    Sizes3 sz      = mat.GetSizes();
    DT_S32 width   = sz.m_width;
    DT_S32 channel = sz.m_channel;

    DT_S32 row_elem_count   = width * channel;
    DT_S32 elem_count_align = row_elem_count & (-VEC_SIZE);

    VType res_vec;
    neon::vdup(res_vec, 0);

    for (DT_S32 y = start_row; y < end_row; ++y)
    {
        auto mat_row = mat.Ptr<Tp>(y);

        DT_S32 x = 0;
        for (; x < elem_count_align; x += VEC_SIZE)
        {
            typename neon::QVector<Tp>::VType src_data;
            neon::vload(mat_row + x, src_data);
            VType var = neon::vreinterpret(neon::vabs(src_data));
            res_vec = neon::vmax(res_vec, var);
        }

        SType max_border_val = 0;
        for (; x < row_elem_count; x++)
        {
            max_border_val = Max(max_border_val, static_cast<SType>(Abs(mat_row[x])));
        }
        VType border_val;
        neon::vdup(border_val, max_border_val);
        res_vec = neon::vmax(res_vec, border_val);
    }

    SType scalar_result[VEC_SIZE] = {0};
    neon::vstore(scalar_result, res_vec);

    DT_F64 max_value = 0;
    for (DT_S32 i = 0; i < VEC_SIZE; i++)
    {
        max_value = Max(max_value, Abs(static_cast<DT_F64>(scalar_result[i])));
    }

    DT_S32 idx = start_row;
    task_result[idx] = max_value;

    return Status::OK;
}

static Status NormInfNeon(Context *ctx, const Mat &mat, DT_F64 *result, const OpTarget &target)
{
    AURA_UNUSED(target);
    Status ret = Status::OK;
    *result = 0.0;

    ElemType type  = mat.GetElemType();
    Sizes3 sz      = mat.GetSizes();
    DT_S32 height  = sz.m_height;

    std::vector<DT_F64> task_result(height, 0.0);

    WorkerPool *wp = ctx->GetWorkerPool();

    if (DT_NULL == wp)
    {
        return Status::ERROR;
    }

    switch (type)
    {
        case ElemType::U8:
        {
            ret = wp->ParallelFor(0, height, NormInfNeonImpl<DT_U8>, std::ref(mat), std::ref(task_result));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ParallelFor NormInfNeonImpl<DT_U8> failed.");
            }
            break;
        }
        case ElemType::S8:
        {
            ret = wp->ParallelFor(0, height, NormInfNeonImpl<DT_S8>, std::ref(mat), std::ref(task_result));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ParallelFor NormInfNeonImpl<DT_S8> failed.");
            }
            break;
        }
        case ElemType::U16:
        {
            ret = wp->ParallelFor(0, height, NormInfNeonImpl<DT_U16>, std::ref(mat), std::ref(task_result));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ParallelFor NormInfNeonImpl<DT_U16> failed.");
            }
            break;
        }
        case ElemType::S16:
        {
            ret = wp->ParallelFor(0, height, NormInfNeonImpl<DT_S16>, std::ref(mat), std::ref(task_result));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ParallelFor NormInfNeonImpl<DT_S16> failed.");
            }
            break;
        }
#if defined(AURA_ENABLE_NEON_FP16)
        case ElemType::F16:
        {
            ret = wp->ParallelFor(0, height, NormInfNeonImpl<MI_F16>, std::ref(mat), std::ref(task_result));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ParallelFor NormInfNeonImpl<MI_F16> failed.");
            }
            break;
        }
#endif
        case ElemType::F32:
        {
            ret = wp->ParallelFor(0, height, NormInfNeonImpl<DT_F32>, std::cref(mat), std::ref(task_result));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ParallelFor NormInfNeonImpl<DT_F32> failed.");
            }
            break;
        }
        default:
        {
            ret = Status::ERROR;
            AURA_ADD_ERROR_STRING(ctx, "unsupported elem_type.");
            break;
        }
    }

    for (const auto &val : task_result)
    {
        *result = Max(*result, val);
    }

    AURA_RETURN(ctx, ret);
}

NormNeon::NormNeon(Context *ctx, const OpTarget &target) : NormImpl(ctx, target)
{}

Status NormNeon::SetArgs(const Array *src, DT_F64 *result, NormType type)
{
    if (NormImpl::SetArgs(src, result, type) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "NormImpl::SetArgs failed");
        return Status::ERROR;
    }

    if (src->GetArrayType() != ArrayType::MAT)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status NormNeon::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);

    if (DT_NULL == src)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src is null");
        return Status::ERROR;
    }

    Status ret = Status::OK;

    switch (m_type)
    {
        case NormType::NORM_INF:
        {
            ret = NormInfNeon(m_ctx, *src, m_result, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "NormInfNeon failed.");
            }
            break;
        }
        case NormType::NORM_L1:
        {
            ret = AbsSumNeon(m_ctx, *src, *m_result, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "AbsSumNeon failed.");
            }
            break;
        }
        case NormType::NORM_L2:
        {
            Scalar scalar_result = Scalar::All(0.0);
            ret = SqSumNeon(m_ctx, *src, scalar_result, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "SqSumNeon failed.");
            }
            *m_result = Sqrt(scalar_result.m_val[0] + scalar_result.m_val[1] + scalar_result.m_val[2] + scalar_result.m_val[3]);
            break;
        }
        case NormType::NORM_L2SQR:
        {
            Scalar scalar_result = Scalar::All(0.0);
            ret = SqSumNeon(m_ctx, *src, scalar_result, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "SqSumNeon failed.");
            }
            *m_result = scalar_result.m_val[0] + scalar_result.m_val[1] + scalar_result.m_val[2] + scalar_result.m_val[3];
            break;
        }
        case NormType::NORM_MINMAX:
        {
            ret = Status::ERROR;
            AURA_ADD_ERROR_STRING(m_ctx, "NormType::NORM_MINMAX is used for normalize function.");
            break;
        }
        default:
        {
            ret = Status::ERROR;
            AURA_ADD_ERROR_STRING(m_ctx, "NormNeonHelper unsupported NormType.");
            break;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura