#include "psnr_impl.hpp"
#include "aura/ops/core.h"
#include "aura/runtime/mat.h"
#include "aura/runtime/logger.h"

namespace aura
{

template <typename Tp>
struct PsnrTraits
{
    static_assert(is_arithmetic<Tp>::value, "PsnrTraits only support arithemetic type.");
    using CastType = typename std::conditional<sizeof(Tp) < 4 && is_integral<Tp>::value, MI_S32, MI_F64>::type;
    using SqType   = typename std::conditional<sizeof(Tp) < 4 && is_integral<Tp>::value, MI_U32, MI_F64>::type;
};

template <typename Tp>
static MI_F64 PsnrRow(const Tp *src0, const Tp *src1, MI_S32 count)
{
    using CastType = typename PsnrTraits<Tp>::CastType;
    using SqType   = typename PsnrTraits<Tp>::SqType;

    MI_F64 row_sum = 0;

    for (MI_S32 x = 0; x < count; ++x)
    {
        CastType diff  = Abs(static_cast<CastType>(src0[x]) - static_cast<CastType>(src1[x]));
        SqType sq_diff = static_cast<SqType>(diff) * diff;
        row_sum += sq_diff;
    }

    return row_sum;
}

template <typename Tp>
static AURA_VOID PsnrNoneImpl(const Mat &mat0, const Mat &mat1, MI_F64 coef_r, MI_F64 *result)
{
    Sizes3 sz = mat0.GetSizes();
    MI_S32 total_elem_count = sz.Total();
    MI_S32 row_elem_count = sz.m_width * sz.m_channel;

    MI_F64 sq_diff_sum = 0;

    for (MI_S32 y = 0; y < sz.m_height; ++y)
    {
        const Tp *src0_c = mat0.Ptr<Tp>(y);
        const Tp *src1_c = mat1.Ptr<Tp>(y);
        sq_diff_sum += PsnrRow<Tp>(src0_c, src1_c, row_elem_count);
    }

    MI_F64 diff_value = Sqrt(sq_diff_sum / total_elem_count);

    *result = 20 * Log10(coef_r / (diff_value + DBL_EPSILON));
}

PsnrNone::PsnrNone(Context *ctx, const OpTarget &target) : PsnrImpl(ctx, target)
{}

Status PsnrNone::SetArgs(const Array *src0, const Array *src1, MI_F64 coef_r, MI_F64 *result)
{
    if (PsnrImpl::SetArgs(src0, src1, coef_r, result) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "PsnrImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src0->GetArrayType() != ArrayType::MAT) || (src1->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status PsnrNone::Run()
{
    const Mat *src0 = dynamic_cast<const Mat*>(m_src0);
    const Mat *src1 = dynamic_cast<const Mat*>(m_src1);

    if ((MI_NULL == src0) || (MI_NULL == src1))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src0 or src1 is null");
        return Status::ERROR;
    }

    Status ret = Status::OK;

    switch (src0->GetElemType())
    {
        case ElemType::U8:
        {
            PsnrNoneImpl<MI_U8>(*src0, *src1, m_coef_r, m_result);
            break;
        }
        case ElemType::S8:
        {
            PsnrNoneImpl<MI_S8>(*src0, *src1, m_coef_r, m_result);
            break;
        }
        case ElemType::U16:
        {
            PsnrNoneImpl<MI_U16>(*src0, *src1, m_coef_r, m_result);
            break;
        }
        case ElemType::S16:
        {
            PsnrNoneImpl<MI_S16>(*src0, *src1, m_coef_r, m_result);
            break;
        }
        case ElemType::U32:
        {
            PsnrNoneImpl<MI_U32>(*src0, *src1, m_coef_r, m_result);
            break;
        }
        case ElemType::S32:
        {
            PsnrNoneImpl<MI_S32>(*src0, *src1, m_coef_r, m_result);
            break;
        }
#if defined(AURA_BUILD_HOST)
        case ElemType::F32:
        {
            PsnrNoneImpl<MI_F32>(*src0, *src1, m_coef_r, m_result);
            break;
        }
#endif
        default:
        {
            ret = Status::ERROR;
            AURA_ADD_ERROR_STRING(m_ctx, "NormNone unsupported element type.");
            break;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace namespace