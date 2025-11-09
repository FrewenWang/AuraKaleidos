#include "norm_impl.hpp"
#include "aura/runtime/logger.h"

namespace aura
{

template <typename Tp>
struct NormAbsTraits
{
    using CastType = Tp;
};

template <>
struct NormAbsTraits<DT_S8>
{
    using CastType = DT_S16;
};

template <>
struct NormAbsTraits<DT_S16>
{
    using CastType = DT_S32;
};

template <>
struct NormAbsTraits<DT_S32>
{
    using CastType = DT_S64;
};

template <typename Tp>
static DT_F64 NormInfNoneImpl(const Mat &mat)
{
    using CastType = typename NormAbsTraits<Tp>::CastType;

    Sizes3 sz = mat.GetSizes();
    DT_S32 row_elem_count = sz.m_width * sz.m_channel;

    CastType result = 0;

    for (DT_S32 y = 0; y < sz.m_height; ++y)
    {
        const Tp *src_row = mat.Ptr<Tp>(y);

        for (DT_S32 x = 0; x < row_elem_count; ++x)
        {
            CastType value = Abs(SaturateCast<CastType>(src_row[x]));
            result = Max(result, value);
        }
    }

    return SaturateCast<DT_F64>(result);
}

template <typename Tp>
static DT_F64 NormL1NoneImpl(const Mat &mat)
{
    using CastType = typename NormAbsTraits<Tp>::CastType;
    using ResType  = typename std::conditional<is_floating_point<Tp>::value, DT_F64, DT_U64>::type;

    ResType result = 0;

    Sizes3 sz = mat.GetSizes();
    DT_S32 row_elem_count = sz.m_width * sz.m_channel;

    for (DT_S32 y = 0; y < sz.m_height; ++y)
    {
        const Tp *src_row = mat.Ptr<Tp>(y);

        for (DT_S32 x = 0; x < row_elem_count; ++x)
        {
            CastType value = Abs(SaturateCast<CastType>(src_row[x]));
            result += value;
        }
    }

    return SaturateCast<DT_F64>(result);
}

template <typename Tp>
struct NormL2Traits
{
    using ResType  = DT_U64;
    using CastType = typename std::conditional<is_signed<Tp>::value, DT_S64, DT_U64>::type;
};

template <>
struct NormL2Traits<DT_S32>
{
    using ResType  = DT_F64;
    using CastType = DT_S64;
};

template <>
struct NormL2Traits<DT_U32>
{
    using ResType  = DT_F64;
    using CastType = DT_U64;
};

#if defined(AURA_BUILD_HOST)
template <>
struct NormL2Traits<MI_F16>
{
    using ResType  = DT_F64;
    using CastType = DT_F64;
};
#endif

template <>
struct NormL2Traits<DT_F32>
{
    using ResType  = DT_F64;
    using CastType = DT_F64;
};

template <typename Tp>
static DT_F64 NormL2SQRNoneImpl(const Mat &mat)
{
    using ResType  = typename NormL2Traits<Tp>::ResType;
    using CastType = typename NormL2Traits<Tp>::CastType;

    ResType result = 0;

    Sizes3 sz = mat.GetSizes();
    DT_S32 row_elem_count = sz.m_width * sz.m_channel;

    for (DT_S32 y = 0; y < sz.m_height; ++y)
    {
        const Tp *src_row = mat.Ptr<Tp>(y);

        for (DT_S32 x = 0; x < row_elem_count; ++x)
        {
            CastType value = SaturateCast<CastType>(src_row[x]);
            result += value * value;
        }
    }

    return SaturateCast<DT_F64>(result);
}

template <typename Tp>
static DT_F64 NormL2NoneImpl(const Mat &mat)
{
    DT_F64 result = NormL2SQRNoneImpl<Tp>(mat);
    return Sqrt(result);
}

template <typename Tp>
static Status NormNoneHelper(Context *ctx, const Mat &mat, DT_F64 *result, NormType type)
{
    Status ret = Status::OK;

    switch (type)
    {
        case NormType::NORM_INF:
        {
            *result = NormInfNoneImpl<Tp>(mat);
            break;
        }
        case NormType::NORM_L1:
        {
            *result = NormL1NoneImpl<Tp>(mat);
            break;
        }
        case NormType::NORM_L2:
        {
            *result = NormL2NoneImpl<Tp>(mat);
            break;
        }
        case NormType::NORM_L2SQR:
        {
            *result = NormL2SQRNoneImpl<Tp>(mat);
            break;
        }
        case NormType::NORM_MINMAX:
        {
            ret = Status::ERROR;
            AURA_ADD_ERROR_STRING(ctx, "NormType::NORM_MINMAX is used for normalize function.");
            break;
        }
        default:
        {
            ret = Status::ERROR;
            AURA_ADD_ERROR_STRING(ctx, "NormNoneHelper unsupported NormType.");
            break;
        }
    }

    AURA_RETURN(ctx, ret);
}

NormNone::NormNone(Context *ctx, const OpTarget &target) : NormImpl(ctx, target)
{}

Status NormNone::SetArgs(const Array *src, DT_F64 *result, NormType type)
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

Status NormNone::Run()
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
            ret = NormNoneHelper<DT_U8>(m_ctx, *src, m_result, m_type);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "NormNoneHelper<DT_U8> failed.");
            }
            break;
        }
        case ElemType::S8:
        {
            ret = NormNoneHelper<DT_S8>(m_ctx, *src, m_result, m_type);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "NormNoneHelper<DT_S8> failed.");
            }
            break;
        }
        case ElemType::U16:
        {
            ret = NormNoneHelper<DT_U16>(m_ctx, *src, m_result, m_type);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "NormNoneHelper<DT_U16> failed.");
            }
            break;
        }
        case ElemType::S16:
        {
            ret = NormNoneHelper<DT_S16>(m_ctx, *src, m_result, m_type);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "NormNoneHelper<DT_S16> failed.");
            }
            break;
        }
        case ElemType::U32:
        {
            ret = NormNoneHelper<DT_U32>(m_ctx, *src, m_result, m_type);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "NormNoneHelper<DT_U32> failed.");
            }
            break;
        }
        case ElemType::S32:
        {
            ret = NormNoneHelper<DT_S32>(m_ctx, *src, m_result, m_type);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "NormNoneHelper<DT_S32> failed.");
            }
            break;
        }
#if defined(AURA_BUILD_HOST)
        case ElemType::F16:
        {
            ret = NormNoneHelper<MI_F16>(m_ctx, *src, m_result, m_type);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "NormNoneHelper<MI_F16> failed.");
            }
            break;
        }
        case ElemType::F32:
        {
            ret = NormNoneHelper<DT_F32>(m_ctx, *src, m_result, m_type);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "NormNoneHelper<DT_F32> failed.");
            }
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

} // namespace aura