#include "aura/ops/filter/gaussian.hpp"
#include "gaussian_impl.hpp"
#include "aura/runtime/logger.h"

namespace aura
{

template <typename Tp>
struct GaussianTraits
{
    // Tp = DT_F32, DT_U8 DT_U16 DT_S16 MI_F16
    using KernelType = typename std::conditional<sizeof(Tp) == 4, Tp, typename Promote<Tp>::Type>::type;
    // Tp = MI_F16 DT_F32, DT_U8, DT_U16 DT_S16
    static constexpr DT_U32 Q = is_floating_point<Tp>::value ? 0 : (sizeof(Tp) == 2 ? 14 : 8);
};


GaussianNeon::GaussianNeon(Context *ctx, const OpTarget &target) : GaussianImpl(ctx, target)
{}

Status GaussianNeon::SetArgs(const Array *src, Array *dst, DT_S32 ksize, DT_F32 sigma,
                             BorderType border_type, const Scalar &border_value)
{
    if (GaussianImpl::SetArgs(src, dst, ksize, sigma, border_type, border_value) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GaussianImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    if (ksize != 3 && ksize != 5 && ksize != 7)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ksize is only suppose 3/5/7");
        return Status::ERROR;
    }

    DT_S32 ch = src->GetSizes().m_channel;

    if (ch != 1 && ch != 2 && ch != 3)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "channel is only suppose 1/2/3");
        return Status::ERROR;
    }

    return Status::OK;
}

Status GaussianNeon::PrepareKmat()
{
    std::vector<DT_F32> kernel = GetGaussianKernel(m_ksize, m_sigma);

#define GET_GAUSSIAN_KMAT(type)                                     \
    using KernelType   = typename GaussianTraits<type>::KernelType; \
    constexpr DT_U32 Q = GaussianTraits<type>::Q;                   \
                                                                    \
    m_kmat = GetGaussianKmat<KernelType, Q>(m_ctx, kernel);         \

    switch (m_src->GetElemType())
    {
        case ElemType::U8:
        {
            GET_GAUSSIAN_KMAT(DT_U8)
            break;
        }

        case ElemType::U16:
        {
            GET_GAUSSIAN_KMAT(DT_U16)
            break;
        }

        case ElemType::S16:
        {
            GET_GAUSSIAN_KMAT(DT_S16)
            break;
        }

        case ElemType::F32:
        {
            m_kmat = GetGaussianKmat(m_ctx, kernel);
            break;
        }

#if defined(AURA_BUILD_HOST)
        case ElemType::F16:
        {
            m_kmat = GetGaussianKmat(m_ctx, kernel);
            break;
        }
#endif // AURA_BUILD_HOST

        default:
        {
            m_kmat = Mat();
            AURA_ADD_ERROR_STRING(m_ctx, "Unsupported source format");
            return Status::ERROR;
        }
    }

#undef GET_GAUSSIAN_KMAT

    return Status::OK;
}

Status GaussianNeon::Initialize()
{
    if (GaussianImpl::Initialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GaussianImpl::Initialize() failed");
        return Status::ERROR;
    }

    // Prepare kmat
    if (PrepareKmat() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "PrepareKmat failed");
        return Status::ERROR;
    }

    return Status::OK;
}

Status GaussianNeon::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst       = dynamic_cast<Mat*>(m_dst);

    if ((DT_NULL == src) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst is null");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    switch (m_ksize)
    {
        case 3:
        {
            ret = Gaussian3x3Neon(m_ctx, *src, *dst, m_kmat, m_border_type, m_border_value, m_target);
            break;
        }
        case 5:
        {
            ret = Gaussian5x5Neon(m_ctx, *src, *dst, m_kmat, m_border_type, m_border_value, m_target);
            break;
        }
        case 7:
        {
            ret = Gaussian7x7Neon(m_ctx, *src, *dst, m_kmat, m_border_type, m_border_value, m_target);
            break;
        }
        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "Unsuported ksize");
            return Status::ERROR;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

Status GaussianNeon::DeInitialize()
{
    m_kmat.Release();

    if (GaussianImpl::DeInitialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GaussianImpl::DeInitialize() failed");
        return Status::ERROR;
    }

    return Status::OK;
}

} // namespace aura