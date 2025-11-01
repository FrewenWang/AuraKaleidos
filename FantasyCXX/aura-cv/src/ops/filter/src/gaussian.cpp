#include "aura/ops/filter/gaussian.hpp"
#include "gaussian_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/tools/json.h"

namespace aura
{

static std::shared_ptr<GaussianImpl> CreateGaussianImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<GaussianImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new GaussianNone(ctx, target));
            break;
        }

        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            impl.reset(new GaussianNeon(ctx, target));
#endif // AURA_ENABLE_NEON
            break;
        }

        case TargetType::OPENCL:
        {
#if defined(AURA_ENABLE_OPENCL)
            impl.reset(new GaussianCL(ctx, target));
#endif // AURA_ENABLE_OPENCL
            break;
        }

        case TargetType::HVX:
        {
#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
            impl.reset(new GaussianHvx(ctx, target));
#endif // (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
            break;
        }

        case TargetType::VDSP:
        {
#if defined(AURA_ENABLE_XTENSA)
            impl.reset(new GaussianVdsp(ctx, target));
#endif // defined(AURA_ENABLE_XTENSA)
            break;
        }

        default :
        {
            break;
        }
    }

    return impl;
}

Gaussian::Gaussian(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status Gaussian::SetArgs(const Array *src, Array *dst, MI_S32 ksize, MI_F32 sigma,
                         BorderType border_type, const Scalar &border_value)
{
    if (MI_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if ((MI_NULL == src) || (MI_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src/dst is null ptr");
        return Status::ERROR;
    }

    OpTarget impl_target = m_target;

    // check impl type
    switch (m_target.m_type)
    {
        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            if (CheckNeonWidth(*src) != Status::OK)
            {
                impl_target = OpTarget::None();
            }
#endif // AURA_ENABLE_NEON
            break;
        }

        case TargetType::HVX:
        {
#if defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON)
            if (CheckHvxWidth(*src) != Status::OK)
            {
                impl_target = OpTarget::None();
            }
#endif // defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON)
            break;
        }

        default:
        {
            break;
        }
    }

    // set m_impl
    if (MI_NULL == m_impl.get() || impl_target != m_impl->GetOpTarget())
    {
        m_impl = CreateGaussianImpl(m_ctx, impl_target);
    }

    // run initialize
    GaussianImpl *gaussian_impl = dynamic_cast<GaussianImpl*>(m_impl.get());
    if (MI_NULL == gaussian_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "gaussian_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = gaussian_impl->SetArgs(src, dst, ksize, sigma, border_type, border_value);

    AURA_RETURN(m_ctx, ret);
}

Status Gaussian::CLPrecompile(Context *ctx, ElemType elem_type, MI_S32 channel, MI_S32 ksize, BorderType border_type)
{
#if defined(AURA_ENABLE_OPENCL)
    if (MI_NULL == ctx)
    {
        return Status::ERROR;
    }
    // 获取不同的kernel
    std::vector<CLKernel> cl_kernels = GaussianCL::GetCLKernels(ctx, elem_type, channel, ksize, border_type);

    if (CheckCLKernels(ctx, cl_kernels) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Gaussian CheckCLKernels failed");
        return Status::ERROR;
    }
#else
    AURA_UNUSED(ctx);
        AURA_UNUSED(elem_type);
    AURA_UNUSED(channel);
    AURA_UNUSED(ksize);
        AURA_UNUSED(border_type);
#endif
    return Status::OK;
}

AURA_EXPORTS Status IGaussian(Context *ctx, const Mat &src, Mat &dst, MI_S32 ksize, MI_F32 sigma,
                              BorderType border_type, const Scalar &border_value, const OpTarget &target)
{
    // 实例化高斯模糊的对象
    Gaussian gaussian(ctx, target);
    // 调用对应的Op算子
    return OpCall(ctx, gaussian, &src, &dst, ksize, sigma, border_type, border_value);
}

GaussianImpl::GaussianImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, AURA_OPS_FILTER_GAUSSIAN_OP_NAME, target),
                                                                   m_ksize(0), m_sigma(0.f), m_border_type(BorderType::REFLECT_101),
                                                                   m_src(MI_NULL), m_dst(MI_NULL)
{}

Status GaussianImpl::SetArgs(const Array *src, Array *dst, MI_S32 ksize, MI_F32 sigma,
                             BorderType border_type, const Scalar &border_value)
{
    if (MI_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if (!(src->IsValid() && dst->IsValid()))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid src or dst");
        return Status::ERROR;
    }

    if (!src->IsEqual(*dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src and dst should have the same size");
        return Status::ERROR;
    }

    if ((ksize & 1) != 1)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ksize must is odd");
        return Status::ERROR;
    }

    if ((src->GetSizes().m_height < ksize) || (src->GetSizes().m_width < ksize))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "height/width must bigger than ksize");
        return Status::ERROR;
    }

    m_src          = src;
    m_dst          = dst;
    m_ksize        = ksize;
    m_sigma        = sigma;
    m_border_type  = border_type;
    m_border_value = border_value;

    return Status::OK;
}

std::vector<const Array*> GaussianImpl::GetInputArrays() const
{
    return {m_src};
}

std::vector<const Array*> GaussianImpl::GetOutputArrays() const
{
    return {m_dst};
}

std::string GaussianImpl::ToString() const
{
    std::string str;

    MI_CHAR sigma_str[20];
    snprintf(sigma_str, sizeof(sigma_str), "%.2f", m_sigma);

    str = "op(Gaussian)";
    str += " target(" + GetOpTarget().ToString() + ")";
    str += " param(" + BorderTypeToString(m_border_type) + " | " +
            "ksize:" + std::to_string(m_ksize) + " | " + "sigma:" + sigma_str + " | "
            "border_value:" + m_border_value.ToString() + ")\n";

    return str;
}

AURA_VOID GaussianImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    std::vector<std::string>  names  = {"src", "dst"};
    std::vector<const Array*> arrays = {m_src, m_dst};

    if (json_wrapper.SetArray(names, arrays) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray src failed");
        return;
    }

    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src, m_dst, m_ksize, m_sigma, m_border_type, m_border_value);
}

std::vector<MI_F32> GetGaussianKernel(MI_S32 ksize, MI_F32 sigma)
{
    std::vector<MI_F32> kernel(ksize, 0);
    constexpr MI_S32 small_gaussian_size = 7;
    constexpr MI_F32 small_gaussian_tab[][small_gaussian_size] =
    {
        {1.f},
        {0.25f, 0.5f, 0.25f},
        {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f},
        {0.03125f, 0.109375f, 0.21875f, 0.28125f, 0.21875f, 0.109375f, 0.03125f}
    };

    if ((ksize <= small_gaussian_size) && (sigma <= 0))
    {
        const MI_F32 *t_ptr = small_gaussian_tab[ksize >> 1];
        for (MI_S32 i = 0; i < ksize; i++)
        {
            kernel[i] = t_ptr[i];
        }
        return kernel;
    }

    std::vector<MI_F32> vec_kernel(ksize, 0);
    MI_F64 sigma_value = sigma > 0 ? static_cast<MI_F64>(sigma) : ((ksize - 1) * 0.5 - 1) * 0.3 + 0.8;
    MI_F64 sigma2 = -0.5 / (sigma_value * sigma_value);
    MI_F64 sum = 0;

    for (MI_S32 i = 0; i < ksize; i++)
    {
        MI_F64 x = i - (ksize - 1) * 0.5;
        vec_kernel[i] = static_cast<MI_F32>(Exp(sigma2 * x * x));
        sum += vec_kernel[i];
    }

    sum = 1.0 / sum;

    for (MI_S32 i = 0; i < ksize; i++)
    {
        kernel[i] = static_cast<MI_F32>(vec_kernel[i] * sum);
    }

    return kernel;
}

} // namespace aura