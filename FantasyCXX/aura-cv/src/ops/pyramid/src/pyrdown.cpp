#include "aura/ops/pyramid/pyrdown.hpp"
#include "pyramid_comm.hpp"
#include "pyrdown_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/ops/filter/gaussian.hpp"
#include "aura/tools/json.h"

namespace aura
{

static std::shared_ptr<PyrDownImpl> CreatePyrDownImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<PyrDownImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new PyrDownNone(ctx, target));
            break;
        }

        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            impl.reset(new PyrDownNeon(ctx, target));
#endif // AURA_ENABLE_NEON
            break;
        }

        case TargetType::OPENCL:
        {
#if defined(AURA_ENABLE_OPENCL)
            impl.reset(new PyrDownCL(ctx, target));
#endif // AURA_ENABLE_OPENCL
            break;
        }

        case TargetType::HVX:
        {
#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
            impl.reset(new PyrDownHvx(ctx, target));
#endif // (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
            break;
        }

        default:
        {
            break;
        }
    }

    return impl;
}

PyrDown::PyrDown(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status PyrDown::SetArgs(const Array *src, Array *dst, DT_S32 ksize, DT_F32 sigma, BorderType border_type)
{
    if ((DT_NULL == m_ctx))
    {
        return Status::ERROR;
    }

    if ((DT_NULL == src) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src/dst is null ptr");
        return Status::ERROR;
    }

    OpTarget impl_target = m_target;

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

        case TargetType::OPENCL:
        {
#if defined(AURA_ENABLE_OPENCL)
            if (CheckCLWidth(*src) != Status::OK)
            {
                impl_target = OpTarget::None();
            }
#endif // AURA_ENABLE_OPENCL
            break;
        }

        case TargetType::HVX:
        {
#if defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON)
            if (CheckHvxWidth(*dst) != Status::OK)
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
    if (DT_NULL == m_impl.get() || impl_target != m_impl->GetOpTarget())
    {
        m_impl = CreatePyrDownImpl(m_ctx, impl_target);
    }

    // run initialize
    PyrDownImpl *pyrdown_impl = dynamic_cast<PyrDownImpl*>(m_impl.get());
    if (DT_NULL == pyrdown_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "pyrdown_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = pyrdown_impl->SetArgs(src, dst, ksize, sigma, border_type);

    AURA_RETURN(m_ctx, ret);
}

Status PyrDown::CLPrecompile(Context *ctx, ElemType elem_type, DT_S32 channel, DT_S32 ksize, BorderType border_type)
{
#if defined(AURA_ENABLE_OPENCL)
    if (DT_NULL == ctx)
    {
        return Status::ERROR;
    }

    std::vector<CLKernel> cl_kernels = PyrDownCL::GetCLKernels(ctx, elem_type, channel, ksize, border_type);

    if (CheckCLKernels(ctx, cl_kernels) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "PyrDown CheckCLKernels failed");
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

AURA_EXPORTS Status IPyrDown(Context *ctx, const Mat &src, Mat &dst, DT_S32 ksize, DT_F32 sigma, BorderType border_type,
                             const OpTarget &target)
{
    PyrDown pyrdown(ctx, target);

    return OpCall(ctx, pyrdown, &src, &dst, ksize, sigma, border_type);
}

PyrDownImpl::PyrDownImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "PyrDown", target),
                                                                 m_ksize(0), m_sigma(0.f), m_border_type(BorderType::REFLECT_101),
                                                                 m_src(DT_NULL), m_dst(DT_NULL)
{}

Status PyrDownImpl::SetArgs(const Array *src, Array *dst, DT_S32 ksize, DT_F32 sigma, BorderType border_type)
{
    if (DT_NULL == m_ctx)
    {
        return Status::ERROR;
    }
    if (!(src->IsValid() && dst->IsValid()))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid src or dst");
        return Status::ERROR;
    }

    if ((src->GetSizes().m_height < ksize) || (src->GetSizes().m_width < ksize))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "height/width must bigger than ksize");
        return Status::ERROR;
    }

    if (src->GetElemType() != dst->GetElemType())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src and dst elemtype is not same");
        return Status::ERROR;
    }

    if ((src->GetElemType() != ElemType::U8) && (src->GetElemType() != ElemType::S16) && (src->GetElemType() != ElemType::U16))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "elem_type only support u8/u16/s16");
        return Status::ERROR;
    }

    const Sizes3 &src_sizes = src->GetSizes();
    const Sizes3 &dst_sizes = dst->GetSizes();
    if ((src_sizes.m_channel            != dst_sizes.m_channel) || (src_sizes.m_channel         != 1) ||
        ((src_sizes.m_height + 1) >> 1) != dst_sizes.m_height || ((src_sizes.m_width + 1) >> 1) != dst_sizes.m_width)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src and dst sizes is not right");
        return Status::ERROR;
    }

    if (ksize != 5)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ksize only support 5");
        return Status::ERROR;
    }

    if ((border_type != BorderType::REFLECT_101) && (border_type != BorderType::REPLICATE))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Invalid border type");
        return Status::ERROR;
    }

    m_src         = src;
    m_dst         = dst;
    m_ksize       = ksize;
    m_sigma       = sigma;
    m_border_type = border_type;

    return Status::OK;
}

Status PyrDownImpl::Initialize()
{
    if (DT_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if (OpImpl::Initialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "OpImpl::Initialize() failed");
        return Status::ERROR;
    }

    return PrepareKmat();
}

Status PyrDownImpl::DeInitialize()
{
    m_kmat.Release();

    if (OpImpl::DeInitialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "OpImpl::DeInitialize() failed");
        return Status::ERROR;
    }

    return Status::OK;
}

std::vector<const Array*> PyrDownImpl::GetInputArrays() const
{
    return {m_src};
}

std::vector<const Array*> PyrDownImpl::GetOutputArrays() const
{
    return {m_dst};
}

std::string PyrDownImpl::ToString() const
{
    std::string str;

    DT_CHAR sigma_str[20];
    snprintf(sigma_str, sizeof(sigma_str), "%.2f", m_sigma);

    str = "op(PyrDown)";
    str += " target(" + GetOpTarget().ToString() + ")";
    str += " param(" + BorderTypeToString(m_border_type) + " | " +
           "ksize:" + std::to_string(m_ksize) + " | " + "sigma:" + sigma_str + ")\n";
    return str;
}

DT_VOID PyrDownImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    std::vector<std::string>  names  = {"src", "dst"};
    std::vector<const Array*> arrays = {m_src, m_dst};

    if (json_wrapper.SetArray(names, arrays) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray src failed");
        return;
    }

    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src, m_dst, m_ksize, m_sigma, m_border_type);
}

Status PyrDownImpl::PrepareKmat()
{
    if (DT_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    std::vector<DT_F32> kernel = GetGaussianKernel(m_ksize, m_sigma);

#define GET_PYRDOWN_KMAT(type)                                     \
    using KernelType   = typename PyrDownTraits<type>::KernelType; \
    constexpr DT_U32 Q = PyrDownTraits<type>::Q;                   \
                                                                   \
    m_kmat = GetPyrKernelMat<KernelType, Q>(m_ctx, kernel);        \

    switch (m_src->GetElemType())
    {
        case ElemType::U8:
        {
            GET_PYRDOWN_KMAT(DT_U8)
            break;
        }

        case ElemType::U16:
        {
            GET_PYRDOWN_KMAT(DT_U16)
            break;
        }

        case ElemType::S16:
        {
            GET_PYRDOWN_KMAT(DT_S16)
            break;
        }

        default:
        {
            m_kmat = Mat();
            AURA_ADD_ERROR_STRING(m_ctx, "Unsupported source format");
            return Status::ERROR;
        }
    }

#undef GET_PYRDOWN_KMAT

    return Status::OK;
}

} // namespace aura