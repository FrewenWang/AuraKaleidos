#include "aura/ops/matrix/convert_to.hpp"
#include "convert_to_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/tools/json.h"

namespace aura
{

static std::shared_ptr<ConvertToImpl> CreateConvertToImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<ConvertToImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new ConvertToNone(ctx, target));
            break;
        }

        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            impl.reset(new ConvertToNeon(ctx, target));
#endif // AURA_ENABLE_NEON
            break;
        }

        case TargetType::OPENCL:
        {
#if defined(AURA_ENABLE_OPENCL)
            impl.reset(new ConvertToCL(ctx, target));
#endif // AURA_ENABLE_OPENCL
            break;
        }

        default :
        {
            break;
        }
    }

    return impl;
}

ConvertTo::ConvertTo(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status ConvertTo::SetArgs(const Array *src, Array *dst, DT_F32 alpha, DT_F32 beta)
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
#endif// AURA_ENABLE_NEON
            break;
        }

        case TargetType::OPENCL:
        {
#if defined(AURA_ENABLE_OPENCL)
            if (CheckCLWidth(*src) != Status::OK)
            {
                impl_target = OpTarget::None();
            }
#endif// AURA_ENABLE_OPENCL
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
        m_impl = CreateConvertToImpl(m_ctx, impl_target);
    }

    // run initialize
    ConvertToImpl *convert_to_impl = dynamic_cast<ConvertToImpl*>(m_impl.get());
    if (DT_NULL == convert_to_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "convert_to_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = convert_to_impl->SetArgs(src, dst, alpha, beta);

    AURA_RETURN(m_ctx, ret);
}

Status ConvertTo::CLPrecompile(Context *ctx, ElemType src_elem_type, ElemType dst_elem_type, DT_F32 alpha, DT_F32 beta)
{

#if defined(AURA_ENABLE_OPENCL)
    if (DT_NULL == ctx)
    {
        return Status::ERROR;
    }

    DT_BOOL scale = (Abs(alpha - 1.0) > DBL_EPSILON) || (Abs(beta) > DBL_EPSILON);
    std::vector<CLKernel> cl_kernels = ConvertToCL::GetCLKernels(ctx, src_elem_type, dst_elem_type, scale);

    if (CheckCLKernels(ctx, cl_kernels) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "ConvertTo CheckCLKernels failed");
        return Status::ERROR;
    }
#else
    AURA_UNUSED(ctx);
    AURA_UNUSED(src_elem_type);
    AURA_UNUSED(dst_elem_type);
    AURA_UNUSED(alpha);
    AURA_UNUSED(beta);
#endif // AURA_ENABLE_OPENCL

    return Status::OK;
}

AURA_EXPORTS Status IConvertTo(Context *ctx, const Mat &src, Mat &dst, DT_F32 alpha, DT_F32 beta, const OpTarget &target)
{
    ConvertTo convert_to(ctx, target);

    return OpCall(ctx, convert_to, &src, &dst, alpha, beta);
}

ConvertToImpl::ConvertToImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "ConvertTo", target),
                                                                     m_src(DT_NULL), m_dst(DT_NULL),
                                                                     m_alpha(1), m_beta(0)
{}

Status ConvertToImpl::SetArgs(const Array *src, Array *dst, DT_F32 alpha, DT_F32 beta)
{
    if (DT_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if (!src->IsValid() || !dst->IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid src or dst");
        return Status::ERROR;
    }

    if (!src->IsSizesEqual(*dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "sizes of src or dst mat dismatch");
        return Status::ERROR;
    }

    m_src   = src;
    m_dst   = dst;
    m_alpha = alpha;
    m_beta  = beta;

    return Status::OK;
}

std::vector<const Array*> ConvertToImpl::GetInputArrays() const
{
    return {m_src};
}

std::vector<const Array*> ConvertToImpl::GetOutputArrays() const
{
    return {m_dst};
}

std::string ConvertToImpl::ToString() const
{
    std::string str;

    str = "op(ConvertTo)";
    str += " target(" + GetOpTarget().ToString() + ")";
    str += " alpha(" + std::to_string(m_alpha) + ")";
    str += " beta(" + std::to_string(m_beta) + ")\n";

    return str;
}

DT_VOID ConvertToImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    std::vector<std::string>  names  = {"src", "dst"};
    std::vector<const Array*> arrays = {m_src, m_dst};

    if (json_wrapper.SetArray(names, arrays) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray src failed");
        return;
    }

    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src, m_dst, m_alpha, m_beta);
}

} // namespace aura