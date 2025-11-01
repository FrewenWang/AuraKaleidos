#include "aura/ops/matrix/norm.hpp"
#include "norm_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/tools/json.h"

namespace aura
{

static std::shared_ptr<NormImpl> CreateNormImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<NormImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new NormNone(ctx, target));
            break;
        }

        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            impl.reset(new NormNeon(ctx, target));
#endif // AURA_ENABLE_NEON
            break;
        }

        case TargetType::OPENCL:
        {
#if defined(AURA_ENABLE_OPENCL)
            impl.reset(new NormCL(ctx, target));
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

Norm::Norm(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status Norm::SetArgs(const Array *src, MI_F64 *result, NormType type)
{
    if ((MI_NULL == m_ctx))
    {
        return Status::ERROR;
    }

    if (MI_NULL == src)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src is null ptr");
        return Status::ERROR;
    }

    if (MI_NULL == result)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "result is null ptr");
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

        case TargetType::OPENCL:
        {
#if defined(AURA_ENABLE_OPENCL)
            if (CheckCLWidth(*src) != Status::OK)
            {
                impl_target = OpTarget::None();
            }
#endif // AURA_ENABLE_OPENCL
        }

        default:
        {
            break;
        }
    }

    // set m_impl
    if (MI_NULL == m_impl.get() || impl_target != m_impl->GetOpTarget())
    {
        m_impl = CreateNormImpl(m_ctx, impl_target);
    }

    // run initialize
    NormImpl *norm_impl = dynamic_cast<NormImpl*>(m_impl.get());
    if (MI_NULL == norm_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "norm_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = norm_impl->SetArgs(src, result, type);

    AURA_RETURN(m_ctx, ret);
}

Status Norm::CLPrecompile(Context *ctx, ElemType src_elem_type, ElemType dst_elem_type, NormType type)
{
#if defined(AURA_ENABLE_OPENCL)
    if (MI_NULL == ctx)
    {
        return Status::ERROR;
    }

    std::vector<CLKernel> cl_kernels = NormCL::GetCLKernels(ctx, src_elem_type, dst_elem_type, type);

    if (CheckCLKernels(ctx, cl_kernels) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Norm CheckCLKernels failed");
        return Status::ERROR;
    }
#else
    AURA_UNUSED(ctx);
    AURA_UNUSED(src_elem_type);
    AURA_UNUSED(dst_elem_type);
    AURA_UNUSED(type);
#endif // AURA_ENABLE_OPENCL

    return Status::OK;
}

AURA_EXPORTS Status INorm(Context *ctx, const Mat &src, MI_F64 *result, NormType type, const OpTarget &target)
{
    Norm norm(ctx, target);

    return OpCall(ctx, norm, &src, result, type);
}

NormImpl::NormImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "Norm", target), m_src(MI_NULL), 
                                                           m_result(MI_NULL), m_type(NormType::NORM_INF)
{}

Status NormImpl::SetArgs(const Array *src, MI_F64 *result, NormType type)
{
    if (MI_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if (MI_NULL == src)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src is null ptr");
        return Status::ERROR;
    }

    if (MI_NULL == result)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "result is null ptr");
        return Status::ERROR;
    }

    if (!src->IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst mat is invalid.");
        return Status::ERROR;
    }

    m_src = src;
    m_result = result;
    m_type = type;
    return Status::OK;
}

std::string NormImpl::ToString() const
{
    std::string str;

    str = "op(Norm)";
    str += " target(" + GetOpTarget().ToString() + ")\n";

    return str;
}

std::vector<const Array*> NormImpl::GetInputArrays() const
{
    return {m_src};
}

AURA_VOID NormImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    if (json_wrapper.SetArray("src", m_src) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray src failed");
        return;
    }

    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src, *m_result, m_type);
}

} // namespace aura