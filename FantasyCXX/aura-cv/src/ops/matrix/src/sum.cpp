#include "aura/ops/matrix/sum.hpp"
#include "sum_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/tools/json.h"

namespace aura
{

static std::shared_ptr<SumImpl> CreateSumImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<SumImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new SumNone(ctx, target));
            break;
        }

        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            impl.reset(new SumNeon(ctx, target));
#endif // AURA_ENABLE_NEON
            break;
        }

        case TargetType::OPENCL:
        {
#if defined(AURA_ENABLE_OPENCL)
            impl.reset(new SumCL(ctx, target));
#endif // AURA_ENABLE_OPENCL
            break;
        }

        case TargetType::HVX:
        {
#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
            impl.reset(new SumHvx(ctx, target));
#endif // (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
            break;
        }

        default :
        {
            break;
        }
    }

    return impl;
}

Sum::Sum(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status Sum::SetArgs(const Array *src, Scalar &result)
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
        m_impl = CreateSumImpl(m_ctx, impl_target);
    }

    // run initialize
    SumImpl *sum_impl = dynamic_cast<SumImpl*>(m_impl.get());
    if (MI_NULL == sum_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "sum_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = sum_impl->SetArgs(src, &result);

    AURA_RETURN(m_ctx, ret);
}

Status Sum::CLPrecompile(Context *ctx, ElemType src_elem_type, ElemType dst_elem_type)
{
#if defined(AURA_ENABLE_OPENCL)
    if (MI_NULL == ctx)
    {
        return Status::ERROR;
    }

    std::vector<CLKernel> cl_kernels = SumCL::GetCLKernels(ctx, src_elem_type, dst_elem_type);

    if (CheckCLKernels(ctx, cl_kernels) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Sum CheckCLKernels failed");
        return Status::ERROR;
    }
#else
    AURA_UNUSED(ctx);
    AURA_UNUSED(src_elem_type);
    AURA_UNUSED(dst_elem_type);
#endif // AURA_ENABLE_OPENCL

    return Status::OK;
}

AURA_EXPORTS Status ISum(Context *ctx, const Mat &src, Scalar &result, const OpTarget &target)
{
    Sum sum(ctx, target);

    return OpCall(ctx, sum, &src, result);
}

static std::shared_ptr<SumImpl> CreateMeanImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<SumImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new MeanNone(ctx, target));
            break;
        }

        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            impl.reset(new MeanNeon(ctx, target));
#endif // AURA_ENABLE_NEON
            break;
        }

        case TargetType::OPENCL:
        {
#if defined(AURA_ENABLE_OPENCL)
            impl.reset(new MeanCL(ctx, target));
#endif // AURA_ENABLE_OPENCL
            break;
        }

        case TargetType::HVX:
        {
#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
            impl.reset(new MeanHvx(ctx, target));
#endif // (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
            break;
        }

        default :
        {
            break;
        }
    }

    return impl;
}

Mean::Mean(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status Mean::SetArgs(const Array *src, Scalar &result)
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
        m_impl = CreateMeanImpl(m_ctx, impl_target);
    }

    // run initialize
    SumImpl *sum_impl = dynamic_cast<SumImpl*>(m_impl.get());
    if (MI_NULL == sum_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "sum_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = sum_impl->SetArgs(src, &result);

    AURA_RETURN(m_ctx, ret);
}

AURA_EXPORTS Status IMean(Context *ctx, const Mat &src, Scalar &result, const OpTarget &target)
{
    Mean mean(ctx, target);

    return OpCall(ctx, mean, &src, result);
}

SumImpl::SumImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "Sum", target), m_result(MI_NULL),
                                                                         m_src(MI_NULL)
{}

Status SumImpl::SetArgs(const Array *src, Scalar *result)
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

    if (src->GetSizes().m_channel >= 4)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Sum only support channels <=3.");
        return Status::ERROR;
    }

    m_src = src;
    m_result = result;
    return Status::OK;
}

std::vector<const Array*> SumImpl::GetInputArrays() const
{
    return {m_src};
}

std::string SumImpl::ToString() const
{
    std::string str;

    str = "op(Sum)";
    str += " target(" + GetOpTarget().ToString() + ")\n";

    return str;
}

AURA_VOID SumImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    if (json_wrapper.SetArray("src", m_src) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray src failed");
        return;
    }

    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src, *m_result);
}

} // namespace aura