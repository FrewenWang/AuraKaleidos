#include "aura/ops/matrix/transpose.hpp"
#include "transpose_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/tools/json.h"

namespace aura
{

static std::shared_ptr<TransposeImpl> CreateTransposeImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<TransposeImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new TransposeNone(ctx, target));
            break;
        }

        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            impl.reset(new TransposeNeon(ctx, target));
#endif // AURA_ENABLE_NEON
            break;
        }

        case TargetType::OPENCL:
        {
#if defined(AURA_ENABLE_OPENCL)
            impl.reset(new TransposeCL(ctx, target));
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

#if defined(AURA_ENABLE_NEON)
AURA_INLINE Status CheckTransposeNeonParam(const Array *src)
{
    MI_S32 width  = src->GetSizes().m_width;
    MI_S32 height = src->GetSizes().m_height;
    if ((width < 8) || (height < 8))
    {
        return Status::ERROR;
    }
    return Status::OK;
}
#endif // AURA_ENABLE_NEON

Transpose::Transpose(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status Transpose::SetArgs(const Array *src, Array *dst)
{
    if ((MI_NULL == m_ctx))
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
            if (CheckTransposeNeonParam(src) != Status::OK)
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
        m_impl = CreateTransposeImpl(m_ctx, impl_target);
    }

    // run initialize
    TransposeImpl *transpose_impl = dynamic_cast<TransposeImpl*>(m_impl.get());
    if (MI_NULL == transpose_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "transpose_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = transpose_impl->SetArgs(src, dst);

    AURA_RETURN(m_ctx, ret);
}

Status Transpose::CLPrecompile(Context *ctx, ElemType elem_type, MI_S32 ochannel)
{
#if defined(AURA_ENABLE_OPENCL)
    if (MI_NULL == ctx)
    {
        return Status::ERROR;
    }

    std::vector<CLKernel> cl_kernels = TransposeCL::GetCLKernels(ctx, elem_type, ochannel);

    if (CheckCLKernels(ctx, cl_kernels) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Gaussian CheckCLKernels failed");
        return Status::ERROR;
    }
#else
    AURA_UNUSED(ctx);
    AURA_UNUSED(elem_type);
    AURA_UNUSED(ochannel);
#endif
    return Status::OK;
}

AURA_EXPORTS Status ITranspose(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    Transpose transpose(ctx, target);

    return OpCall(ctx, transpose, &src, &dst);
}

TransposeImpl::TransposeImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "Transpose", target),
                                                                     m_src(MI_NULL), m_dst(MI_NULL)
{}

Status TransposeImpl::SetArgs(const Array *src, Array *dst)
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

    if (!src->IsValid() || !dst->IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst mat is invalid.");
        return Status::ERROR;
    }

    if (src->GetElemType() != dst->GetElemType())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst has different mem type.");
        return Status::ERROR;
    }

    Sizes3 src_sz = src->GetSizes();
    Sizes3 dst_sz = dst->GetSizes();

    if ((src_sz.m_width != dst_sz.m_height) || (src_sz.m_height != dst_sz.m_width) || (src_sz.m_channel != dst_sz.m_channel))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src and dst shape doesn't match");
        return Status::ERROR;
    }

    m_src = src;
    m_dst = dst;
    return Status::OK;
}

std::vector<const Array*> TransposeImpl::GetInputArrays() const
{
    return {m_src};
}

std::vector<const Array*> TransposeImpl::GetOutputArrays() const
{
    return {m_dst};
}

std::string TransposeImpl::ToString() const
{
    std::string str;

    str = "op(Transpose)";
    str += " target(" + GetOpTarget().ToString() + ")\n";

    return str;
}

AURA_VOID TransposeImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    std::vector<std::string>  names  = {"src", "dst"};
    std::vector<const Array*> arrays = {m_src, m_dst};

    if (json_wrapper.SetArray(names, arrays) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray src failed");
        return;
    }

    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src, m_dst);
}

} // namespace aura