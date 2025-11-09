#include "aura/ops/matrix/rotate.hpp"
#include "rotate_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/tools/json.h"

namespace aura
{

static std::shared_ptr<RotateImpl> CreateRotateImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<RotateImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new RotateNone(ctx, target));
            break;
        }

        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            impl.reset(new RotateNeon(ctx, target));
#endif // AURA_ENABLE_NEON
            break;
        }

        case TargetType::OPENCL:
        {
#if defined(AURA_ENABLE_OPENCL)
            impl.reset(new RotateCL(ctx, target));
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
AURA_INLINE Status CheckRotateNeonParam(const Array *src)
{
    DT_S32 width  = src->GetSizes().m_width;
    DT_S32 height = src->GetSizes().m_height;
    if ((width < 8) || (height < 8))
    {
        return Status::ERROR;
    }
    return Status::OK;
}
#endif // AURA_ENABLE_NEON

Rotate::Rotate(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status Rotate::SetArgs(const Array *src, Array *dst, RotateType type)
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
            if (CheckRotateNeonParam(src) != Status::OK)
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
    if (DT_NULL == m_impl.get() || impl_target != m_impl->GetOpTarget())
    {
        m_impl = CreateRotateImpl(m_ctx, impl_target);
    }

    // run initialize
    RotateImpl *rotate_impl = dynamic_cast<RotateImpl*>(m_impl.get());
    if (DT_NULL == rotate_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "rotate_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = rotate_impl->SetArgs(src, dst, type);

    AURA_RETURN(m_ctx, ret);
}

Status Rotate::CLPrecompile(Context *ctx, ElemType elem_type, DT_S32 ochannel, RotateType type)
{
#if defined(AURA_ENABLE_OPENCL)
    if (DT_NULL == ctx)
    {
        return Status::ERROR;
    }

    std::vector<CLKernel> cl_kernels = RotateCL::GetCLKernels(ctx, elem_type, ochannel, type);

    if (CheckCLKernels(ctx, cl_kernels) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Rotate CheckCLKernels failed");
        return Status::ERROR;
    }
#else
    AURA_UNUSED(ctx);
    AURA_UNUSED(elem_type);
    AURA_UNUSED(ochannel);
    AURA_UNUSED(type);
#endif // AURA_ENABLE_OPENCL

    return Status::OK;
}

AURA_EXPORTS Status IRotate(Context *ctx, const Mat &src, Mat &dst, RotateType type, const OpTarget &target)
{
    Rotate rotate(ctx, target);

    return OpCall(ctx, rotate, &src, &dst, type);
}

RotateImpl::RotateImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "Rotate", target),
                                                               m_src(DT_NULL), m_dst(DT_NULL),
                                                               m_type(RotateType::ROTATE_180)
{}

Status RotateImpl::SetArgs(const Array *src, Array *dst, RotateType type)
{
    if (DT_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if ((DT_NULL == src) || (DT_NULL == dst))
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

    if (RotateType::ROTATE_90 == type || RotateType::ROTATE_270 == type)
    {
        Sizes3 src_sz = src->GetSizes();
        Sizes3 dst_sz = dst->GetSizes();

        if ((src_sz.m_width != dst_sz.m_height) || (src_sz.m_height != dst_sz.m_width) || (src_sz.m_channel != dst_sz.m_channel))
        {
            AURA_ADD_ERROR_STRING(m_ctx, "src and dst shape doesn't match");
            return Status::ERROR;
        }
    }
    else
    {
        if (!src->IsEqual(*dst))
        {
            AURA_ADD_ERROR_STRING(m_ctx, "failed, src and dst has different elem_type or size.");
            return Status::ERROR;
        }
    }

    m_src = src;
    m_dst = dst;
    m_type = type;
    return Status::OK;
}

std::vector<const Array*> RotateImpl::GetInputArrays() const
{
    return {m_src};
}

std::vector<const Array*> RotateImpl::GetOutputArrays() const
{
    return {m_dst};
}

std::string RotateImpl::ToString() const
{
    std::string str;

    str = "op(Rotate)";
    str += " target(" + GetOpTarget().ToString() + ")\n";

    return str;
}

DT_VOID RotateImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    std::vector<std::string>  names  = {"src", "dst"};
    std::vector<const Array*> arrays = {m_src, m_dst};

    if (json_wrapper.SetArray(names, arrays) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray src failed");
        return;
    }

    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src, m_dst, m_type);
}

} // namespace aura