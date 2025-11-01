#include "aura/ops/resize/resize.hpp"
#include "resize_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/tools/json.h"

namespace aura
{

static std::shared_ptr<ResizeImpl> CreateResizeImpl(Context *ctx, const OpTarget &target,  InterpType type)
{
    std::shared_ptr<ResizeImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            AURA_UNUSED(type);
            impl.reset(new ResizeNone(ctx, target));
            break;
        }

        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            AURA_UNUSED(type);
            impl.reset(new ResizeNeon(ctx, target));
#endif // AURA_ENABLE_NEON
            break;
        }

        case TargetType::OPENCL:
        {
#if defined(AURA_ENABLE_OPENCL)
            switch (type)
            {
                case InterpType::NEAREST:
                {
                    impl.reset(new ResizeNnCL(ctx, target));
                    break;
                }
                case InterpType::LINEAR:
                {
                    impl.reset(new ResizeBnCL(ctx, target));
                    break;
                }
                case InterpType::CUBIC:
                {
                    impl.reset(new ResizeCuCL(ctx, target));
                    break;
                }
                case InterpType::AREA:
                {
                    impl.reset(new ResizeAreaCL(ctx, target));
                    break;
                }
                default:
                {
                    break;
                }
            }

#endif // AURA_ENABLE_OPENCL
            break;
        }

        case TargetType::HVX:
        {
#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
            impl.reset(new ResizeHvx(ctx, target));
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

Resize::Resize(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status Resize::SetArgs(const Array *src, Array *dst, InterpType type)
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
            if ((CheckNeonWidth(*src) != Status::OK) || (CheckNeonWidth(*dst) != Status::OK))
            {
                impl_target = OpTarget::None();
            }
#endif // AURA_ENABLE_NEON
            break;
        }

        case TargetType::OPENCL:
        {
#if defined(AURA_ENABLE_OPENCL)
            const Sizes3 &src_sz = src->GetSizes();
            const Sizes3 &dst_sz = dst->GetSizes();
            if ((Max(src_sz.m_width, src_sz.m_height)) < 64 && (Max(dst_sz.m_width, dst_sz.m_height) < 64))
            {
                impl_target = OpTarget::None();
            }
#endif // defined(AURA_ENABLE_OPENCL)
            break;
        }

        case TargetType::HVX:
        {
#if defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON)
            if ((CheckHvxWidth(*src) != Status::OK) || (CheckHvxWidth(*dst) != Status::OK))
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
        m_impl = CreateResizeImpl(m_ctx, impl_target, type);
    }

    // run initialize
    ResizeImpl *resize_impl = dynamic_cast<ResizeImpl*>(m_impl.get());
    if (MI_NULL == resize_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "resize_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = resize_impl->SetArgs(src, dst, type);

    AURA_RETURN(m_ctx, ret);
}

Status Resize::CLPrecompile(Context *ctx, ElemType elem_type, MI_S32 channel, MI_S32 iwidth, MI_S32 iheight, MI_S32 owidth, MI_S32 oheight,
                            InterpType interp_type)
{
#if defined(AURA_ENABLE_OPENCL)
    if (MI_NULL == ctx)
    {
        return Status::ERROR;
    }

    std::vector<CLKernel> cl_kernels;

    switch (interp_type)
    {
        case InterpType::NEAREST:
        {
            cl_kernels = ResizeNnCL::GetCLKernels(ctx, elem_type, channel);

            break;
        }
        case InterpType::LINEAR:
        {
            cl_kernels = ResizeBnCL::GetCLKernels(ctx, elem_type, channel);

            break;
        }
        case InterpType::CUBIC:
        {
            cl_kernels = ResizeCuCL::GetCLKernels(ctx, elem_type, iwidth, iheight, owidth, oheight, channel);
            break;
        }
        case InterpType::AREA:
        {
            cl_kernels = ResizeAreaCL::GetCLKernels(ctx, elem_type, iwidth, iheight, owidth, oheight, channel);
            break;
        }
        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "wrong interp type");
            return Status::ERROR;
        }
    }

    if (CheckCLKernels(ctx, cl_kernels) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "resize CheckCLKernels failed");
        return Status::ERROR;
    }
#else
    AURA_UNUSED(ctx);
    AURA_UNUSED(elem_type);
    AURA_UNUSED(channel);
    AURA_UNUSED(iwidth);
    AURA_UNUSED(iheight);
    AURA_UNUSED(owidth);
    AURA_UNUSED(oheight);
    AURA_UNUSED(interp_type);
#endif
    return Status::OK;
}

AURA_EXPORTS Status IResize(Context *ctx, const Mat &src, Mat &dst, InterpType type, const OpTarget &target)
{
    Resize resize(ctx, target);

    return OpCall(ctx, resize, &src, &dst, type);
}

ResizeImpl::ResizeImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "Resize", target),
                                                               m_src(MI_NULL), m_dst(MI_NULL),
                                                               m_type(InterpType::LINEAR)
{}

Status ResizeImpl::SetArgs(const Array *src, Array *dst, InterpType type)
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

    if (src->GetElemType() != dst->GetElemType())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src and dst must have the same elem type");
        return Status::ERROR;
    }

    if (src->GetElemType() == ElemType::S32 || src->GetElemType() == ElemType::U32 || src->GetElemType() == ElemType::F64)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "resize current only support elem type: U8/S8/U16/S16/F16/F32");
        return Status::ERROR;
    }

    if (!src->IsChannelEqual(*dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src and dst should have the same channel num");
        return Status::ERROR;
    }

    m_src = src;
    m_dst = dst;
    m_type = type;

    return Status::OK;
}

std::vector<const Array*> ResizeImpl::GetInputArrays() const
{
    return {m_src};
}

std::vector<const Array*> ResizeImpl::GetOutputArrays() const
{
    return {m_dst};
}

std::string ResizeImpl::ToString() const
{
    std::string str;

    str = "op(Resize)";
    str += " target(" + GetOpTarget().ToString() + ")";
    str += " param(" + InterpTypeToString(m_type) + ")\n";

    return str;
}

AURA_VOID ResizeImpl::Dump(const std::string &prefix) const
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