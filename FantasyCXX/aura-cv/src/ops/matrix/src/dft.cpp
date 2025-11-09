#include "aura/ops/matrix/dft.hpp"
#include "dft_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/tools/json.h"

namespace aura
{

static std::shared_ptr<DftImpl> CreateDftImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<DftImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new DftNone(ctx, target));
            break;
        }

        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            impl.reset(new DftNeon(ctx, target));
#endif // AURA_ENABLE_NEON
            break;
        }

        case TargetType::OPENCL:
        {
#if defined(AURA_ENABLE_OPENCL)
            impl.reset(new DftCL(ctx, target));
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

Dft::Dft(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status Dft::SetArgs(const Array *src, Array *dst)
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

        default:
        {
            break;
        }
    }

    // set m_impl
    if (DT_NULL == m_impl.get() || impl_target != m_impl->GetOpTarget())
    {
        m_impl = CreateDftImpl(m_ctx, impl_target);
    }

    // run initialize
    DftImpl *dft_impl = dynamic_cast<DftImpl*>(m_impl.get());
    if (DT_NULL == dft_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "dft_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = dft_impl->SetArgs(src, dst);

    AURA_RETURN(m_ctx, ret);
}

Status Dft::CLPrecompile(Context *ctx, ElemType src_elem_type, ElemType dst_elem_type)
{
#if defined(AURA_ENABLE_OPENCL)
    if (DT_NULL == ctx)
    {
        return Status::ERROR;
    }

    std::vector<CLKernel> cl_kernels = DftCL::GetCLKernels(ctx, src_elem_type, dst_elem_type);

    if (CheckCLKernels(ctx, cl_kernels) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Dft CheckCLKernels failed");
        return Status::ERROR;
    }
#else
    AURA_UNUSED(ctx);
    AURA_UNUSED(src_elem_type);
    AURA_UNUSED(dst_elem_type);
#endif // AURA_ENABLE_OPENCL

    return Status::OK;
}

AURA_EXPORTS Status IDft(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    Dft dft(ctx, target);

    return OpCall(ctx, dft, &src, &dst);
}

DftImpl::DftImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "Dft", target), m_src(DT_NULL), m_dst(DT_NULL)
{}

Status DftImpl::SetArgs(const Array *src, Array *dst)
{
    if (DT_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if (!src->IsValid() || !dst->IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst mat is invalid.");
        return Status::ERROR;
    }

    if (src->GetElemType() == ElemType::F64)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "current src does not support DT_F64 type.");
        return Status::ERROR;
    }

    if (dst->GetElemType() != ElemType::F32)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "current dst only support DT_F32 type.");
        return Status::ERROR;
    }

    Sizes3 src_sz = src->GetSizes();
    Sizes3 dst_sz = dst->GetSizes();

    if (src_sz.m_width != dst_sz.m_width || src_sz.m_height != dst_sz.m_height)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src and dst shape doesn't match.");
        return Status::ERROR;
    }

    if (src_sz.m_channel != 1 || dst_sz.m_channel != 2)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Dft only support real input with ch = 1 and complex output with ch = 2 ");
        return Status::ERROR;
    }

    m_src   = src;
    m_dst   = dst;
    return Status::OK;
}

std::vector<const Array*> DftImpl::GetInputArrays() const
{
    return {m_src};
}

std::vector<const Array*> DftImpl::GetOutputArrays() const
{
    return {m_dst};
}

std::string DftImpl::ToString() const
{
    std::string str;

    str = "op(Dft)";
    str += " target(" + GetOpTarget().ToString() + ")\n";

    return str;
}

DT_VOID DftImpl::Dump(const std::string &prefix) const
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

static std::shared_ptr<InverseDftImpl> CreateInverseDftImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<InverseDftImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new InverseDftNone(ctx, target));
            break;
        }

        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            impl.reset(new InverseDftNeon(ctx, target));
#endif // AURA_ENABLE_NEON
            break;
        }

        case TargetType::OPENCL:
        {
#if defined(AURA_ENABLE_OPENCL)
            impl.reset(new InverseDftCL(ctx, target));
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

InverseDft::InverseDft(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status InverseDft::SetArgs(const Array *src, Array *dst, DT_BOOL with_scale)
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

        default:
        {
            break;
        }
    }

    // set m_impl
    if (DT_NULL == m_impl.get() || impl_target != m_impl->GetOpTarget())
    {
        m_impl = CreateInverseDftImpl(m_ctx, impl_target);
    }

    // run initialize
    InverseDftImpl *idft_impl = dynamic_cast<InverseDftImpl*>(m_impl.get());
    if (DT_NULL == idft_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "idft_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = idft_impl->SetArgs(src, dst, with_scale);

    AURA_RETURN(m_ctx, ret);
}

Status InverseDft::CLPrecompile(Context *ctx, ElemType src_elem_type, ElemType dst_elem_type, DT_BOOL with_scale, DT_BOOL is_dst_c1)
{
#if defined(AURA_ENABLE_OPENCL)
    if (DT_NULL == ctx)
    {
        return Status::ERROR;
    }

    std::vector<CLKernel> cl_kernels = InverseDftCL::GetCLKernels(ctx, src_elem_type, dst_elem_type, with_scale, is_dst_c1);

    if (CheckCLKernels(ctx, cl_kernels) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "InverseDft CheckCLKernels failed");
        return Status::ERROR;
    }
#else
    AURA_UNUSED(ctx);
    AURA_UNUSED(src_elem_type);
    AURA_UNUSED(dst_elem_type);
    AURA_UNUSED(with_scale);
    AURA_UNUSED(is_dst_c1);
#endif // AURA_ENABLE_OPENCL

    return Status::OK;
}

AURA_EXPORTS Status IInverseDft(Context *ctx, const Mat &src, Mat &dst, DT_BOOL with_scale, const OpTarget &target)
{
    InverseDft idft(ctx, target);

    return OpCall(ctx, idft, &src, &dst, with_scale);
}

InverseDftImpl::InverseDftImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "InverseDft", target),
                                                                       m_src(DT_NULL), m_dst(DT_NULL),
                                                                       m_with_scale(0)
{}

Status InverseDftImpl::SetArgs(const Array *src, Array *dst, DT_BOOL with_scale)
{
    if (DT_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if ((DT_NULL == src) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is null");
        return Status::ERROR;
    }

    if (!src->IsValid() || !dst->IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst mat is invalid.");
        return Status::ERROR;
    }

    Sizes3 src_sz = src->GetSizes();
    Sizes3 dst_sz = dst->GetSizes();

    if ((src_sz.m_height != dst_sz.m_height) || (src_sz.m_width != dst_sz.m_width))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src and dst shape doesn't match.");
        return Status::ERROR;
    }

    if (src_sz.m_channel != 2 || (dst_sz.m_channel != 2 && dst_sz.m_channel != 1))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "InverseDft only support input with ch = 2 and output with ch = 2 or 1");
        return Status::ERROR;
    }

    if (ElemType::F32 != src->GetElemType())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "InverseDft only support input DT_F32 type.");
        return Status::ERROR;
    }

    if (ElemType::F64 == dst->GetElemType())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "currently dst does not support DT_F64 type.");
        return Status::ERROR;
    }

    if ((2 == dst_sz.m_channel) && (ElemType::F32 != dst->GetElemType()))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "if the dst is C2, only support DT_F32 type.");
        return Status::ERROR;
    }

    m_src        = src;
    m_dst        = dst;
    m_with_scale = with_scale;
    return Status::OK;
}

Status InverseDftImpl::Initialize()
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

    Sizes3 dst_size   = m_dst->GetSizes();
    Sizes3 dst_stride = m_dst->GetStrides();

    if ((1 == dst_size.m_channel) || (ElemType::F32 != m_dst->GetElemType()))
    {
        Sizes3 m_mid_size(dst_size.m_height, dst_size.m_width, 2);
        Sizes3 m_mid_stride;

        m_mid_stride.m_height = dst_stride.m_height;
        m_mid_stride.m_width  = dst_size.m_width * ElemTypeSize(ElemType::F32) * 2;

        if (IsPowOf2(dst_size.m_width))
        {
            m_mid_stride.m_width  = dst_size.m_width * ElemTypeSize(ElemType::F32) * 2 + 32;
        }

        m_mid = aura::Mat(m_ctx, ElemType::F32, m_mid_size, AURA_MEM_DEFAULT, m_mid_stride);
        if (!m_mid.IsValid())
        {
            AURA_ADD_ERROR_STRING(m_ctx, "Create m_mid mat failed.");
            return Status::ERROR;
        }
    }

    return Status::OK;
}

Status InverseDftImpl::DeInitialize()
{
    m_mid.Release();

    if (OpImpl::DeInitialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "OpImpl::DeInitialize() failed");
        return Status::ERROR;
    }

    return Status::OK;
}

std::vector<const Array*> InverseDftImpl::GetInputArrays() const
{
    return {m_src};
}

std::vector<const Array*> InverseDftImpl::GetOutputArrays() const
{
    return {m_dst};
}

std::string InverseDftImpl::ToString() const
{
    std::string str;

    str = "op(InverseDft)";
    str += " target(" + GetOpTarget().ToString() + ")\n";

    return str;
}

DT_VOID InverseDftImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    std::vector<std::string>  names  = {"src", "dst"};
    std::vector<const Array*> arrays = {m_src, m_dst};

    if (json_wrapper.SetArray(names, arrays) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray src failed");
        return;
    }

    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src, m_dst, m_with_scale);
}

} // namespace aura