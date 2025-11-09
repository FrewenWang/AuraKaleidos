#include "aura/ops/matrix/mul_spectrums.hpp"
#include "mul_spectrums_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/tools/json.h"

namespace aura
{

static std::shared_ptr<MulSpectrumsImpl> CreateMulSpectrumsImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<MulSpectrumsImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new MulSpectrumsNone(ctx, target));
            break;
        }

        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            impl.reset(new MulSpectrumsNeon(ctx, target));
#endif // AURA_ENABLE_NEON
            break;
        }

        case TargetType::OPENCL:
        {
#if defined(AURA_ENABLE_OPENCL)
            impl.reset(new MulSpectrumsCL(ctx, target));
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

MulSpectrums::MulSpectrums(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status MulSpectrums::SetArgs(const Array *src0, const Array *src1, Array *dst, DT_BOOL conj_src1)
{
    if ((DT_NULL == m_ctx))
    {
        return Status::ERROR;
    }

    if ((DT_NULL == src0) || (DT_NULL == src1) || (DT_NULL == dst))
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
            if (CheckNeonWidth(*src0) != Status::OK)
            {
                impl_target = OpTarget::None();
            }
#endif // AURA_ENABLE_NEON
            break;
        }

        case TargetType::OPENCL:
        {
#if defined(AURA_ENABLE_OPENCL)
            if (src0->GetSizes().m_width < 4)
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
        m_impl = CreateMulSpectrumsImpl(m_ctx, impl_target);
    }

    // run initialize
    MulSpectrumsImpl *mul_spectrums_impl = dynamic_cast<MulSpectrumsImpl*>(m_impl.get());
    if (DT_NULL == mul_spectrums_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "mul_spectrums_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = mul_spectrums_impl->SetArgs(src0, src1, dst, conj_src1);

    AURA_RETURN(m_ctx, ret);
}

Status MulSpectrums::CLPrecompile(Context *ctx, ElemType elem_type, DT_BOOL conj_src1)
{
#if defined(AURA_ENABLE_OPENCL)
    if (DT_NULL == ctx)
    {
        return Status::ERROR;
    }

    std::vector<CLKernel> cl_kernels = MulSpectrumsCL::GetCLKernels(ctx, elem_type, conj_src1);

    if (CheckCLKernels(ctx, cl_kernels) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "MulSpectrums CheckCLKernels failed");
        return Status::ERROR;
    }
#else
    AURA_UNUSED(ctx);
    AURA_UNUSED(elem_type);
    AURA_UNUSED(conj_src1);
#endif // AURA_ENABLE_OPENCL
    return Status::OK;
}

AURA_EXPORTS Status IMulSpectrums(Context *ctx, const Mat &src0, const Mat &src1, Mat &dst, DT_BOOL conj_src1, const OpTarget &target)
{
    MulSpectrums mul_spectrums(ctx, target);

    return OpCall(ctx, mul_spectrums, &src0, &src1, &dst, conj_src1);
}

MulSpectrumsImpl::MulSpectrumsImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "MulSpectrums", target),
                                                                           m_src0(DT_NULL), m_src1(DT_NULL), m_dst(DT_NULL)
{}

Status MulSpectrumsImpl::SetArgs(const Array *src0, const Array *src1, Array *dst, DT_BOOL conj_src1)
{
    if (DT_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if ((DT_NULL == src0) || (DT_NULL == src1) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src/dst is null ptr");
        return Status::ERROR;
    }

    if ((!src0->IsValid()) || (!src1->IsValid()) || (!dst->IsValid()))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst mat is invalid.");
        return Status::ERROR;
    }

    if (src0->GetElemType() != ElemType::F32 || src1->GetElemType() != ElemType::F32 ||
        dst->GetElemType() != ElemType::F32)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "current src and dst only support DT_F32 type.");
        return Status::ERROR;
    }

    if (!src0->IsEqual(*src1) || !src1->IsEqual(*dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src0, src1 and dst must have same size and same type");
        return Status::ERROR;
    }

    if (dst->GetSizes().m_channel != 2)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "currently only support complex input.");
        return Status::ERROR;
    }

    m_src0      = src0;
    m_src1      = src1;
    m_dst       = dst;
    m_conj_src1 = conj_src1;
    return Status::OK;
}

std::vector<const Array*> MulSpectrumsImpl::GetInputArrays() const
{
    return {m_src0, m_src1};
}

std::vector<const Array*> MulSpectrumsImpl::GetOutputArrays() const
{
    return {m_dst};
}

std::string MulSpectrumsImpl::ToString() const
{
    std::string str;

    str = "op(MulSpectrums)";
    str += " target(" + GetOpTarget().ToString() + ")\n";

    return str;
}

DT_VOID MulSpectrumsImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    std::vector<std::string>  names  = {"src_0", "src_1", "dst"};
    std::vector<const Array*> arrays = {m_src0, m_src1, m_dst};

    if (json_wrapper.SetArray(names, arrays) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray src failed");
        return;
    }

    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src0, m_src1, m_dst);
}

} // namespace aura