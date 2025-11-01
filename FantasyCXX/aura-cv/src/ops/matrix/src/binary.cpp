#include "aura/ops/matrix/binary.hpp"
#include "binary_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/tools/json.h"

namespace aura
{

static std::shared_ptr<BinaryImpl> CreateBinaryImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<BinaryImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new BinaryNone(ctx, target));
            break;
        }

        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            impl.reset(new BinaryNeon(ctx, target));
#endif // AURA_ENABLE_NEON
            break;
        }

        case TargetType::OPENCL:
        {
#if defined(AURA_ENABLE_OPENCL)
            impl.reset(new BinaryCL(ctx, target));
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

Binary::Binary(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status Binary::SetArgs(const Array *src0, const Array *src1, Array *dst, BinaryOpType type)
{
    if ((MI_NULL == m_ctx))
    {
        return Status::ERROR;
    }

    if ((MI_NULL == src0) || (MI_NULL == src1) || (MI_NULL == dst))
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
            if (CheckCLWidth(*src0) != Status::OK)
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
    if (MI_NULL == m_impl.get() || impl_target != m_impl->GetOpTarget())
    {
        m_impl = CreateBinaryImpl(m_ctx, impl_target);
    }

    // run initialize
    BinaryImpl *binary_impl = dynamic_cast<BinaryImpl*>(m_impl.get());
    if (MI_NULL == binary_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "binary_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = binary_impl->SetArgs(src0, src1, dst, type);

    AURA_RETURN(m_ctx, ret);
}

Status Binary::CLPrecompile(Context *ctx, ElemType elem_type, BinaryOpType op_type)
{
#if defined(AURA_ENABLE_OPENCL)
    if (MI_NULL == ctx)
    {
        return Status::ERROR;
    }

    std::vector<CLKernel> cl_kernels = BinaryCL::GetCLKernels(ctx, elem_type, op_type);

    if (CheckCLKernels(ctx, cl_kernels) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Binary CheckCLKernels failed");
        return Status::ERROR;
    }
#else
    AURA_UNUSED(ctx);
    AURA_UNUSED(elem_type);
    AURA_UNUSED(op_type);
#endif // AURA_ENABLE_OPENCL

    return Status::OK;
}

AURA_EXPORTS Status IMin(Context *ctx, const Mat &src0, const Mat &src1, Mat &dst, const OpTarget &target)
{
    Binary binary(ctx, target);

    return OpCall(ctx, binary, &src0, &src1, &dst, BinaryOpType::MIN);
}

AURA_EXPORTS Status IMax(Context *ctx, const Mat &src0, const Mat &src1, Mat &dst, const OpTarget &target)
{
    Binary binary(ctx, target);

    return OpCall(ctx, binary, &src0, &src1, &dst, BinaryOpType::MAX);
}

BinaryImpl::BinaryImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "Binary", target),
                                                               m_src0(MI_NULL), m_src1(MI_NULL),
                                                               m_dst(MI_NULL),  m_type(BinaryOpType::MIN)
{}

Status BinaryImpl::SetArgs(const Array *src0, const Array *src1, Array *dst, BinaryOpType type)
{
    if (MI_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if ((!src0->IsValid()) || (!src1->IsValid()) || (!dst->IsValid()))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst mat is invalid.");
        return Status::ERROR;
    }

    if (!src0->IsEqual(*src1) || !src0->IsEqual(*dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src0, src1 and dst must have same size and same type");
        return Status::ERROR;
    }

    m_src0 = src0;
    m_src1 = src1;
    m_dst  = dst;
    m_type = type;
    return Status::OK;
}

std::vector<const Array*> BinaryImpl::GetInputArrays() const
{
    return {m_src0, m_src1};
}

std::vector<const Array*> BinaryImpl::GetOutputArrays() const
{
    return {m_dst};
}

std::string BinaryImpl::ToString() const
{
    std::string str;

    str = "op(Binary)";
    str += " target(" + GetOpTarget().ToString() + ")\n";

    return str;
}

AURA_VOID BinaryImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    std::vector<std::string>  names  = {"src_0", "src_1", "dst"};
    std::vector<const Array*> arrays = {m_src0, m_src1, m_dst};

    if (json_wrapper.SetArray(names, arrays) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray src failed");
        return;
    }

    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src0, m_src1, m_dst, m_type);
}

} // namespace aura