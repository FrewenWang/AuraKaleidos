#include "aura/ops/matrix/arithmetic.hpp"
#include "aura/ops/matrix/convert_to.hpp"
#include "arithmetic_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/tools/json.h"

namespace aura
{

static std::shared_ptr<ArithmeticImpl> CreateArithmeticImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<ArithmeticImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new ArithmeticNone(ctx, target));
            break;
        }

        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            impl.reset(new ArithmeticNeon(ctx, target));
#endif // AURA_ENABLE_NEON
            break;
        }

        case TargetType::OPENCL:
        {
#if defined(AURA_ENABLE_OPENCL)
            impl.reset(new ArithmeticCL(ctx, target));
#endif // AURA_ENABLE_OPENCL
            break;
        }

        case TargetType::HVX:
        {
#if defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON)
            impl.reset(new ArithmeticHvx(ctx, target));
#endif // (AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON)
            break;
        }

        default :
        {
            break;
        }
    }

    return impl;
}

Arithmetic::Arithmetic(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status Arithmetic::CLPrecompile(Context *ctx, ElemType src_elem_type, ElemType dst_elem_type, ArithmOpType op_type)
{
#if defined(AURA_ENABLE_OPENCL)
    if (MI_NULL == ctx)
    {
        return Status::ERROR;
    }

    std::vector<CLKernel> cl_kernels = ArithmeticCL::GetCLKernels(ctx, src_elem_type, dst_elem_type, op_type);

    if (CheckCLKernels(ctx, cl_kernels) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Arithmetic CheckCLKernels failed");
        return Status::ERROR;
    }
#else
    AURA_UNUSED(ctx);
    AURA_UNUSED(src_elem_type);
    AURA_UNUSED(dst_elem_type);
    AURA_UNUSED(op_type);
#endif // AURA_ENABLE_OPENCL

    return Status::OK;
}

Status Arithmetic::SetArgs(const Array *src0, const Array *src1, Array *dst, ArithmOpType op)
{
    if (MI_NULL == m_ctx)
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
        case TargetType::HVX:
        {
#if defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON)
            if (CheckHvxWidth(*src0) != Status::OK)
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
        m_impl = CreateArithmeticImpl(m_ctx, impl_target);
    }

    ArithmeticImpl *arithmetic_impl = dynamic_cast<ArithmeticImpl*>(m_impl.get());
    if (MI_NULL == arithmetic_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "arithmetic_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = arithmetic_impl->SetArgs(src0, src1, dst, op);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "arithmetic_impl SetArgs failed.");
        return Status::ERROR;
    }

    return ret;
}

static std::shared_ptr<ScalarDivideImpl> CreateScalarDivideImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<ScalarDivideImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new ScalarDivideNone(ctx, target));
            break;
        }

        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            impl.reset(new ScalarDivideNeon(ctx, target));
#endif // AURA_ENABLE_NEON
            break;
        }

        default:
        {
            break;
        }
    }

    return impl;
}

ScalarDivide::ScalarDivide(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status ScalarDivide::SetArgs(MI_F32 scalar, const Array *src, Array *dst)
{
    Status ret = Status::ERROR;

    if (MI_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if ((MI_NULL == src) || (MI_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src/dst is null ptr");
        return Status::ERROR;
    }

    OpTarget impl_target = m_target;

    // set m_impl
    if (MI_NULL == m_impl.get() || impl_target != m_impl->GetOpTarget())
    {
        m_impl = CreateScalarDivideImpl(m_ctx, impl_target);
    }

    ScalarDivideImpl *scalar_divide_impl = dynamic_cast<ScalarDivideImpl*>(m_impl.get());
    if (MI_NULL == scalar_divide_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "scalar_divide_impl is null ptr");
        return Status::ERROR;
    }

    ret = scalar_divide_impl->SetArgs(scalar, src, dst);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "scalar_divide_impl SetArgs failed.");
        return Status::ERROR;
    }

    return ret;
}

AURA_EXPORTS Status IAdd(Context *ctx, const Mat &src0, const Mat &src1, Mat &dst, const OpTarget &target)
{
    Arithmetic arithmetic(ctx, target);

    return OpCall(ctx, arithmetic, &src0, &src1, &dst, ArithmOpType::ADD);
}

AURA_EXPORTS Status ISubtract(Context *ctx, const Mat &src0, const Mat &src1, Mat &dst, const OpTarget &target)
{
    Arithmetic arithmetic(ctx, target);

    return OpCall(ctx, arithmetic, &src0, &src1, &dst, ArithmOpType::SUB);
}

AURA_EXPORTS Status IMultiply(Context *ctx, const Mat &src0, const Mat &src1, Mat &dst, const OpTarget &target)
{
    Arithmetic arithmetic(ctx, target);

    return OpCall(ctx, arithmetic, &src0, &src1, &dst, ArithmOpType::MUL);
}

AURA_EXPORTS Status IDivide(Context *ctx, const Mat &src0, const Mat &src1, Mat &dst, const OpTarget &target)
{
    Arithmetic arithmetic(ctx, target);

    return OpCall(ctx, arithmetic, &src0, &src1, &dst, ArithmOpType::DIV);
}

AURA_EXPORTS Status IAdd(Context *ctx, const Mat &src, MI_F32 scalar, Mat &dst, const OpTarget &target)
{
    Status ret = IConvertTo(ctx, src, dst, 1.f, scalar, target);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "IConvertTo failed");
    }
    return ret;
}

AURA_EXPORTS Status ISubtract(Context *ctx, const Mat &src, MI_F32 scalar, Mat &dst, const OpTarget &target)
{
    Status ret = IConvertTo(ctx, src, dst, 1.f, -scalar, target);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "IConvertTo failed");
    }
    return ret;
}

AURA_EXPORTS Status IMultiply(Context *ctx, const Mat &src, MI_F32 scalar, Mat &dst, const OpTarget &target)
{
    Status ret = IConvertTo(ctx, src, dst, scalar, 0.f, target);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "IConvertTo failed");
    }
    return ret;
}

AURA_EXPORTS Status IDivide(Context *ctx, const Mat &src, MI_F32 scalar, Mat &dst, const OpTarget &target)
{
    Status ret = IConvertTo(ctx, src, dst, 1.f / scalar, 0.f, target);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "IConvertTo failed");
    }
    return ret;
}

AURA_EXPORTS Status ISubtract(Context *ctx, MI_F32 scalar, const Mat &src, Mat &dst, const OpTarget &target)
{
    Status ret = IConvertTo(ctx, src, dst, -1.f, scalar, target);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "IConvertTo failed");
    }
    return ret;
}

AURA_EXPORTS Status IDivide(Context *ctx, MI_F32 scalar, const Mat &src, Mat &dst, const OpTarget &target)
{
    ScalarDivide scalar_divide(ctx, target);

    return OpCall(ctx, scalar_divide, scalar, &src, &dst);
}

ArithmeticImpl::ArithmeticImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "Arithmetic", target),
                                                                       m_op_type(ArithmOpType::ADD), m_src0(MI_NULL),
                                                                       m_src1(MI_NULL), m_dst(MI_NULL)
{}

Status ArithmeticImpl::SetArgs(const Array *src0, const Array *src1, Array *dst, ArithmOpType op)
{
    if (MI_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if (!src0->IsValid() || !src1->IsValid() || !dst->IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid src or dst");
        return Status::ERROR;
    }

    if (!src0->IsEqual(*src1) || !src0->IsSizesEqual(*dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src0 and src1 must be same datatype and same size, src and dst must be same size.");
        return Status::ERROR;
    }

    m_src0    = src0;
    m_src1    = src1;
    m_dst     = dst;
    m_op_type = op;

    return Status::OK;
}

std::vector<const Array*> ArithmeticImpl::GetInputArrays() const
{
    return {m_src0, m_src1};
}

std::vector<const Array*> ArithmeticImpl::GetOutputArrays() const
{
    return {m_dst};
}

std::string ArithmeticImpl::ToString() const
{
    std::string str;

    str = "op(Arithmetic)";
    str += " target(" + GetOpTarget().ToString() + ")";
    str += " type(" + ArithmeOpTypeToString(m_op_type) + ")\n";

    return str;
}

AURA_VOID ArithmeticImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    std::vector<std::string>  names  = {"src_0", "src_1", "dst"};
    std::vector<const Array*> arrays = {m_src0, m_src1, m_dst};

    if (json_wrapper.SetArray(names, arrays) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray src failed");
        return;
    }

    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src0, m_src1, m_dst, m_op_type);
}

ScalarDivideImpl::ScalarDivideImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "ScalarDivide", target),
                                                                           m_src(MI_NULL), m_dst(MI_NULL),
                                                                           m_scalar(1)
{}

Status ScalarDivideImpl::SetArgs(MI_F32 scalar, const Array *src, Array *dst)
{
    if (MI_NULL == m_ctx)
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
        AURA_ADD_ERROR_STRING(m_ctx, "sizes of src and dst mat dismatch");
        return Status::ERROR;
    }

    m_src    = src;
    m_dst    = dst;
    m_scalar = scalar;

    return Status::OK;
}

std::vector<const Array*> ScalarDivideImpl::GetInputArrays() const
{
    return {m_src};
}

std::vector<const Array*> ScalarDivideImpl::GetOutputArrays() const
{
    return {m_dst};
}

std::string ScalarDivideImpl::ToString() const
{
    std::string str;

    str = "op(ScalarDivide)";
    str += " target(" + GetOpTarget().ToString() + ")\n";

    return str;
}

AURA_VOID ScalarDivideImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    std::vector<std::string>  names  = {"src", "dst"};
    std::vector<const Array*> arrays = {m_src, m_dst};

    if (json_wrapper.SetArray(names, arrays) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray src failed");
        return;
    }

    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src, m_dst, m_scalar);
}

} // namespace aura