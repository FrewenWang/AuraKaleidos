#include "connect_component_label_impl.hpp"
#include "aura/tools/json.h"

namespace aura
{

#if defined(AURA_ENABLE_OPENCL)
AURA_INLINE Status CheckCLSupport(Context *ctx, const Array &array)
{
    DT_S32 width  = array.GetSizes().m_width;
    DT_S32 height = array.GetSizes().m_height;
    if (width < 16 || height < 4)
    {
        AURA_ADD_ERROR_STRING(ctx, "size is too small for opencl");
        return Status::ERROR;
    }

    return Status::OK;
}
#endif // AURA_ENABLE_OPENCL

static std::shared_ptr<ConnectComponentLabelImpl> CreateCCLImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<ConnectComponentLabelImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new ConnectComponentLabelNone(ctx, target));
            break;
        }

        case TargetType::OPENCL:
        {
#if defined(AURA_ENABLE_OPENCL)
            impl.reset(new ConnectComponentLabelCL(ctx, target));
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

ConnectComponentLabel::ConnectComponentLabel(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status ConnectComponentLabel::SetArgs(const Array *src, Array *dst, CCLAlgo algo_type, ConnectivityType connectivity_type,
                                      EquivalenceSolver solver_type)
{
    if ((DT_NULL == m_ctx))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Context is null");
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

        case TargetType::OPENCL:
        {
#if defined(AURA_ENABLE_OPENCL)
            if (CheckCLSupport(m_ctx, *src) != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "switch opencl impl to neon impl");
                impl_target = OpTarget::None(DT_TRUE);
                algo_type   = aura::CCLAlgo::SPAGHETTI;
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
        m_impl = CreateCCLImpl(m_ctx, impl_target);
    }

    // run initialize
    ConnectComponentLabelImpl *ccl_impl = dynamic_cast<ConnectComponentLabelImpl *>(m_impl.get());
    if (DT_NULL == ccl_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ccl_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = ccl_impl->SetArgs(src, dst, algo_type, connectivity_type, solver_type);

    AURA_RETURN(m_ctx, ret);
}

Status ConnectComponentLabel::CLPrecompile(Context *ctx, ElemType dst_elem_type, CCLAlgo algo_type, ConnectivityType connectivity_type)
{
#if defined(AURA_ENABLE_OPENCL)
    if (DT_NULL == ctx)
    {
        return Status::ERROR;
    }

    std::vector<CLKernel> cl_kernels = ConnectComponentLabelCL::GetCLKernels(ctx, dst_elem_type, algo_type, connectivity_type);

    if (CheckCLKernels(ctx, cl_kernels) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "ConnectComponentLabel CheckCLKernels failed");
        return Status::ERROR;
    }
#else
    AURA_UNUSED(ctx);
    AURA_UNUSED(dst_elem_type);
    AURA_UNUSED(algo_type);
    AURA_UNUSED(connectivity_type);
#endif
    return Status::OK;
}

AURA_EXPORTS Status IConnectComponentLabel(Context *ctx, const Mat &src, Mat &dst, CCLAlgo algo_type, ConnectivityType connectivity_type,
                                           EquivalenceSolver solver_type, const OpTarget &target)
{
    ConnectComponentLabel ccl_impl(ctx, target);

    return OpCall(ctx, ccl_impl, &src, &dst, algo_type, connectivity_type, solver_type);
}

ConnectComponentLabelImpl::ConnectComponentLabelImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "ConnectComponentLabel", target),
                                                                                             m_src(DT_NULL), m_dst(DT_NULL),
                                                                                             m_connectivity_type(ConnectivityType::CROSS),
                                                                                             m_algo_type(CCLAlgo::SPAGHETTI)
{}

Status ConnectComponentLabelImpl::SetArgs(const Array *src, Array *dst, CCLAlgo algo_type,
                                          ConnectivityType connectivity_type, EquivalenceSolver solver_type)
{
    AURA_UNUSED(solver_type);
    if (DT_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if (!(src->IsValid() && dst->IsValid()))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid src or dst");
        return Status::ERROR;
    }

    if (!src->IsSizesEqual(*dst) || 1 != src->GetSizes().m_channel)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src and dst should have the same size and channel must be 1");
        return Status::ERROR;
    }

    if (src->GetElemType() != aura::ElemType::U8)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src elem type must be U8");
        return Status::ERROR;
    }

    if (dst->GetElemType() >= aura::ElemType::F32)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "dst elem type can't be float point");
        return Status::ERROR;
    }

    m_src               = src;
    m_dst               = dst;
    m_algo_type         = algo_type;
    m_connectivity_type = connectivity_type;

    return Status::OK;
}

std::vector<const Array*> ConnectComponentLabelImpl::GetInputArrays() const
{
    return {m_src};
}

std::vector<const Array*> ConnectComponentLabelImpl::GetOutputArrays() const
{
    return {m_dst};
}

std::string ConnectComponentLabelImpl::ToString() const
{
    std::string str;

    str =  "op(ConnectComponentLabel)";
    str += " target(" + GetOpTarget().ToString() + ")";
    str += " param(""algo_type:" + CCLAlgoTypeToString(m_algo_type) + " | " +
            "connectivity_type:" + ConnectivityTypeToString(m_connectivity_type) + ")\n";

    return str;
}

DT_VOID ConnectComponentLabelImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    std::vector<std::string>  names  = {"src", "dst"};
    std::vector<const Array*> arrays = {m_src, m_dst};

    if (json_wrapper.SetArray(names, arrays) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray src failed");
        return;
    }

    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src, m_dst, m_connectivity_type, m_algo_type);
}

} // namespace aura