#include "aura/ops/warp/remap.hpp"
#include "remap_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/tools/json.h"

namespace aura
{

#if defined(AURA_ENABLE_OPENCL)
AURA_INLINE Status CheckCLSupport(const Array *array)
{
    if (array->GetSizes().m_width < 4)
    {
        return Status::ERROR;
    }

    return Status::OK;
}
#endif // AURA_ENABLE_OPENCL

static std::shared_ptr<RemapImpl> CreateRemapImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<RemapImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new RemapNone(ctx, target));
            break;
        }

        case TargetType::OPENCL:
        {
#if defined(AURA_ENABLE_OPENCL)
            impl.reset(new RemapCL(ctx, target));
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

Remap::Remap(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status Remap::SetArgs(const Array *src, const Array *map, Array *dst, InterpType interp_type,
                      BorderType border_type, const Scalar &border_value)
{
    if ((DT_NULL == m_ctx))
    {
        return Status::ERROR;
    }

    if ((DT_NULL == src) || (DT_NULL == dst) || (DT_NULL == map))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src/dst/map is null ptr");
        return Status::ERROR;
    }

    OpTarget impl_target = m_target;

    // check impl type
    switch (m_target.m_type)
    {
        case TargetType::NONE:
        {
            impl_target = OpTarget::None();
            break;
        }

        case TargetType::OPENCL:
        {
#if defined(AURA_ENABLE_OPENCL)
            if (CheckCLSupport(src) != Status::OK)
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
        m_impl = CreateRemapImpl(m_ctx, impl_target);
    }

    // run initialize
    RemapImpl *remap_impl = dynamic_cast<RemapImpl*>(m_impl.get());
    if (DT_NULL == remap_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "remap_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = remap_impl->SetArgs(src, map, dst, interp_type, border_type, border_value);

    AURA_RETURN(m_ctx, ret);
}

Status Remap::CLPrecompile(Context *ctx, ElemType map_elem_type, ElemType dst_elem_type, DT_S32 channel,
                           BorderType border_type, InterpType interp_type)
{
#if defined(AURA_ENABLE_OPENCL)
    if (DT_NULL == ctx)
    {
        return Status::ERROR;
    }

    std::vector<CLKernel> cl_kernels = RemapCL::GetCLKernels(ctx, map_elem_type, dst_elem_type, channel, border_type, interp_type);

    if (CheckCLKernels(ctx, cl_kernels) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Remap CheckCLKernels failed");
        return Status::ERROR;
    }
#else
    AURA_UNUSED(ctx);
    AURA_UNUSED(map_elem_type);
    AURA_UNUSED(dst_elem_type);
    AURA_UNUSED(channel);
    AURA_UNUSED(border_type);
    AURA_UNUSED(interp_type);
#endif
    return Status::OK;
}

AURA_EXPORTS Status IRemap(Context *ctx, const Mat &src, const Mat &map, Mat &dst, InterpType interp_type,
                           BorderType border_type, const Scalar &border_value, const OpTarget &target)
{
    Remap remap(ctx, target);

    return OpCall(ctx, remap, &src, &map, &dst, interp_type, border_type, border_value);
}

RemapImpl::RemapImpl(Context *ctx, const OpTarget &target = OpTarget::Default()) : OpImpl(ctx, "Remap", target),
                                                                                   m_src(DT_NULL), m_map(DT_NULL),
                                                                                   m_dst(DT_NULL), m_interp_type(InterpType::LINEAR),
                                                                                   m_border_type(BorderType::REPLICATE)
{}

Status RemapImpl::SetArgs(const Array *src, const Array *map, Array *dst, InterpType interp_type,
                          BorderType border_type, const Scalar &border_value)
{
    if (DT_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if (!src->IsValid() || !map->IsValid() || !dst->IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "input mat is invalid.");
        return Status::ERROR;
    }

    if ((src->GetElemType() != dst->GetElemType()) || (!src->IsChannelEqual(*dst)))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src and dst should have the same elemtype/channels");
        return Status::ERROR;
    }

    if (map->GetSizes().m_height  != dst->GetSizes().m_height ||
        map->GetSizes().m_width   != dst->GetSizes().m_width  ||
        map->GetSizes().m_channel != 2)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "map and dst should have the same height/width");
        return Status::ERROR;
    }

    m_src = src;
    m_map = map;
    m_dst = dst;
    m_interp_type  = interp_type;
    m_border_type  = border_type;
    m_border_value = border_value;

    return Status::OK;
}

std::vector<const Array*> RemapImpl::GetInputArrays() const
{
    return {m_src, m_map};
}

std::vector<const Array*> RemapImpl::GetOutputArrays() const
{
    return {m_dst};
}

std::string RemapImpl::ToString() const
{
    std::string str;

    str = "op(Remap)";
    str += " target(" + GetOpTarget().ToString() + ")";
    str += " param(" + InterpTypeToString(m_interp_type) + " | " +
           BorderTypeToString(m_border_type) + " | " +
           "border_value:" + m_border_value.ToString() + ")\n";

    return str;
}

DT_VOID RemapImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    std::vector<std::string>  names  = {"src", "map", "dst"};
    std::vector<const Array*> arrays = {m_src, m_map, m_dst};

    if (json_wrapper.SetArray(names, arrays) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray failed");
        return;
    }

    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src, m_map, m_dst, m_interp_type, m_border_type, m_border_value);
}

} // namespace aura