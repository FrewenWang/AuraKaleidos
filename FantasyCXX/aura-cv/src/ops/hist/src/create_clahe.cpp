#include "aura/ops/hist/create_clahe.hpp"
#include "create_clahe_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/tools/json.h"

namespace aura
{

static std::shared_ptr<CreateClAHEImpl> CreateCreateClAHEImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<CreateClAHEImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new CreateClAHENone(ctx, target));
            break;
        }

        default :
        {
            break;
        }
    }

    return impl;
}

CreateClAHE::CreateClAHE(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status CreateClAHE::SetArgs(const Array *src, Array *dst, DT_F64 clip_limit, const Sizes &tile_grid_size)
{
    Status ret = Status::ERROR;

    if ((DT_NULL == m_ctx))
    {
        return ret;
    }

    if ((DT_NULL == src) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src/dst is null ptr");
        return ret;
    }

    OpTarget impl_target = m_target;

    // set m_impl
    if (DT_NULL == m_impl.get() || impl_target != m_impl->GetOpTarget())
    {
        m_impl = CreateCreateClAHEImpl(m_ctx, impl_target);
    }

    // run SetArgs
    CreateClAHEImpl *create_clahe_impl = dynamic_cast<CreateClAHEImpl *>(m_impl.get());
    if (DT_NULL == create_clahe_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "create_clahe_impl is null ptr");
        return ret;
    }

    ret = create_clahe_impl->SetArgs(src, dst, clip_limit, tile_grid_size);

    AURA_RETURN(m_ctx, ret);
}

AURA_EXPORTS Status ICreateClAHE(Context *ctx, const Mat &src, Mat &dst, DT_F64 clip_limit,
                                 const Sizes &tile_grid_size, const OpTarget &target)
{
    CreateClAHE create_clahe(ctx, target);

    return OpCall(ctx, create_clahe, &src, &dst, clip_limit, tile_grid_size);
}

CreateClAHEImpl::CreateClAHEImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "CreateClAHE", target),
                                                                         m_clip_limit(0.0), m_src(DT_NULL), m_dst(DT_NULL)
{}

Status CreateClAHEImpl::SetArgs(const Array *src, Array *dst, DT_F64 clip_limit, const Sizes &tile_grid_size)
{
    Status ret = Status::ERROR;

    if (DT_NULL == m_ctx)
    {
        return ret;
    }

    if (!(src->IsValid() && dst->IsValid()))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid src or dst");
        return ret;
    }

    if (!src->IsEqual(*dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src and dst should have the same size");
        return ret;
    }

    if (src->GetSizes().m_channel != 1 || dst->GetSizes().m_channel != 1)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src and dst channel must be 1");
        return ret;
    }

    if (!(ElemType::U8 == src->GetElemType() || ElemType::U16 == src->GetElemType()))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src elem type support u8/u16");
        return ret;
    }

    m_src            = src;
    m_dst            = dst;
    m_clip_limit     = clip_limit;
    m_tile_grid_size = tile_grid_size;

    return Status::OK;
}

std::vector<const Array*> CreateClAHEImpl::GetInputArrays() const
{
    return {m_src};
}

std::vector<const Array*> CreateClAHEImpl::GetOutputArrays() const
{
    return {m_dst};
}

std::string CreateClAHEImpl::ToString() const
{
    std::string str;

    str = "op(CreateClAHE)";
    str += " target(" + GetOpTarget().ToString() + ")";
    str += " param(" + std::string("clip_limit:") + std::to_string(m_clip_limit) + " | " +
            "tile_grid_size:" + m_tile_grid_size.ToString() + ")\n";

    return str;
}

DT_VOID CreateClAHEImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    std::vector<std::string>  names  = {"src", "dst"};
    std::vector<const Array*> arrays = {m_src, m_dst};

    if (json_wrapper.SetArray(names, arrays) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray src failed");
        return;
    }

    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src, m_dst, m_clip_limit, m_tile_grid_size);
}

} // namespace aura