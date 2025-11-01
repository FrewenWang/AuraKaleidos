#include "aura/ops/misc/mipi.hpp"
#include "mipi_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/tools/json.h"

namespace aura
{

static std::shared_ptr<MipiPackImpl> CreateMipiPackImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<MipiPackImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new MipiPackNone(ctx, target));
            break;
        }

        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            impl.reset(new MipiPackNeon(ctx, target));
#endif // AURA_ENABLE_NEON
            break;
        }

        case TargetType::HVX:
        {
#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
            impl.reset(new MipiPackHvx(ctx, target));
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

MipiPack::MipiPack(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status MipiPack::SetArgs(const Array *src, Array *dst)
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

        case TargetType::HVX:
        {
#if defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON)
            if (CheckHvxWidth(*src) != Status::OK)
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
        m_impl = CreateMipiPackImpl(m_ctx, impl_target);
    }

    // run SetArgs
    MipiPackImpl *mipipack_impl = dynamic_cast<MipiPackImpl *>(m_impl.get());
    if (MI_NULL == mipipack_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "mipipack_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = mipipack_impl->SetArgs(src, dst);

    AURA_RETURN(m_ctx, ret);
}

AURA_EXPORTS Status IMipiPack(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    MipiPack mipi_pack(ctx, target);

    return OpCall(ctx, mipi_pack, &src, &dst);
}

MipiPackImpl::MipiPackImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "MipiPack", target),
                                                                   m_src(MI_NULL), m_dst(MI_NULL)
{}

Status MipiPackImpl::SetArgs(const Array *src, Array *dst)
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

    if (!src->IsChannelEqual(*dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src and dst should have the same channel num");
        return Status::ERROR;
    }

    if (src->GetElemType() != ElemType::U16 || dst->GetElemType() != ElemType::U8)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src/dst elem type error, only support src is u16, dst is u8");
        return Status::ERROR;
    }

    MI_S32 iwidth  = src->GetSizes().m_width;
    MI_S32 iheight = src->GetSizes().m_height;
    MI_S32 channel = src->GetSizes().m_channel;
    MI_S32 owidth  = dst->GetSizes().m_width;
    MI_S32 oheight = dst->GetSizes().m_height;
    if (((iwidth % 4) != 0) || ((owidth % 5) != 0) || (((owidth / 5) << 2) != iwidth) || (iheight != oheight) || (channel != 1))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "size err, iwidth should be 4/5 of owidth, height should be equal, channel should be 1");
        return Status::ERROR;
    }

    m_src = src;
    m_dst = dst;

    return Status::OK;
}

std::vector<const Array*> MipiPackImpl::GetInputArrays() const
{
    return {m_src};
}

std::vector<const Array*> MipiPackImpl::GetOutputArrays() const
{
    return {m_dst};
}

std::string MipiPackImpl::ToString() const
{
    std::string str("op(MipiPack)");

    return str;
}

AURA_VOID MipiPackImpl::Dump(const std::string &prefix) const
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

static std::shared_ptr<MipiUnPackImpl> CreateMipiUnPackImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<MipiUnPackImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new MipiUnPackNone(ctx, target));
            break;
        }

        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            impl.reset(new MipiUnPackNeon(ctx, target));
#endif // AURA_ENABLE_NEON
            break;
        }

        case TargetType::HVX:
        {
#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
            impl.reset(new MipiUnPackHvx(ctx, target));
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

MipiUnPack::MipiUnPack(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status MipiUnPack::SetArgs(const Array *src, Array *dst)
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

    OpTarget impl_target = m_target;

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

        case TargetType::HVX:
        {
#if defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON)
            if (CheckHvxWidth(*src) != Status::OK)
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
        m_impl = CreateMipiUnPackImpl(m_ctx, impl_target);
    }

    // run SetArgs
    MipiUnPackImpl *mipiunpack_impl = dynamic_cast<MipiUnPackImpl *>(m_impl.get());
    if (MI_NULL == mipiunpack_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "mipiunpack_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = mipiunpack_impl->SetArgs(src, dst);

    AURA_RETURN(m_ctx, ret);
}

AURA_EXPORTS Status IMipiUnpack(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    MipiUnPack mipi_unpack(ctx, target);

    return OpCall(ctx, mipi_unpack, &src, &dst);
}

MipiUnPackImpl::MipiUnPackImpl(Context *ctx, const OpTarget &target): OpImpl(ctx, "MipiUnPack", target),
                                                                      m_src(MI_NULL), m_dst(MI_NULL)
{}

Status MipiUnPackImpl::SetArgs(const Array *src, Array *dst)
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

    if (!src->IsChannelEqual(*dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src and dst should have the same channel num");
        return Status::ERROR;
    }

    if (src->GetElemType() != ElemType::U8 || !(ElemType::U8 == dst->GetElemType() || ElemType::U16 == dst->GetElemType()))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst elem type error");
        return Status::ERROR;
    }

    MI_S32 iwidth  = src->GetSizes().m_width;
    MI_S32 iheight = src->GetSizes().m_height;
    MI_S32 channel = src->GetSizes().m_channel;
    MI_S32 owidth  = dst->GetSizes().m_width;
    MI_S32 oheight = dst->GetSizes().m_height;

    if (((iwidth % 5) != 0) || (owidth != iwidth / 5 * 4) || (iheight != oheight) || (channel != 1))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "size err, owidth should be 4/5 of iwidth, height should be equal, channel should be 1");
        return Status::ERROR;
    }

    m_src = src;
    m_dst = dst;

    return Status::OK;
}

std::vector<const Array*> MipiUnPackImpl::GetInputArrays() const
{
    return {m_src};
}

std::vector<const Array*> MipiUnPackImpl::GetOutputArrays() const
{
    return {m_dst};
}

std::string MipiUnPackImpl::ToString() const
{
    std::string str("op(MipiUnPack)");

    return str;
}

AURA_VOID MipiUnPackImpl::Dump(const std::string &prefix) const
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