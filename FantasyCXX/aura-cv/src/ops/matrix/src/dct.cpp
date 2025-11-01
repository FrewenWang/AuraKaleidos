#include "aura/ops/matrix/dct.hpp"
#include "dct_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/tools/json.h"

namespace aura
{

static std::shared_ptr<DctImpl> CreateDctImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<DctImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new DctNone(ctx, target));
            break;
        }

        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            impl.reset(new DctNeon(ctx, target));
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

Dct::Dct(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status Dct::SetArgs(const Array *src, Array *dst)
{
    if ((MI_NULL == m_ctx))
    {
        return Status::ERROR;
    }

    if ((MI_NULL == src || MI_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "dst or src null ptr");
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

        default:
        {
            break;
        }
    }

    if (MI_NULL == m_impl.get() || impl_target != m_impl->GetOpTarget())
    {
        m_impl = CreateDctImpl(m_ctx, impl_target);
    }

    DctImpl *dct_impl = dynamic_cast<DctImpl *>(m_impl.get());
    if (MI_NULL == dct_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "dct_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = dct_impl->SetArgs(src, dst);

    AURA_RETURN(m_ctx, ret);
}

AURA_EXPORTS Status IDct(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    Dct dct(ctx, target);

    if (dct.SetArgs(&src, &dst) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Initialize failed");
        return Status::ERROR;
    }

    return OpCall(ctx, dct, &src, &dst);
}

DctImpl::DctImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "Dct", target)
{}

Status DctImpl::SetArgs(const Array *src, Array *dst)
{
    if (MI_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if ((MI_NULL == src) || (MI_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "dct / dst mat is null");
        return Status::ERROR;
    }

    if ((MI_FALSE == src->IsValid()) || (MI_FALSE == dst->IsValid()))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "dct / dst mat not vaild");
        return Status::ERROR;
    }

    if (MI_FALSE == src->IsSizesEqual(*dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "dct and dst mat not meeting the requirement of equal size");
        return Status::ERROR;
    }

    Sizes3 src_sz = src->GetSizes();
    if (src_sz.m_channel != 1)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Dct only support input and output with ch = 1");
        return Status::ERROR;
    }

    if (dst->GetElemType() != ElemType::F32)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "dct dst mat element type must F32");
        return Status::ERROR;
    }

    m_src = src;
    m_dst = dst;
    return Status::OK;
}

std::vector<const Array*> DctImpl::GetInputArrays() const
{
    return {m_src};
}

std::vector<const Array*> DctImpl::GetOutputArrays() const
{
    return {m_dst};
}

std::string DctImpl::ToString() const
{
    std::string str;
    str = "op(Dct)";
    str += " target(" + GetOpTarget().ToString() + ")";

    return str;
}

AURA_VOID DctImpl::Dump(const std::string &prefix) const
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

static std::shared_ptr<IDctImpl> CreateIDctImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<IDctImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new IDctNone(ctx, target));
            break;
        }

        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            impl.reset(new IDctNeon(ctx, target));
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

InverseDct::InverseDct(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status InverseDct::SetArgs(const Array *src, Array *dst)
{
    if ((MI_NULL == m_ctx))
    {
        return Status::ERROR;
    }

    if ((MI_NULL == src || MI_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "dst or src null ptr");
        return Status::ERROR;
    }

    OpTarget impl_target = m_target;

    if (MI_NULL == m_impl.get() || impl_target != m_impl->GetOpTarget())
    {
        m_impl = CreateIDctImpl(m_ctx, impl_target);
    }

    IDctImpl *idct_impl = dynamic_cast<IDctImpl *>(m_impl.get());
    if (MI_NULL == idct_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "dct_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = idct_impl->SetArgs(src, dst);

    AURA_RETURN(m_ctx, ret);
}

AURA_EXPORTS Status IInverseDct(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    InverseDct idct(ctx, target);

    if (idct.SetArgs(&src, &dst) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Initialize failed");
        return Status::ERROR;
    }

    return OpCall(ctx, idct, &src, &dst);
}

IDctImpl::IDctImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "IDct", target), m_src(MI_NULL), m_dst(MI_NULL)
{}

Status IDctImpl::SetArgs(const Array *src, Array *dst)
{
    if (MI_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if ((MI_NULL == src) || (MI_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "dct / dst mat is null");
        return Status::ERROR;
    }

    if ((MI_FALSE == src->IsValid()) || (MI_FALSE == dst->IsValid()))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "dct / dst mat not vaild");
        return Status::ERROR;
    }

    if (MI_FALSE == src->IsSizesEqual(*dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "dct and dst size not equal");
        return Status::ERROR;
    }

    Sizes3 src_sz = src->GetSizes();
    if (src_sz.m_channel != 1)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "IDct only support real input and output with ch = 1");
        return Status::ERROR;
    }

    if (ElemType::F32 != src->GetElemType())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "dct src mat element type must F32");
        return Status::ERROR;
    }

    m_src = src;
    m_dst = dst;

    return Status::OK;
}

std::vector<const Array*> IDctImpl::GetInputArrays() const
{
    return {m_src};
}

std::vector<const Array*> IDctImpl::GetOutputArrays() const
{
    return {m_dst};
}

std::string IDctImpl::ToString() const
{
    std::string str;
    str = "op(IDct)";
    str += " target(" + GetOpTarget().ToString() + ")";

    return str;
}

AURA_VOID IDctImpl::Dump(const std::string &prefix) const
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

Status IDctImpl::Initialize()
{
    if (MI_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if (OpImpl::Initialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "OpImpl::Initialize() failed");
        return Status::ERROR;
    }

    if (m_dst->GetElemType() != ElemType::F32)
    {
        Sizes3 dst_sz = m_dst->GetSizes();
        m_mid = Mat(m_ctx, ElemType::F32, dst_sz, AURA_MEM_DEFAULT);
        if (MI_FALSE == m_mid.IsValid())
        {
            AURA_ADD_ERROR_STRING(m_ctx, "Create m_mid mat failed.");
            return Status::ERROR;
        }
    }

    return Status::OK;
}

Status IDctImpl::DeInitialize()
{
    m_mid.Release();

    if (OpImpl::DeInitialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "OpImpl::DeInitialize() failed");
        return Status::ERROR;
    }

    return Status::OK;
}

} // namespace aura