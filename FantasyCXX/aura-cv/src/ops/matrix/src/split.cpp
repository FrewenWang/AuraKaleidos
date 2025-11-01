#include "aura/ops/matrix/split.hpp"
#include "split_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/tools/json.h"

namespace aura
{

static std::shared_ptr<SplitImpl> CreateSplitImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<SplitImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new SplitNone(ctx, target));
            break;
        }

        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            impl.reset(new SplitNeon(ctx, target));
#endif
            break;
        }

        default :
        {
            break;
        }
    }

    return impl;
}

Split::Split(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status Split::SetArgs(const Array *src, const std::vector<Array*> &dst)
{
    if ((MI_NULL == m_ctx))
    {
        return Status::ERROR;
    }

    if (MI_NULL == src)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src is null ptr");
        return Status::ERROR;
    }

    for (auto mat : dst)
    {
        if (MI_NULL == mat)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "dst is null ptr");
            return Status::ERROR;
        }
    }

    OpTarget impl_target = m_target;

    // set m_impl
    if (MI_NULL == m_impl.get() || impl_target != m_impl->GetOpTarget())
    {
        m_impl = CreateSplitImpl(m_ctx, impl_target);
    }

    // run initialize
    SplitImpl *split_impl = dynamic_cast<SplitImpl*>(m_impl.get());
    if (MI_NULL == split_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "split_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = split_impl->SetArgs(src, dst);

    AURA_RETURN(m_ctx, ret);
}

AURA_EXPORTS Status ISplit(Context *ctx, const Mat &src, std::vector<Mat> &dst, const OpTarget &target)
{
    Split split(ctx, target);

    std::vector<Array*> dst_arrays;
    dst_arrays.reserve(dst.size());

    for (auto &mat : dst)
    {
        dst_arrays.emplace_back(&mat);
    }

    return OpCall(ctx, split, &src, dst_arrays);
}

SplitImpl::SplitImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "Split", target), m_src(MI_NULL)
{}

Status SplitImpl::SetArgs(const Array *src, const std::vector<Array*> &dst)
{
    if (MI_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if (MI_NULL == src)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src is null ptr");
        return Status::ERROR;
    }

    for (auto mat : dst)
    {
        if (MI_NULL == mat)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "dst is null ptr");
            return Status::ERROR;
        }
    }

    const size_t dst_size = dst.size();
    m_dst.resize(dst_size);

    if (!src->IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "dst or src mat is invalid.");
        return Status::ERROR;
    }

    const ElemType src_type = src->GetElemType();
    const Sizes3 src_sz     = src->GetSizes();
    MI_S32 dst_total_ch     = 0;

    for (size_t i = 0; i < dst_size; ++i)
    {
        if (!dst[i]->IsValid())
        {
            AURA_ADD_ERROR_STRING(m_ctx, "dst is invalid.");
            return Status::ERROR;
        }

        if (dst[i]->GetElemType() != src_type)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "dst and src mat has different elem_type.");
            return Status::ERROR;
        }

        const Sizes3 dst_sz = dst[i]->GetSizes();

        if (dst_sz.m_width != src_sz.m_width || dst_sz.m_height != src_sz.m_height)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "dst and src mat width or height does not match.");
            return Status::ERROR;
        }

        dst_total_ch += dst_sz.m_channel;
        m_dst[i] = dst[i];
    }

    if (dst_total_ch != src_sz.m_channel)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "dst mats' total channel count does not match src mat's channel count.");
        return Status::ERROR;
    }

    m_src = src;
    return Status::OK;
}

std::vector<const Array*> SplitImpl::GetInputArrays() const
{
    return {m_src};
}

std::vector<const Array*> SplitImpl::GetOutputArrays() const
{
    std::vector<const Array*> dst_out;

    for (MI_U32 i = 0; i < m_dst.size(); i++)
    {
        dst_out.push_back(m_dst[i]);
    }

    return dst_out;
}

std::string SplitImpl::ToString() const
{
    std::string str;

    str = "op(Split)";
    str += " target(" + GetOpTarget().ToString() + ")\n";

    return str;
}

AURA_VOID SplitImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    if (json_wrapper.SetArray("src", m_src) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray src failed");
        return;
    }

    if (json_wrapper.SetArray("dst", m_dst) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray dst failed");
        return;
    }

    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src, m_dst);
}

} // namespace aura