#include "aura/ops/matrix/merge.hpp"
#include "merge_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/tools/json.h"

namespace aura
{

static std::shared_ptr<MergeImpl> CreateMergeImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<MergeImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new MergeNone(ctx, target));
            break;
        }

        default :
        {
            break;
        }
    }

    return impl;
}

Merge::Merge(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status Merge::SetArgs(const std::vector<const Array*> &src, Array *dst)
{
    if ((MI_NULL == m_ctx))
    {
        return Status::ERROR;
    }

    for (auto mat : src)
    {
        if (MI_NULL == mat)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "src is null ptr");
            return Status::ERROR;
        }
    }

    if (MI_NULL == dst)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "dst is null ptr");
        return Status::ERROR;
    }

    OpTarget impl_target = m_target;

    // set m_impl
    if (MI_NULL == m_impl.get() || impl_target != m_impl->GetOpTarget())
    {
        m_impl = CreateMergeImpl(m_ctx, impl_target);
    }

    // run initialize
    MergeImpl *merge_impl = dynamic_cast<MergeImpl*>(m_impl.get());
    if (MI_NULL == merge_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "merge_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = merge_impl->SetArgs(src, dst);

    AURA_RETURN(m_ctx, ret);
}

AURA_EXPORTS Status IMerge(Context *ctx, const std::vector<Mat> &src, Mat &dst, const OpTarget &target)
{
    Merge merge(ctx, target);

    std::vector<const Array*> src_arrays;
    src_arrays.reserve(src.size());

    for (auto &mat : src)
    {
        src_arrays.emplace_back(dynamic_cast<const Array*>(&mat));
    }

    return OpCall(ctx, merge, src_arrays, &dst);
}

MergeImpl::MergeImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "Merge", target),
                                                             m_dst(MI_NULL)
{}

Status MergeImpl::SetArgs(const std::vector<const Array*> &src, Array *dst)
{
    if (MI_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    for (auto mat : src)
    {
        if (MI_NULL == mat)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "src is null ptr");
            return Status::ERROR;
        }
    }

    if (MI_NULL == dst)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "dst is null ptr");
        return Status::ERROR;
    }

    const size_t src_size = src.size();
    m_src.resize(src_size);

    if (!dst->IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst mat is invalid.");
        return Status::ERROR;
    }

    const ElemType dst_type = dst->GetElemType();
    const Sizes3 dst_sz     = dst->GetSizes();
    MI_S32 src_total_ch     = 0;

    for (size_t i = 0; i < src_size; ++i)
    {
        if (!src[i]->IsValid())
        {
            AURA_ADD_ERROR_STRING(m_ctx, "src is invalid.");
            return Status::ERROR;
        }

        if (src[i]->GetElemType() != dst_type)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "src and dst mat has different elem_type.");
            return Status::ERROR;
        }

        const Sizes3 src_sz = src[i]->GetSizes();

        if (src_sz.m_width != dst_sz.m_width || src_sz.m_height != dst_sz.m_height)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "src and dst mat width or height does not match.");
            return Status::ERROR;
        }

        src_total_ch += src_sz.m_channel;
        m_src[i] = src[i];
    }

    if (src_total_ch != dst_sz.m_channel)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src mats' total channel count does not match dst mat's channel count.");
        return Status::ERROR;
    }

    m_dst = dst;

    return Status::OK;
}

std::vector<const Array*> MergeImpl::GetInputArrays() const
{
    return m_src;
}

std::vector<const Array*> MergeImpl::GetOutputArrays() const
{
    return {m_dst};
}

std::string MergeImpl::ToString() const
{
    std::string str;

    str = "op(Merge)";
    str += " target(" + GetOpTarget().ToString() + ")\n";

    return str;
}

AURA_VOID MergeImpl::Dump(const std::string &prefix) const
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