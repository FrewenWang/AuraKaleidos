#include "aura/ops/matrix/psnr.hpp"
#include "psnr_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/tools/json.h"

namespace aura
{

static std::shared_ptr<PsnrImpl> CreatePsnrImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<PsnrImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new PsnrNone(ctx, target));
            break;
        }

        default :
        {
            break;
        }
    }

    return impl;
}

Psnr::Psnr(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status Psnr::SetArgs(const Array *src0, const Array *src1, DT_F64 coef_r, DT_F64 *result)
{
    if ((DT_NULL == m_ctx))
    {
        return Status::ERROR;
    }

    if ((DT_NULL == src0) || (DT_NULL == src1))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src/dst is null ptr");
        return Status::ERROR;
    }

    OpTarget impl_target = m_target;

    // set m_impl
    if (DT_NULL == m_impl.get() || impl_target != m_impl->GetOpTarget())
    {
        m_impl = CreatePsnrImpl(m_ctx, impl_target);
    }

    // run initialize
    PsnrImpl *psnr_impl = dynamic_cast<PsnrImpl*>(m_impl.get());
    if (DT_NULL == psnr_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "psnr_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = psnr_impl->SetArgs(src0, src1, coef_r, result);

    AURA_RETURN(m_ctx, ret);
}

AURA_EXPORTS Status IPsnr(Context *ctx, const Mat &src0, const Mat &src1, DT_F64 coef_r, DT_F64 *result, const OpTarget &target)
{
    Psnr psnr(ctx, target);

    return OpCall(ctx, psnr, &src0, &src1, coef_r, result);
}

PsnrImpl::PsnrImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "Psnr", target), m_src0(DT_NULL),
                                                           m_src1(DT_NULL), m_coef_r(0), m_result(DT_NULL)
{}

Status PsnrImpl::SetArgs(const Array *src0, const Array *src1, DT_F64 coef_r, DT_F64 *result)
{
    if (DT_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if ((DT_NULL == src0) || (DT_NULL == src1))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src/dst is null ptr");
        return Status::ERROR;
    }

    if (DT_NULL == result)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "result is null ptr");
        return Status::ERROR;
    }

    if ((!src0->IsValid()) || (!src1->IsValid()))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst mat is invalid.");
        return Status::ERROR;
    }

    if (!src0->IsEqual(*src1))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src0, src1 must have same size and same type");
        return Status::ERROR;
    }

    m_src0   = src0;
    m_src1   = src1;
    m_coef_r = coef_r;
    m_result = result;
    return Status::OK;
}

std::vector<const Array*> PsnrImpl::GetInputArrays() const
{
    return {m_src0, m_src1};
}

std::string PsnrImpl::ToString() const
{
    std::string str;

    str = "op(Psnr)";
    str += " target(" + GetOpTarget().ToString() + ")\n";

    return str;
}

DT_VOID PsnrImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    std::vector<const Array*> arrays = {m_src0, m_src1};

    if (json_wrapper.SetArray("src", arrays) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray src failed");
        return;
    }

    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src0, m_src1, m_coef_r, *m_result);
}

} // namespace aura