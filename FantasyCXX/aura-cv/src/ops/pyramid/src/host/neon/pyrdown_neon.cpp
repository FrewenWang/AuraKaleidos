#include "pyrdown_impl.hpp"
#include "aura/runtime/logger.h"

namespace aura
{

PyrDownNeon::PyrDownNeon(Context *ctx, const OpTarget &target) : PyrDownImpl(ctx, target)
{}

Status PyrDownNeon::SetArgs(const Array *src, Array *dst, DT_S32 ksize, DT_F32 sigma,
                               BorderType border_type)
{
    if (PyrDownImpl::SetArgs(src, dst, ksize, sigma, border_type) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "PyrDownImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status PyrDownNeon::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst       = dynamic_cast<Mat*>(m_dst);

    if ((DT_NULL == src) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst is null");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    switch (m_ksize)
    {
        case 5:
        {
            ret = PyrDown5x5Neon(m_ctx, *src, *dst, m_kmat, m_border_type, m_target);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "Unsupported ksize");
            return Status::ERROR;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

} //namespace aura