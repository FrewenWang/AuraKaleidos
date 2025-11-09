#include "resize_impl.hpp"
#include "aura/runtime/logger.h"

namespace aura
{

ResizeNeon::ResizeNeon(Context *ctx, const OpTarget &target) : ResizeImpl(ctx, target)
{}

Status ResizeNeon::SetArgs(const Array *src, Array *dst, InterpType type)
{
    if (ResizeImpl::SetArgs(src, dst, type) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ResizeImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status ResizeNeon::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst       = dynamic_cast<Mat*>(m_dst);

    if ((DT_NULL == src) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst is null");
        return Status::ERROR;
    }

    if (src->IsEqual(*dst))
    {
        return src->CopyTo(*dst);
    }

    Status ret = Status::ERROR;

    switch (m_type)
    {
        case InterpType::NEAREST:
        {
            ret = ResizeNnNeon(m_ctx, *src, *dst, m_target);
            break;
        }

        case InterpType::LINEAR:
        {
            ret = ResizeBnNeon(m_ctx, *src, *dst, m_target);
            break;
        }

        case InterpType::CUBIC:
        {
            ret = ResizeCuNeon(m_ctx, *src, *dst, m_target);
            break;
        }

        case InterpType::AREA:
        {
            ret = ResizeAreaNeon(m_ctx, *src, *dst, m_target);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "interp type error");
            ret = Status::ERROR;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura