#include "transpose_impl.hpp"
#include "aura/runtime/logger.h"

namespace aura
{

TransposeNeon::TransposeNeon(Context *ctx, const OpTarget &target) : TransposeImpl(ctx, target)
{}

Status TransposeNeon::SetArgs(const Array *src, Array *dst)
{
    if (TransposeImpl::SetArgs(src, dst) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "TransposeImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status TransposeNeon::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst       = dynamic_cast<Mat*>(m_dst);

    if ((DT_NULL == src) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is null");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    switch (src->GetElemType())
    {
        case ElemType::U8:
        case ElemType::S8:
        {
            ret = TransposeU8Neon(m_ctx, *src, *dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "TransposeU8Neon failed.");
            }
            break;
        }
        case ElemType::U16:
        case ElemType::S16:
#if defined(AURA_ENABLE_NEON_FP16)
        case ElemType::F16:
#endif
        {
            ret = TransposeU16Neon(m_ctx, *src, *dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "TransposeU16Neon failed.");
            }
            break;
        }
        case ElemType::U32:
        case ElemType::S32:
        case ElemType::F32:
        {
            ret = TransposeU32Neon(m_ctx, *src, *dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "TransposeU32Neon failed.");
            }
            break;
        }
        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "TransposeNeon with invalid ElemType.");
            ret = Status::ERROR;
            break;
        }
    }

    AURA_RETURN(m_ctx, ret);
}
} // namespace aura