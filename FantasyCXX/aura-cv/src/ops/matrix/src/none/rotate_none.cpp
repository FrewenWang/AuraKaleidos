#include "rotate_impl.hpp"
#include "aura/ops/matrix.h"
#include "aura/runtime/logger.h"

namespace aura
{

static Status RotateCommNone(Context *ctx, const Mat &src, Mat &dst, RotateType type, const OpTarget &target)
{
    Status ret = Status::ERROR;

    switch(type)
    {
        case RotateType::ROTATE_90:
        {
            ret = ITranspose(ctx, src, dst, target);

            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Rotate ROTATE_90 call transpose failed.");
                break;
            }

            ret = IFlip(ctx, dst, dst, FlipType::HORIZONTAL, target);

            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Rotate ROTATE_90 call flip failed.");
                break;
            }

            break;
        }
        case RotateType::ROTATE_180:
        {
            ret = IFlip(ctx, src, dst, FlipType::BOTH, target);

            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Rotate ROTATE_180 call flip failed.");
                break;
            }

            break;
        }
        case RotateType::ROTATE_270:
        {
            ret = ITranspose(ctx, src, dst, target);

            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Rotate ROTATE_270 call transpose failed.");
                break;
            }

            ret = IFlip(ctx, dst, dst, FlipType::VERTICAL, target);

            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Rotate ROTATE_270 call flip failed.");
                break;
            }

            break;
        }
        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "Rotate call with invalid RotateType.");
            ret = Status::ERROR;
            break;
        }
    }

    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "RotateNone failed.");
    }

    AURA_RETURN(ctx, ret);
}


template <typename Tp, RotateType ROTATE_TYPE>
static Status RotateNoneHelper(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    Status ret = Status::ERROR;
    MI_S32 height = dst.GetSizes().m_height;

    switch (dst.GetSizes().m_channel)
    {
        case 1:
        {
            ret = RotateNoneFunctor<Tp, ROTATE_TYPE, 1>()(src, dst, 0, height);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "RotateNoneFunctor with C1 failed.");
            }
            break;
        }
        case 2:
        {
            ret = RotateNoneFunctor<Tp, ROTATE_TYPE, 2>()(src, dst, 0, height);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "RotateNoneFunctor with C2 failed.");
            }
            break;
        }
        case 3:
        {
            ret = RotateNoneFunctor<Tp, ROTATE_TYPE, 3>()(src, dst, 0, height);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "RotateNoneFunctor with C3 failed.");
            }
            break;
        }
        case 4:
        {
            ret = RotateNoneFunctor<Tp, ROTATE_TYPE, 4>()(src, dst, 0, height);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "RotateNoneFunctor with C4 failed.");
            }
            break;
        }
        default:
        {
            ret = RotateCommNone(ctx, src, dst, ROTATE_TYPE, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "RotateCommNone failed.");
            }
            break;
        }
    }

    AURA_RETURN(ctx, ret);
}

template <typename Tp>
static Status RotateNoneHelper(Context *ctx, const Mat &src, Mat &dst, RotateType type, const OpTarget &target)
{
    Status ret = Status::ERROR;

    switch (type)
    {
        case RotateType::ROTATE_90:
        {
            ret = RotateNoneHelper<Tp, RotateType::ROTATE_90>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "RotateNone with ROTATE_90 failed.");
            }
            break;
        }
        case RotateType::ROTATE_180:
        {
            ret = RotateNoneHelper<Tp, RotateType::ROTATE_180>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "RotateNone with ROTATE_180 failed.");
            }
            break;
        }
        case RotateType::ROTATE_270:
        {
            ret = RotateNoneHelper<Tp, RotateType::ROTATE_270>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "RotateNone with ROTATE_270 failed.");
            }
            break;
        }
        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "RotateNone call with invalid RotateType.");
            ret = Status::ERROR;
            break;
        }
    }

    AURA_RETURN(ctx, ret);
}

RotateNone::RotateNone(Context *ctx, const OpTarget &target) : RotateImpl(ctx, target)
{}

Status RotateNone::SetArgs(const Array *src, Array *dst, RotateType type)
{
    if (RotateImpl::SetArgs(src, dst, type) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "RotateImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status RotateNone::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst       = dynamic_cast<Mat*>(m_dst);

    if ((MI_NULL == src) || (MI_NULL == dst))
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
            ret = RotateNoneHelper<MI_U8>(m_ctx, *src, *dst, m_type, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "RotateNone Elem8 failed.");
            }
            break;
        }
        case ElemType::U16:
        case ElemType::S16:
#if defined(AURA_BUILD_HOST)
        case ElemType::F16:
#endif
        {
            ret = RotateNoneHelper<MI_U16>(m_ctx, *src, *dst, m_type, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "RotateNone Elem16 failed.");
            }
            break;
        }
        case ElemType::U32:
        case ElemType::S32:
#if defined(AURA_BUILD_HOST)
        case ElemType::F32:
#endif
        {
            ret = RotateNoneHelper<MI_U32>(m_ctx, *src, *dst, m_type, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "RotateNone Elem32 failed.");
            }
            break;
        }
        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "RotateNone call with invalid ElemType.");
            ret = Status::ERROR;
            break;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura
