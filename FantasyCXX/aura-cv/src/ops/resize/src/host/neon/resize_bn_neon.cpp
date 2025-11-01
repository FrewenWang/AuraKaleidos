#include "resize_impl.hpp"
#include "aura/runtime/logger.h"

namespace aura
{

static Status ResizeBnFastNeon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    Status ret = Status::ERROR;

    switch(src.GetSizes().m_channel)
    {
        case 1:
        {
            ret = ResizeBnFastC1Neon(ctx, src, dst, target);
            break;
        }

        case 2:
        {
            ret = ResizeBnFastC2Neon(ctx, src, dst, target);
            break;
        }

        case 3:
        {
            ret = ResizeBnFastC3Neon(ctx, src, dst, target);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "do not surpport channel more than 3");
            break;
        }
    }

    AURA_RETURN(ctx, ret);
}

Status ResizeBnNeon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    Status ret = Status::ERROR;

    MI_F32 scale_x = static_cast<MI_F64>(src.GetSizes().m_width) / dst.GetSizes().m_width;
    MI_F32 scale_y = static_cast<MI_F64>(src.GetSizes().m_height) / dst.GetSizes().m_height;

    if (NearlyEqual(scale_x, scale_y) &&
        (NearlyEqual(scale_x, 0.25f) || NearlyEqual(scale_x, 0.5f) ||
        NearlyEqual(scale_x, 2.f) || NearlyEqual(scale_x, 4.f)))
    {
        ret = ResizeBnFastNeon(ctx, src, dst, target);
    }
    else
    {
        ret = ResizeBnCommNeon(ctx, src, dst, MI_FALSE, target);
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura