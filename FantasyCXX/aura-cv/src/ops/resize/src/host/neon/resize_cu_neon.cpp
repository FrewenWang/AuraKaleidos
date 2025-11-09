#include "resize_impl.hpp"
#include "aura/runtime/logger.h"

namespace aura
{

static Status ResizeCuFastNeon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    Status ret = Status::ERROR;

    switch(src.GetSizes().m_channel)
    {
        case 1:
        {
            ret = ResizeCuFastC1Neon(ctx, src, dst, target);
            break;
        }

        case 2:
        {
            ret = ResizeCuFastC2Neon(ctx, src, dst, target);
            break;
        }

        case 3:
        {
            ret = ResizeCuFastC3Neon(ctx, src, dst, target);
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

Status ResizeCuNeon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    DT_F32 scale_x = static_cast<DT_F64>(src.GetSizes().m_width) / dst.GetSizes().m_width;
    DT_F32 scale_y = static_cast<DT_F64>(src.GetSizes().m_height) / dst.GetSizes().m_height;

    Status ret = Status::ERROR;

    if (NearlyEqual(scale_x, scale_y) && (NearlyEqual(scale_x, 2.f) || NearlyEqual(scale_x, 4.f)))
    {
        ret = ResizeCuFastNeon(ctx, src, dst, target);
    }
    else
    {
        ret = ResizeCuCommNeon(ctx, src, dst, target);
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura