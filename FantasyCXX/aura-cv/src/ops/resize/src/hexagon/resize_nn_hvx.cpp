#include "resize_impl.hpp"
#include "aura/runtime/logger.h"

namespace aura
{

Status ResizeNnHvx(Context *ctx, const Mat &src, Mat &dst)
{
    Status ret = Status::ERROR;
    const DT_S32 iwidth = src.GetSizes().m_width;
    const DT_S32 owidth = dst.GetSizes().m_width;

    if ((iwidth == 2 * owidth) || (iwidth == 4 * owidth) ||
        (owidth == 2 * iwidth) || (owidth == 4 * iwidth))
    {
        ret = ResizeNnFastHvx(ctx, src, dst);
    }
    else
    {
        ret = ResizeNnCommHvx(ctx, src, dst);
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura