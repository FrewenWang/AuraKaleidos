#include "resize_impl.hpp"
#include "aura/runtime/logger.h"

namespace aura
{

Status ResizeBnHvx(Context *ctx, const Mat &src, Mat &dst)
{
    Status ret = Status::ERROR;
    const MI_S32 iwidth  = src.GetSizes().m_width;
    const MI_S32 owidth  = dst.GetSizes().m_width;
    const MI_S32 iheight = src.GetSizes().m_height;
    const MI_S32 oheight = dst.GetSizes().m_height;

    if (((iwidth  == 2 * owidth)  || (iwidth  == 4 * owidth)) &&
        ((iheight == 2 * oheight) || (iheight == 4 * oheight)))
    {
        ret = ResizeBnFastDnHvx(ctx, src, dst);
    }
    else if (((owidth == 2 * iwidth) && (oheight == 2 * iheight)) ||
             ((owidth == 4 * iwidth) && (oheight == 4 * oheight)))
    {
        ret = ResizeBnFastUpHvx(ctx, src, dst);
    }
    else
    {
        ret = ResizeBnCommHvx(ctx, src, dst, MI_FALSE);
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura