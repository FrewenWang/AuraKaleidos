#include "resize_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"

namespace aura
{

Status ResizeAreaHvx(Context *ctx, const Mat &src, Mat &dst)
{
    DT_F32 scale_x = static_cast<DT_F32>(src.GetSizes().m_width) / dst.GetSizes().m_width;
    DT_F32 scale_y = static_cast<DT_F32>(src.GetSizes().m_height) / dst.GetSizes().m_height;
    DT_S32 int_scale_x = SaturateCast<DT_S32>(scale_x);
    DT_S32 int_scale_y = SaturateCast<DT_S32>(scale_y);

    Status ret = Status::ERROR;

    DT_S32 fast_flag = (NearlyEqual(scale_x, int_scale_x) && NearlyEqual(scale_y, int_scale_y)) ? 1 : 0;
    if ((NearlyEqual(scale_x, scale_y) &&
        (NearlyEqual(scale_x, 0.25f) || NearlyEqual(scale_x, 0.5f) ||
         NearlyEqual(scale_x, 2.f)   || NearlyEqual(scale_x, 4.f)))
         || fast_flag)
    {
        ret = ResizeAreaFastHvx(ctx, src, dst);
    }
    else
    {
        ret = ResizeAreaCommonHvx(ctx, src, dst);
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura
