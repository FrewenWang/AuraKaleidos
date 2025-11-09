/** @brief     : pyramid_comm header for aura
*  @file       : pyramid_comm.hpp
*  @author     : zhangpengfei10@xiaomi.com
*  @version    : 1.0.0
*  @date       : Sep. 4, 2023
*  @Copyright  : Copyright 2023 Xiaomi Mobile Software Co., Ltd. All Rights reserved.
*/

#ifndef AURA_OPS_PYRAMID_PYRAMID_COMM_HPP__
#define AURA_OPS_PYRAMID_PYRAMID_COMM_HPP__

#include "aura/config.h"
#include "aura/ops/core.h"

#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
#  define AURA_OPS_PYRAMID_PACKAGE_NAME              "aura.ops.pyramid"
#endif

namespace aura
{

template <typename Tp, DT_U32 Q>
AURA_INLINE Mat GetPyrKernelMat(Context *ctx, const std::vector<DT_F32> &kernel)
{
    const DT_S32 ksize = kernel.size();

    Mat kmat(ctx, GetElemType<Tp>(), Sizes3(1, ksize, 1));
    Tp *ker_row = kmat.Ptr<Tp>(0);

    DT_S32 sum = 0;
    DT_F32 err = 0.f;

    for (DT_S32 i = 0; i < ksize / 2; i++)
    {
        DT_F32 tmp             = kernel[i] * (1 << Q) + err;
        Tp result              = static_cast<Tp>(Round(tmp));
        err                    = tmp - (DT_F32)result;
        ker_row[i]             = result;
        ker_row[ksize - 1 - i] = result;
        sum += result;
    }

    ker_row[ksize / 2] = (1 << Q) - sum * 2;

    return kmat;
};

} // namespace aura

#endif // AURA_OPS_PYRAMID_PYRAMID_COMM_HPP__