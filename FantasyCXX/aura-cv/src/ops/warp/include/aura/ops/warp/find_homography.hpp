#ifndef AURA_OPS_WARP_FINDHOMOGRAPHY_HPP__
#define AURA_OPS_WARP_FINDHOMOGRAPHY_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

/**
 * @defgroup ops Operators
 * @{
 *    @defgroup warp Warp
 * @}
 */

namespace aura
{
/**
 * @addtogroup warp
 * @{
 */

/**
 * @brief Find a homography matrix for a set of src and dst points.
 *
 * This function calculates the 3x3 homography matrix between the src points and dst points using RANSAC method.
 *
 * @param ctx The pointer to the Context object.
 * @param src_points The source points.
 * @param dst_points The destination points.
 * @param reproj_threshold The maximum allowed reprojection error to treat a point pair as an inlier, default is 3.0.
 * @param max_iters The maximum number of iterations, default is 2000.
 * @param confidence Confidence level, between 0 and 1, default is 0.995.
 *
 * @return The homography matrix(3x3, MI_F64). If the function fails, an empty matrix will be returned.
 */
AURA_EXPORTS Mat FindHomography(Context *ctx, const std::vector<Point2> &src_points, const std::vector<Point2> &dst_points,
                                MI_F64 reproj_threshold = 3.0, MI_S32 max_iters = 2000, MI_F64 confidence = 0.995);

/**
 * @}
 */
} // namespace aura

#endif // AURA_OPS_WARP_FINDHOMOGRAPHY_HPP__