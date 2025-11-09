#ifndef AURA_OPS_FEATURE2D_TOMASI_HPP__
#define AURA_OPS_FEATURE2D_TOMASI_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

/**
 * @defgroup ops Operators
 * @{
 *    @defgroup feature2d Feature Detection and Description
 * @}
 */

namespace aura
{
/**
 * @addtogroup feature2d
 * @{
 */

/**
 * @brief Tomasi corner detector interface class.
 *
 * The use of this class for Tomasi is not recommended. It is recommended to use the `GoodFeaturesToTrack` API,
 * which internally calls this class. The only recommended scenario for using this class is when the input or output type is `CLMem`.
 * 
 * The approximate internal call within the `GoodFeaturesToTrack` function is as follows:
 * 
 * @code
 * Tomasi tomasi(ctx, target);
 * 
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize().
 * return OpCall(ctx, tomasi, &src, ...);
 * @endcode
 */
class AURA_EXPORTS Tomasi : public Op
{
public:
    /**
     * @brief Constructor for Tomasi class.
     *
     * @param ctx The pointer to the Context object
     * @param target The platform on which this function runs
     */
    Tomasi(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Sets the arguments for the Tomasi corner detector operation.
     *
     * For more details, please refer to @ref goodfeaturestotrack_details
     *
     * @note If the type of src or dst is `CLMem` and is an iaura2D memory object, the row pitch should be aligned to a stride,
     * which is obtained through the `GetCLLengthAlignSize` function and is platform-dependent.
     */
    Status SetArgs(const Array *src, std::vector<KeyPoint> &key_points, DT_S32 max_num_corners,
                   DT_F64 quality_level, DT_F64 min_distance, DT_S32 block_size, DT_S32 gradient_size,
                   DT_BOOL use_harris = DT_FALSE, DT_F64 harris_k = 0.04);
};

/**
 * @brief Determines strong corners on an iaura.
 *
 * @anchor goodfeaturestotrack_details
 * The function finds the most prominent corners in the iaura or in the specified iaura region.
 *
 * @param ctx The pointer to the Context object
 * @param src The input iaura, And see the below for the supported data types.
 * @param key_points The output vector containing the detected keypoints.
 * @param max_num_corners The maximum number of corners to detect.
 * @param quality_level The minimal accepted quality of corners.
 * @param min_distance Minimum possible Euclidean distance between the returned corners.
 * @param block_size Size of the averaging block for computing derivative covariation matrix.
 * @param gradient_size Aperture parameter for the Sobel operator.
 * @param use_harris Flag indicating whether to use Harris detector (default is DT_FALSE).
 * @param harris_k Free parameter of the Harris detector (default is 0.04).
 * @param target The platform on which this function runs.
 * 
 * @return Status::OK if successful; otherwise, an appropriate error status.
 * 
 * ### The supported data types and platforms
 * Platforms    | Data type(src or dst)                            
 * -------------|------------------------------------------------- 
 * NONE         | U8C1
 * NEON         | U8C1
 */
AURA_EXPORTS Status GoodFeaturesToTrack(Context *ctx, const Mat &src, std::vector<KeyPoint> &key_points, DT_S32 max_num_corners,
                                        DT_F64 quality_level, DT_F64 min_distance, DT_S32 block_size, DT_S32 gradient_size,
                                        DT_BOOL use_harris = DT_FALSE, DT_F64 harris_k = 0.04, const OpTarget &target = OpTarget::Default());
/**
 * @}
*/
} // namespace aura

#endif // AURA_OPS_FEATURE2D_TOMASI_HPP__