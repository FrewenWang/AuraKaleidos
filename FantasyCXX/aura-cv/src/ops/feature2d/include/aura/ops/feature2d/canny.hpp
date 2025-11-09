#ifndef AURA_OPS_FEATURE2D_CANNY_HPP__
#define AURA_OPS_FEATURE2D_CANNY_HPP__

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
 * @brief Canny edge detection class.
 *
 * The use of this class for canny is not recommended. It is recommended to use the `Icanny` API,
 * which internally calls this class. The only recommended scenario for using this class is when the input or output type is `CLMem`.
 * 
 * The approximate internal call within the `Icanny` function is as follows:
 * 
 * @code
 * Canny canny(ctx, target);
 * 
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize().
 * return OpCall(ctx, canny, &src, &dst, ...);
 * @endcode
 */
class AURA_EXPORTS Canny : public Op
{
public:
    /**
     * @brief Constructor for the Canny operation.
     *
     * @param ctx The pointer to the Context object
     * @param target The platform on which this function runs
     */
    Canny(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Set the arguments for the Canny operation.
     *
     * For more details, please refer to @ref canny_details
     *
     * @note If the type of src or dst is `CLMem` and is an iaura2D memory object, the row pitch should be aligned to a stride,
     * which is obtained through the `GetCLLengthAlignSize` function and is platform-dependent. 
     */
    Status SetArgs(const Array *src, Array *dst, DT_F64 low_thresh, DT_F64 high_thresh,
                   DT_S32 aperture_size = 3, DT_BOOL l2_gradient = DT_FALSE);

    /**
     * @brief Set the arguments for the Canny operation.
     *
     * For more details, please refer to @ref canny_gradients_details
     *
     * @note If the type of src or dst is `CLMem` and is an iaura2D memory object, the row pitch should be aligned to a stride,
     * which is obtained through the `GetCLLengthAlignSize` function and is platform-dependent. 
     */
    Status SetArgs(const Array *dx, const Array *dy, Array *dst, DT_F64 low_thresh,
                   DT_F64 high_thresh, DT_BOOL l2_gradient = DT_FALSE);
};

/**
 * @brief Performs Canny edge detection on iaura.
 * And in-place operation is not supported.
 *
 * @anchor canny_details
 *
 * @param ctx The pointer to the Context object
 * @param src Reference to the src Mat object. And see the below for the supported data types.
 * @param dst Reference to the destination Mat object. And see the below for the supported data types.
 * @param low_thresh The lower threshold for the hysteresis procedure.
 * @param high_thresh The upper threshold for the hysteresis procedure.
 * @param aperture_size The aperture size for the Sobel operator (default is 3).
 * @param l2_gradient Flag indicating whether to use the L2 norm for gradient magnitude (default is DT_FALSE).
 * @param target The platform on which this function runs.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 * 
 * ### The supported data types and platforms
 * Platforms    | Data type(src)  | Data type(dsy)
 * -------------|-----------------|--------------------
 * NONE         | U8Cx, x = N     | U8C1
 * NEON         | U8Cx, x = N     | U8C1
 *
 * @note N is positive integer.
 */
AURA_EXPORTS Status ICanny(Context *ctx, const Mat &src, Mat &dst, DT_F64 low_thresh, DT_F64 high_thresh,
                           DT_S32 aperture_size = 3, DT_BOOL l2_gradient = DT_FALSE, const OpTarget &target = OpTarget::Default());

/**
 * @brief Performs Canny edge detection with separate x and y gradients.
 * And in-place operation is not supported.
 *
 * @anchor canny_gradients_details
 * 
 * @param ctx The pointer to the Context object
 * @param dx Reference to the x-gradient Mat object. And see the below for the supported data types.
 * @param dy Reference to the y-gradient Mat object. And see the below for the supported data types.
 * @param dst Reference to the destination Mat object. And see the below for the supported data types.
 * @param low_thresh The lower threshold for the hysteresis procedure.
 * @param high_thresh The upper threshold for the hysteresis procedure.
 * @param l2_gradient Flag indicating whether to use the L2 norm for gradient magnitude (default is DT_FALSE).
 * @param target The platform on which this function runs.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 * 
 * ### The supported data types and platforms
 * Platforms    | Data type(dx and dy)  | Data type(dst)
 * -------------|-----------------------|--------------------
 * NONE         | S16Cx, x = N          | U8C1
 * NEON         | S16Cx, x = N          | U8C1
 *
 * @note N is positive integer.
 */
AURA_EXPORTS Status ICanny(Context *ctx, const Mat &dx, const Mat &dy, Mat &dst, DT_F64 low_thresh,
                           DT_F64 high_thresh, DT_BOOL l2_gradient = DT_FALSE, const OpTarget &target = OpTarget::Default());
/**
 * @}
*/
} // namespace aura

#endif // AURA_OPS_FEATURE2D_Canny_HPP__