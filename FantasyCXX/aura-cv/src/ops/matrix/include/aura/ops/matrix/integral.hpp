#ifndef AURA_OPS_MATRIX_INTEGRAL_HPP__
#define AURA_OPS_MATRIX_INTEGRAL_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

/**
 * @defgroup ops Operators
 * @{
 *      @defgroup matrix Matrix Process
 *      @{
 *          @defgroup integral Integral
 *      @}
 * @}
 */

namespace aura
{

/**
 * @addtogroup integral
 * @{
 */

/**
 * @brief Interface class representing of integral and squared integral iauras.
 *
 * The use of this class for computation of integral and squared integral iauras operation is not recommended.
 * It is recommended to use the `IIntegral` API, which internally calls this class.
 * The only recommended scenario for using this class is when the input or output type is 'CLMem'.
 *
 * The approximate internal call within the `IIntegral` function is as follows:
 *
 * @code
 * Integral integral(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize().
 * return OpCall(ctx, integral, &src, &dst, &dst_sq);
 * @endcode
 */
class AURA_EXPORTS Integral : public Op
{
public:
    /**
     * @brief Constructor for the Integral class.
     *
     * @param ctx The pointer to the Context object
     * @param target The platform on which this function runs
     */
    Integral(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Sets the arguments for computing integral and squared integral iauras.
     *
     * For more details, please refer to @ref integral_details.
     */
    Status SetArgs(const Array *src, Array *dst, Array *dst_sq = MI_NULL);
};

/**
 * @brief Computes the integral and squared integral iauras of the source matrix.
 *
 * @anchor integral_details
 * The integral iaura is an auxiliary data structure where each pixel value represents the cumulative sum of all pixels
 * in the original iaura above and to the left of that pixel position. The squared integral iaura is similar, but each
 * pixel value represents the cumulative sum of the squared pixel values in the original iaura. And in-place operation
 * is not supported.
 *
 * @param ctx The pointer to the Context object
 * @param src The source matrix to compute the integral and squared integral iauras.
 * @param dst The destination matrix to store the computed integral iaura.
 * @param dst_sq The destination matrix to store the computed squared integral iaura.
 * @param target The platform on which this function runs
 *
 * @return Status Status::OK if successful; otherwise, an appropriate error status.
 *
 * Mode                  | Platforms   | Data type(src, dst, dst_sq)
 *-----------------------|-------------|-------------------------------------------------------------------------------------------------------------------
 * NORMAL + SQUARE       | NONE        | (U8Cx, U32Cx, F64Cx),  (U8Cx, F32Cx, F64Cx),  (U8Cx, F64Cx, F64Cx),  x = 1, 2, 3, 4
 * NORMAL + SQUARE       | NONE        | (U8Cx, U32Cx, U32Cx),  (U8Cx, F32Cx, U32Cx),  (U8Cx, F64Cx, U32Cx),  x = 1, 2, 3, 4
 * NORMAL + SQUARE       | NONE        | (S8Cx, S32Cx, F64Cx),  (S8Cx, F32Cx, F64Cx),  (S8Cx, F64Cx, F64Cx),  x = 1, 2, 3, 4
 * NORMAL + SQUARE       | NONE        | (S8Cx, S32Cx, U32Cx),  (S8Cx, F32Cx, U32Cx),  (S8Cx, F64Cx, U32Cx),  x = 1, 2, 3, 4
 * NORMAL + SQUARE       | NONE        | (U16Cx, S32Cx, F64Cx), (U16Cx, F32Cx, F64Cx), (U16Cx, F64Cx, F64Cx), x = 1, 2, 3, 4
 * NORMAL + SQUARE       | NONE        | (S16Cx, S32Cx, F64Cx), (S16Cx, F32Cx, F64Cx), (S16Cx, F64Cx, F64Cx), x = 1, 2, 3, 4
 * NORMAL + SQUARE       | NONE        | (F32Cx, F32Cx, F64Cx), (F32Cx, F64Cx, F64Cx), x = 1, 2, 3, 4
 * NORMAL(dst_sq = null) | NONE        | (U8Cx, U32Cx),  (U8Cx, F32Cx),  (U8Cx, F64Cx),  x = 1, 2, 3, 4
 * NORMAL(dst_sq = null) | NONE        | (S8Cx, S32Cx),  (S8Cx, F32Cx),  (S8Cx, F64Cx),  x = 1, 2, 3, 4
 * NORMAL(dst_sq = null) | NONE        | (U16Cx, U32Cx), (U16Cx, F32Cx), (U16Cx, F64Cx), x = 1, 2, 3, 4
 * NORMAL(dst_sq = null) | NONE        | (S16Cx, S32Cx), (S16Cx, F32Cx), (S16Cx, F64Cx), x = 1, 2, 3, 4
 * NORMAL(dst_sq = null) | NONE        | (F32Cx, F32Cx), (F32Cx, F64Cx), x = 1, 2, 3, 4
 * SQUARE(dst = null)    | NONE        | (U8Cx, F64Cx),  (S8Cx, F64Cx), (U16Cx, F64Cx), (S16Cx, F64Cx), (F32Cx, F64Cx), x = 1, 2, 3, 4
 * SQUARE(dst = null)    | NONE        | (U8Cx, U32Cx),  (S8Cx, U32Cx), x = 1, 2, 3, 4
 * NORMAL(dst_sq = null) | NEON        | (U8Cx, U32Cx),  (S8Cx, S32Cx), x = 1
 * NORMAL(dst_sq = null) | NEON        | (U8Cx, F32Cx),  (S8Cx, F32Cx), x = 1
 * SQUARE(dst = null)    | NEON        | (U8Cx, U32Cx),  (S8Cx, U32Cx), x = 1
 * NORMAL + SQUARE       | NEON        | (U8Cx, F32Cx, U32Cx),  (U8Cx, U32Cx, U32Cx), (S8Cx, S32Cx, U32Cx), (S8Cx, F32Cx, U32Cx), x = 1
 * NORMAL(dst_sq = null) | HVX         | (U8Cx, U32Cx),  (S8Cx, S32Cx), x = 1, 2
 */
AURA_EXPORTS Status IIntegral(Context *ctx, const Mat &src, Mat &dst, Mat &dst_sq, const OpTarget &target = OpTarget::Default());

/**
 * @}
 */

}// namespace aura

#endif // AURA_OPS_MATRIX_INTEGRAL_HPP__
