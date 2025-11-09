#ifndef AURA_OPS_MATRIX_NORMALIZE_HPP__
#define AURA_OPS_MATRIX_NORMALIZE_HPP__

#include "aura/ops/matrix/norm.hpp"
#include "norm.hpp"
#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

/**
 * @defgroup ops Operators
 * @{
 *      @defgroup matrix Matrix Process
 *      @{
 *          @defgroup normalize Normalize
 *      @}
 * @}
 */

namespace aura
{

/**
 * @addtogroup normalize
 * @{
 */

/**
 * @brief The matrix normalization operation class.
 *
 * The use of this class for the above-mentioned operation is not recommended.
 * It is recommended to use the `INormalize` API, which internally calls this class.
 * The only recommended scenario for using this class is when the input or output type is 'CLMem'.
 *
 * The approximate internal call within the `INormalize` function is as follows:
 *
 * @code
 * Normalize normalize(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize().
 * return OpCall(ctx, normalize, &src, &dst, alpha, beta, type);
 * @endcode
 */
class AURA_EXPORTS Normalize : public Op
{
public:
    /**
     * @brief Constructor for the Normalize class.
     *
     * @param ctx The pointer to the Context object
     * @param target The platform on which this function runs
     */
    Normalize(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Sets the arguments for normalizing the source array and storing the result in the destination array.
     *
     * For more details, please refer to @ref normlizes_details
     */
    Status SetArgs(const Array *src, Array *dst, DT_F32 alpha, DT_F32 beta, NormType type);
};

/**
 * @brief The matrix normalization operations interface.
 *
 * @anchor normlizes_details
 * This function normalizes the src matrix using the provided parameters (alpha, beta, type) and stores
 * the normalized result in the dst matrix. And in-place operation is not supported.
 *
 * @param ctx The pointer to the Context object
 * @param src The source matrix to be normalized.
 * @param dst The destination matrix to store the normalized result.
 * @param alpha Scaling factor.
 * @param beta Offset added to the scaled values.
 * @param type The type of norm used for normalization.
 * @param target The platform on which this function runs
 *
 * @return Status Status::OK if successful; otherwise, an appropriate error status.
 *
 * ### the supported data types and platforms
 * Platforms    | Src data type                                           | Types
 * -------------|---------------------------------------------------------|-------------------------------------------------------
 * NONE         | U8Cx/S8Cx/U16Cx/S16Cx/F16Cx/U32Cx/S32Cx/F32Cx, x = N    | NORM_INF, NORM_L1, NORM_L2, NORM_L2SQR, NORM_MINMAX
 * NEON         | U8Cx/S8Cx/U16Cx/S16Cx/F16Cx/U32Cx/S32Cx/F32Cx, x = N    | NORM_MINMAX
 * NEON         | U8Cx/S8Cx/U16Cx/S16Cx/F16Cx/F32Cx, x = 1, 2, 3          | NORM_INF, NORM_L1, NORM_L2, NORM_L2SQR,
 *
 * @note N is positive integer. Src and dst must have same size and data type.
 */
AURA_EXPORTS Status INormalize(Context *ctx, const Mat &src, Mat &dst, DT_F32 alpha, DT_F32 beta, NormType type, const OpTarget &target = OpTarget::Default());

/**
 * @}
 */

}// namespace aura

#endif // AURA_OPS_MATRIX_NORMALIZE_HPP__
