#ifndef AURA_OPS_MATRIX_MAKE_BORDER_HPP__
#define AURA_OPS_MATRIX_MAKE_BORDER_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

/**
 * @defgroup ops Operators
 * @{
 *      @defgroup matrix Matrix Process
 *      @{
 *          @defgroup make_border Make Border
 *      @}
 * @}
 */

namespace aura
{

/**
 * @addtogroup make_border
 * @{
 */

/**
 * @brief Interface class representing an operations to add borders to a matrix.
 *
 * The use of this class for adding borders to a matrix is not recommended.
 * It is recommended to use the `IMakeBorder` API, which internally calls this class.
 *
 * The approximate internal call within the `IMakeBorder` function is as follows:
 *
 * @code
 * MakeBorder make_border(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize().
 * return OpCall(ctx, make_border, &src, &dst, top, bottom, left, right, type, border_value);
 * @endcode
 */
class AURA_EXPORTS MakeBorder : public Op
{
public:
    /**
     * @brief Constructor for the MakeBorder class.
     *
     * @param ctx The pointer to the Context object
     * @param target The platform on which this function runs
     */
    MakeBorder(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Sets the arguments for adding borders to the array.
     *
     * For more details, please refer to @ref makeborder_details
     */
    Status SetArgs(const Array *src, Array *dst, DT_S32 top, DT_S32 bottom,
                   DT_S32 left, DT_S32 right, BorderType type, const Scalar &border_value = Scalar());
};

/**
 * @brief Adds borders to the source matrix.
 *
 * @anchor makeborder_details
 * This function adds borders to the src matrix and stores the result in the dst matrix
 * based on the specified border sizes, type, and border value. And in-place operation is not supported.
 *
 * @param ctx The pointer to the Context object
 * @param src The source matrix to which borders will be added.
 * @param dst The destination matrix to store the result with added borders.
 * @param top The size of the top border.
 * @param bottom The size of the bottom border.
 * @param left The size of the left border.
 * @param right The size of the right border.
 * @param type The border type for handling border pixels.
 * @param border_value The scalar values for border pixels used only when border_type is BorderType::CONSTANT.
 * @param target The platform on which this function runs.
 *
 * @return Status Status::OK if successful; otherwise, an appropriate error status.
 *
 * ### the supported data types and platforms
 * Platforms    | Data type(src or dst)
 * -------------|--------------------------------------------------------------
 * NONE         | U8Cx/S8Cx/U16Cx/S16Cx/F16Cx/U32Cx/S32Cx/F32Cx/F64Cx, x = N
 *
 * @note 1. N is positive integer; <br>
 *       2. (top, bottom, left, right) must be greater than or equal to 0; <br>
 *       3. The above implementations support all border types(CONSTANT/REPLICATE/REFLECT_101).
 */
AURA_EXPORTS Status IMakeBorder(Context *ctx, const Mat &src, Mat &dst, DT_S32 top, DT_S32 bottom,
                                 DT_S32 left, DT_S32 right, BorderType type, const Scalar &border_value = Scalar(),
                                 const OpTarget &target = OpTarget::Default());

/**
 * @}
 */

} // namespace aura

#endif // AURA_OPS_MATRIX_MAKE_BORDER_HPP__
