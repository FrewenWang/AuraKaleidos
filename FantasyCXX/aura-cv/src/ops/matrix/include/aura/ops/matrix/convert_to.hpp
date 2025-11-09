#ifndef AURA_OPS_MATRIX_CONVERT_TO_HPP__
#define AURA_OPS_MATRIX_CONVERT_TO_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

/**
 * @defgroup ops Operators
 * @{
 *      @defgroup matrix Matrix Process
 *      @{
 *          @defgroup convert_to Convert To
 *      @}
 * @}
 */

namespace aura
{

/**
 * @addtogroup convert_to
 * @{
 */

/**
 * @brief The matrix data type conversion operation class.
 *
 * The use of this class for conversion operations is not recommended.
 * It is recommended to use the `IConvertTo` API, which internally calls this class.
 * The only recommended scenario for using this class is when the input or output type is 'CLMem'.
 *
 * The approximate internal call within the `IConvertTo` function is as follows:
 *
 * @code
 * ConvertTo convert_to(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize().
 * return OpCall(ctx, convert_to, &src, &dst, alpha, beta);
 * @endcode
 */
class AURA_EXPORTS ConvertTo : public Op
{
public:
    /**
     * @brief Constructor for the ConvertTo class.
     *
     * @param ctx ctx The pointer to the Context object
     * @param target The platform on which this function runs
     */
    ConvertTo(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Set the arguments for the matrix conversion operation.
     *
     * For more details, please refer to @ref convert_to_details.
     *
     * @note If the type of src or dst is `CLMem` and is an iaura2D memory object, the row pitch
     * should be aligned to a stride, which is obtained through the `GetCLLengthAlignSize` function and is
     * platform-dependent.
     */
    Status SetArgs(const Array *src, Array *dst, DT_F32 alpha, DT_F32 beta);

    /**
     * @brief Generate covertto opencl precompiled cache.
     *
     * @param src_elem_type The covertto src array element type.
     * @param dst_elem_type The covertto dst array element type.
     * @param alpha The alpha scaling factor applied during conversion (defaults to 1.0f).
     * @param beta The beta scaling factor applied during conversion (defaults to 0.0f).
     */
    static Status CLPrecompile(Context *ctx, ElemType src_elem_type, ElemType dst_elem_type, DT_F32 alpha, DT_F32 beta);
};


/**
 * @brief Converts the input matrix to the specified data type with optional scaling.
 *
 * @anchor convert_to_details
 * This function converts the elements of the src matrix to the dst matrix
 * using alpha and beta scaling factors. And in-place operation is not supported.
 *
 * @param ctx ctx The pointer to the Context object
 * @param src The source matrix to be converted.
 * @param dst The destination matrix to store the converted elements.
 * @param alpha The alpha scaling factor applied during conversion (defaults to 1.0f).
 * @param beta The beta scaling factor applied during conversion (defaults to 0.0f).
 * @param target The platform on which this function runs
 *
 * @return Status Status::OK if successful; otherwise, an appropriate error status.
 *
 * ### the supported data types and platforms
 * Platforms    | Data type(src or dst)
 * -------------|---------------------------------------------------------------
 * NONE         | U8Cx/S8Cx/U16Cx/S16Cx/F16Cx/U32Cx/S32Cx/F32Cx, x = N
 * NEON         | U8Cx/S8Cx/U16Cx/S16Cx/F16Cx/U32Cx/S32Cx/F32Cx, x = N
 * OpenCL       | U8Cx/S8Cx/U16Cx/S16Cx/F16Cx/U32Cx/S32Cx/F32Cx, x = N
 *
 * @note N is positive integer.
 */
AURA_EXPORTS Status IConvertTo(Context *ctx, const Mat &src, Mat &dst, DT_F32 alpha = 1.0f, DT_F32 beta = 0.0f,
                               const OpTarget &target = OpTarget::Default());

/**
 * @}
 */

} // namespace aura

#endif // AURA_OPS_MATRIX_CONVERT_TO_HPP__
