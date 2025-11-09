#ifndef AURA_OPS_MATRIX_TRANSPOSE_HPP__
#define AURA_OPS_MATRIX_TRANSPOSE_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

/**
 * @defgroup ops Operators
 * @{
 *      @defgroup matrix Matrix Process
 *      @{
 *          @defgroup rotate_transpose Rotate And Transpose
 *      @}
 * @}
 */

namespace aura
{

/**
 * @addtogroup rotate_transpose
 * @{
 */

/**
 * @brief The matrix transpose operation class.
 *
 * The use of this class for the above-mentioned operation is not recommended.
 * It is recommended to use the `ITranspose` API, which internally calls this class.
 * The only recommended scenario for using this class is when the input or output type is 'CLMem'.
 *
 * The approximate internal call within the `ITranspose` function is as follows:
 *
 * @code
 * Transpose transpose(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize().
 * return OpCall(ctx, transpose, &src, &dst);
 * @endcode
 */
class AURA_EXPORTS Transpose : public Op
{
public:
    /**
     * @brief Constructor for the Transpose class.
     *
     * @param ctx The pointer to the Context object
     * @param target The platform on which this function runs
     */
    Transpose(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Sets the arguments for transposing the source array and storing the result in the destination array.
     *
     * For more details, please refer to @ref transposes_details
     *
     * @note If the type of src or dst is `CLMem` and is an iaura2D memory object, the row pitch
     * should be aligned to a stride, which is obtained through the `GetCLLengthAlignSize` function and is
     * platform-dependent.
     */
    Status SetArgs(const Array *src, Array *dst);

    /**
     * @brief Generate transpose opencl precompiled cache.
     *
     * @param elem_type The transpose src/dst array element type.
     * @param ochannel The transpose dst array channel.
     */
    static Status CLPrecompile(Context *ctx, ElemType elem_type, DT_S32 ochannel);
};

/**
 * @brief Performs a matrix transpose operation.
 *
 * @anchor transposes_details
 * This function transposes the src matrix and stores the transposed result in the dst matrix
 * based on the specified target. And in-place operation is not supported.
 *
 * @param ctx The pointer to the Context object
 * @param src The source matrix to be transposed.
 * @param dst The destination matrix to store the transposed result.
 * @param target The platform on which this function runs
 *
 * @return Status Status::OK if successful; otherwise, an appropriate error status.
 *
 * ### the supported data types and platforms
 * Platforms    | Data type(src or dst)
 * -------------|------------------------------------------------------------------
 * NONE         | U8Cx/S8Cx/U16Cx/S16Cx/F16Cx/U32Cx/S32Cx/F32Cx, x = 1, 2, 3, 4
 * NEON         | U8Cx/S8Cx/U16Cx/S16Cx/F16Cx/U32Cx/S32Cx/F32Cx, x = 1, 2, 3, 4
 * OpenCL       | U8Cx/S8Cx/U16Cx/S16Cx/F16Cx/U32Cx/S32Cx/F32Cx, x = 1, 2, 3, 4
 *
 * @note Src and dst must have same size and data type.
 */
AURA_EXPORTS Status ITranspose(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target = OpTarget::Default());

/**
 * @}
 */

}// namespace aura

#endif // AURA_OPS_MATRIX_TRANSPOSE_HPP__
