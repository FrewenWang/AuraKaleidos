#ifndef AURA_OPS_MATRIX_MUL_SPECTRUMS_HPP__
#define AURA_OPS_MATRIX_MUL_SPECTRUMS_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

/**
 * @defgroup ops Operators
 * @{
 *      @defgroup matrix Matrix Process
 *      @{
 *          @defgroup mul_spectrums MulSpectrums
 *      @}
 * @}
 */

namespace aura
{

/**
 * @addtogroup mul_spectrums
 * @{
 */

/**
 * @brief Interface class representing an operation to multiply two spectrum matrices.
 *
 * The use of this class for the above-mentioned operation is not recommended.
 * It is recommended to use the `IMulSpectrums` API, which internally calls this class.
 *
 * The approximate internal call within the `IMulSpectrums` function is as follows:
 *
 * @code
 * MulSpectrums mul_spectrums(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize().
 * return OpCall(ctx, mul_spectrums, &src0, &src1, &dst);
 * @endcode
 */
class AURA_EXPORTS MulSpectrums : public Op
{
public:
    /**
     * @brief Constructor for the MulSpectrums class.
     *
     * @param ctx The pointer to the Context object
     * @param target The platform on which this function runs
     */
    MulSpectrums(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Sets the arguments for multiplying two spectrums.
     *
     * For more details, please refer to @ref mulspectrums_details
     */
    Status SetArgs(const Array *src0, const Array *src1, Array *dst, DT_BOOL conj_src1);

    /**
     * @brief Generate MulSpectrums opencl precompiled cache.
     *
     * @param elem_type The element type of the src/dst array.
     * @param conj_src1 The flag that conjugates the second input array before the multiplication.
     */
    static Status CLPrecompile(Context *ctx, ElemType elem_type, DT_BOOL conj_src1);
};

/**
 * @brief Multiplies two source matrices (spectrums).
 *
 * @anchor mulspectrums_details
 * This function multiplies two source matrices (src0 and src1), representing spectrums, and stores the result
 * in the dst matrix. And in-place operation is not supported.
 *
 * @param ctx The pointer to the Context object
 * @param src0 The first source matrix (spectrum) for multiplication.
 * @param src1 The second source matrix (spectrum) for multiplication.
 * @param dst The destination matrix to store the multiplied result.
 * @param conj_src1 The flag that conjugates the second input array before the multiplication
 * @param target The platform on which this function runs
 *
 * @return Status Status::OK if successful; otherwise, an appropriate error status.
 *
 * ### the supported data types and platforms
 * Platforms    | Data type(src0, src1 and dst)
 * -------------|---------------------------------
 * NONE         | F32C2
 * NEON         | F32C2
 * OpenCL       | F32C2
 */
AURA_EXPORTS Status IMulSpectrums(Context *ctx, const Mat &src0, const Mat &src1, Mat &dst, DT_BOOL conj_src1, const OpTarget &target = OpTarget::Default());

/**
 * @}
 */

}// namespace aura

#endif// AURA_OPS_MATRIX_MUL_SPECTRUMS_HPP__
