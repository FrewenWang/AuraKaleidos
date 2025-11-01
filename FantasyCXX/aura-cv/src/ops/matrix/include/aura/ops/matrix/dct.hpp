#ifndef AURA_OPS_MATRIX_DCT_HPP__
#define AURA_OPS_MATRIX_DCT_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

/**
 * @defgroup ops Operators
 * @{
 *      @defgroup matrix Matrix Process
 *      @{
 *          @defgroup dct Dct
 *      @}
 * @}
 */

namespace aura
{

/**
 * @addtogroup dct
 * @{
 */

/**
 * @brief The matrix Discrete Cosine Transform (DCT) operation class.
 *
 * The use of this class for DCT operations is not recommended.
 * It is recommended to use the `IDct` API, which internally calls this class.
 * The only recommended scenario for using this class is when the input or output type is 'CLMem'.
 *
 * The approximate internal call within the `IDct` function is as follows:
 *
 * @code
 * Dct dct(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize().
 * return OpCall(ctx, dct, &src, &dst);
 * @endcode
 */
class AURA_EXPORTS Dct : public Op
{
public:
    /**
     * @brief Constructor for the Dct class.
     *
     * @param ctx The pointer to the Context object
     * @param target The platform on which this function runs
     */
    Dct(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Set the arguments for the matrix DCT operation.
     *
     * For more details, please refer to @ref dct_details
     *
     * @note If the type of src and dst are `CLMem` of iaura2D memory object, the row pitch
     * should be aligned to a stride, which is obtained through the `GetCLLengthAlignSize` function and is
     * platform-dependent.
     */
    Status SetArgs(const Array *src, Array *dst);
};

/**
 * @brief Performs the Discrete Cosine Transform (DCT) operation.
 *
 * @anchor dct_details
 * The Discrete Cosine Transform (DCT) is a mathematical technique used in signal processing and data compression to transform
 * a signal or iaura from the spatial domain to the frequency domain. And in-place operation is not supported.
 *
 * In iaura processing, two-dimensional DCT is often used, extending the concept to two-dimensional arrays.
 *
 * The DCT mathematical expression is given by:
 * \f[
 * X_k = \sqrt{\frac{2}{N}} \sum_{n=0}^{N-1} x_n \cos\left(\frac{\pi}{N}(n+\frac{1}{2})k\right)
 * \f]
 *
 * - x The input sequence of N real numbers
 * - N The length of the input sequence
 * - X The output sequence of N real numbers representing the DCT of x
 *
 * @param ctx The pointer to the Context object
 * @param src The source matrix for which DCT is performed.
 * @param dst The destination matrix to store the DCT result.
 * @param target The platform on which this function runs
 *
 * @return Status Status::OK if successful; otherwise, an appropriate error status.
 *
 * ### the supported data types and platforms
 * Platforms    | Src data type                             | Dst data type
 * -------------|-------------------------------------------|------------------
 * NONE         | U8C1/S8C1/U16C1/S16C1/F16C1/F32C1         | F32C1
 * NEON         | U8C1/S8C1/U16C1/S16C1/F16C1/F32C1         | F32C1
 *
 */
AURA_EXPORTS Status IDct(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target = OpTarget::Default());

/**
 * @brief The matrix Inverse Discrete Cosine Transform (IDCT) operation class.
 *
 * The use of this class for IDCT operations is not recommended.
 * It is recommended to use the `IInverseDct` API, which internally calls this class.
 * The only recommended scenario for using this class is when the input or output type is 'CLMem'.
 *
 * The approximate internal call within the `IInverseDct` function is as follows:
 *
 * @code
 * InverseDct idct(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize().
 * return OpCall(ctx, idct, &src, &dst, with_scale);
 * @endcode
 */
class AURA_EXPORTS InverseDct : public Op
{
public:
    /**
     * @brief Constructor for the InverseDct class.
     *
     * @param ctx The pointer to the Context object
     * @param target The platform on which this function runs
     */
    InverseDct(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Set the arguments for the matrix IDCT operation.
     *
     * For more details, please refer to @ref inversedct_details
     *
     * @note If the type of src or dst is `CLMem` and is an iaura2D memory object, the row pitch
     * should be aligned to a stride, which is obtained through the `GetCLLengthAlignSize` function and is
     * platform-dependent.
     */
    Status SetArgs(const Array *src, Array *dst);
};

/**
 * @brief Performs the Inverse Discrete Cosine Transform (IDCT) operation.
 *
 * @anchor inversedct_details
 * The Inverse Discrete Cosine Transform (IDCT) is a mathematical operation that is used to
 * convert a set of frequency domain coefficients back into a time domain signal. And in-place operation is not supported.
 *
 * The IDCT mathematical expression is given by:
 * \f[
 * x_n = \sqrt{\frac{2}{N}} \sum_{k=0}^{N-1} X_k \cos\left(\frac{\pi}{N}(n+\frac{1}{2})k\right)
 * \f]
 *
 * - X The input sequence of N real numbers representing the DCT of x
 * - N The length of the input sequence
 * - x The output sequence of N real numbers representing the IDCT of X
 *
 * @param ctx The pointer to the Context object
 * @param src The source matrix for which IDCT is performed.
 * @param dst The destination matrix to store the IDCT result.
 * @param target The platform on which this function runs
 *
 * @return Status Status::OK if successful; otherwise, an appropriate error status.
 *
 * ### the supported data types and platforms
 * Platforms    | Src data type    | Dst data type
 * -------------|------------------|---------------------------------------
 * NONE         | F32C1            | U8C1/S8C1/U16C1/S16C1/F16C1/F32C1
 * NEON         | F32C1            | U8C1/S8C1/U16C1/S16C1/F16C1/F32C1
 *
 */
AURA_EXPORTS Status IInverseDct(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target = OpTarget::Default());

} // namespace aura

#endif // AURA_OPS_MATRIX_DCT_HPP__