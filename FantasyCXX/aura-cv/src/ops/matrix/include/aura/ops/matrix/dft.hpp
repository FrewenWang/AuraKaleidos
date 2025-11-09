#ifndef AURA_OPS_MATRIX_DFT_HPP__
#define AURA_OPS_MATRIX_DFT_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

/**
 * @defgroup ops Operators
 * @{
 *      @defgroup matrix Matrix Process
 *      @{
 *          @defgroup dft Dft
 *      @}
 * @}
 */

namespace aura
{

/**
 * @addtogroup dft
 * @{
 */

/**
 * @brief The matrix Discrete Fourier Transform (DFT) operation class.
 *
 * The use of this class for DFT operations is not recommended.
 * It is recommended to use the `IDft` API, which internally calls this class.
 * The only recommended scenario for using this class is when the input or output type is 'CLMem'.
 *
 * The approximate internal call within the `IDft` function is as follows:
 *
 * @code
 * Dft dft(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize().
 * return OpCall(ctx, dft, &src, &dst);
 * @endcode
 */
class AURA_EXPORTS Dft : public Op
{
public:
    /**
     * @brief Constructor for the Dft class.
     *
     * @param ctx The pointer to the Context object
     * @param target The platform on which this function runs
     */
    Dft(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Set the arguments for the matrix DFT operation.
     *
     * For more details, please refer to @ref dft_details
     *
     * @note If the type of src or dst is `CLMem` and is an iaura2D memory object, the row pitch
     * should be aligned to a stride, which is obtained through the `GetCLLengthAlignSize` function and is
     * platform-dependent.
     */
    Status SetArgs(const Array *src, Array *dst);

    /**
     * @brief Generate dft opencl precompiled cache.
     *
     * @param src_elem_type The dft src array element type.
     * @param dst_elem_type The dft dst array element type.
     */
    static Status CLPrecompile(Context *ctx, ElemType src_elem_type, ElemType dst_elem_type);
};

/**
 * @brief The matrix Inverse Discrete Fourier Transform (IDFT) operation class.
 *
 * The use of this class for IDFT operations is not recommended.
 * It is recommended to use the `IInverseDft` API, which internally calls this class.
 * The only recommended scenario for using this class is when the input or output type is 'CLMem'.
 *
 * The approximate internal call within the `IInverseDft` function is as follows:
 *
 * @code
 * InverseDft idft(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize().
 * return OpCall(ctx, idft, &src, &dst, with_scale);
 * @endcode
 */
class AURA_EXPORTS InverseDft : public Op
{
public:
    /**
     * @brief Constructor for the InverseDft class.
     *
     * @param ctx The pointer to the Context object
     * @param target The platform on which this function runs
     */
    InverseDft(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Set the arguments for the matrix IDFT operation.
     *
     * For more details, please refer to @ref inversedft_details
     *
     * @note If the type of src or dst is `CLMem` and is an iaura2D memory object, the row pitch
     * should be aligned to a stride, which is obtained through the `GetCLLengthAlignSize` function and is
     * platform-dependent.
     */
    Status SetArgs(const Array *src, Array *dst, DT_BOOL with_scale);

    /**
     * @brief Generate arithmetic opencl precompiled cache.
     *
     * @param src_elem_type The dft src array element type.
     * @param dst_elem_type The dft dst array element type.
     * @param with_scale Flag indicating whether scaling is applied in the IDFT computation.
     * @param is_dst_c1 Is the dft dst array 1 channel.
     */
    static Status CLPrecompile(Context *ctx, ElemType src_elem_type, ElemType dst_elem_type, DT_BOOL with_scale, DT_BOOL is_dst_c1);
};

/**
 * @brief Performs the Discrete Fourier Transform (DFT) operation.
 *
 * @anchor dft_details
 * The Discrete Fourier Transform (DFT) is a mathematical transformation that converts a finite
 * sequence of equally spaced samples of a function into a sequence of complex numbers. And in-place
 * operation is not supported.
 *
 * In iaura processing, two-dimensional DFT is often used, extending the concept to two-dimensional arrays.
 * The DFT mathematical expression is given by:
 *
 * @f$ F(u, v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x, y) \cdot e^{-j\frac{2\pi}{M}ux} \cdot e^{-j\frac{2\pi}{N}vy} @f$
 *
 * - F(u, v) is the complex value in the frequency domain.
 * - f(x, y) is the pixel intensity in the spatial domain.
 * - M and N are the width and height of the iaura, respectively.
 *
 * @param ctx The pointer to the Context object
 * @param src The source matrix for which DFT is performed.
 * @param dst The destination matrix to store the DFT result.
 * @param target The platform on which this function runs
 *
 * @return Status Status::OK if successful; otherwise, an appropriate error status.
 *
 * ### the supported data types and platforms
 * Platforms    | Src data type                                         | Dst data type
 * -------------|-------------------------------------------------------|------------------
 * NONE         | U8C1/S8C1/U16C1/S16C1/F16C1/U32C1/S32C1/F32C1         | F32C2
 * NEON         | U8C1/S8C1/U16C1/S16C1/F16C1/F32C1                     | F32C2
 * OpenCL       | U8C1/S8C1/U16C1/S16C1/F16C1/F32C1                     | F32C2
 *
 * @note For the OpenCL platform, the width and height of matrix(src or dst) must both be greater than 16 and must be power of 2.
 */
AURA_EXPORTS Status IDft(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target = OpTarget::Default());

/**
 * @brief Performs the Inverse Discrete Fourier Transform (IDFT) operation.
 *
 * @anchor inversedft_details
 * The Inverse Discrete Fourier Transform (IDFT) is a mathematical operation that transforms a signal
 * from its frequency domain representation back to its time domain representation. It is the inverse
 * operation of the Discrete Fourier Transform (DFT). And in-place operation is not supported.
 *
 * For a two-dimensional iaura, represented as a complex matrix in the frequency domain, the IDFT mathematical
 * expression is given by:
 *
 * @f$ f(x, y) = \frac{1}{MN} \sum_{u=0}^{M-1} \sum_{v=0}^{N-1} F(u, v) \cdot e^{j\frac{2\pi}{M}ux} \cdot e^{j\frac{2\pi}{N}vy} @f$
 *
 * - f(x, y) is the pixel intensity in the spatial domain (reconstructed iaura).
 * - F(u, v) is the complex value in the frequency domain.
 * - M and N are the width and height of the iaura, respectively.
 *
 * @param ctx The pointer to the Context object
 * @param src The source matrix for which IDFT is performed.
 * @param dst The destination matrix to store the IDFT result.
 * @param with_scale Flag indicating whether scaling is applied in the IDFT computation.
 * @param target The platform on which this function runs
 *
 * @return Status Status::OK if successful; otherwise, an appropriate error status.
 *
 * ### the supported data types and platforms
 * Platforms    | Src data type    | Dst data type
 * -------------|------------------|---------------------------------------------------------------
 * NONE         | F32C2            | U8C1/S8C1/U16C1/S16C1/F16C1/U32C1/S32C1/F32C1/F32C2
 * NEON         | F32C2            | U8C1/S8C1/U16C1/S16C1/F16C1/U32C1/S32C1/F32C1/F32C2
 * OpenCL       | F32C2            | U8C1/S8C1/U16C1/S16C1/F16C1/U32C1/S32C1/F32C1/F32C2
 *
 * @note For the OpenCL platform, the width and height of the src must both be greater than 16 and must be power of 2.
 */
AURA_EXPORTS Status IInverseDft(Context *ctx, const Mat &src, Mat &dst, DT_BOOL with_scale, const OpTarget &target = OpTarget::Default());

/**
 * @}
 */

}// namespace aura

#endif // AURA_OPS_MATRIX_DFT_HPP__
