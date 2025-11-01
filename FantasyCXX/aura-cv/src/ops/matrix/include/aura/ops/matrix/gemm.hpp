#ifndef AURA_OPS_MATRIX_GEMM_HPP__
#define AURA_OPS_MATRIX_GEMM_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

/**
 * @defgroup ops Operators
 * @{
 *      @defgroup matrix Matrix Process
 *      @{
 *          @defgroup gemm GEMM
 *      @}
 * @}
 */

namespace aura
{

/**
 * @addtogroup gemm
 * @{
 */

/**
 * @brief The General Matrix Multiply (GEMM) operation class.
 *
 * The use of this class for GEMM operations is not recommended.
 * It is recommended to use the `IGemm` API, which internally calls this class.
 * The only recommended scenario for using this class is when the input or output type is 'CLMem'.
 *
 * The approximate internal call within the `IGemm` function is as follows:
 *
 * @code
 * Gemm gemm(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize().
 * return OpCall(ctx, gemm, &src0, &src1, &dst);
 * @endcode
 */
class AURA_EXPORTS Gemm : public Op
{
public:
    /**
     * @brief Constructor for the Gemm class.
     *
     * @param ctx The pointer to the Context object
     * @param target The platform on which this function runs
     */
    Gemm(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Set the arguments for the GEMM operation.
     *
     * For more details, please refer to @ref gemm_details.
     *
     * @note If the type of src or dst is `CLMem` and is an iaura2D memory object, the row pitch
     * should be aligned to a stride, which is obtained through the `GetCLLengthAlignSize` function and is
     * platform-dependent.
     */
    Status SetArgs(const Array *src0, const Array *src1, Array *dst);

    /**
     * @brief Generate gemm opencl precompiled cache.
     *
     */
    static Status CLPrecompile(Context *ctx);
};

/**
 * @brief Performs a General Matrix Multiply (GEMM) operation on the source matrices.
 *
 * @anchor gemm_details
 * GEMM is a fundamental operation in linear algebra used for matrix multiplication. In the context of
 * computer science and numerical computing, GEMM typically refers to the multiplication of two matrices
 * to produce a third matrix. And in-place operation is not supported. It is expressed as:
 *
 * @f$ dst_{mn} = \sum_{k=1}^{k} src0_{mk} \cdot src1_{kn} @f$
 *
 * @param ctx The pointer to the Context object
 * @param src0 First input matrix(m x k, m rows, k columns).
 * @param src1 Second input matrix(k x n, k rows, n columns).
 * @param dst Output matrix(m x n, m rows, n columns).
 * @param target The platform on which this function runs
 *
 * @return Status Status::OK if successful; otherwise, an appropriate error status.
 *
 * ### the supported data types and platforms
 * Platforms    | Data type(src or dst)
 * -------------|----------------------
 * NONE         | F32C1
 * NEON         | F32C1
 * OpenCL       | F32C1
 */
AURA_EXPORTS Status IGemm(Context *ctx, const Mat &src0, const Mat &src1, Mat &dst, const OpTarget &target = OpTarget::Default());

/**
 * @}
 */

}// namespace aura

#endif // AURA_OPS_MATRIX_GEMM_HPP__
