#ifndef AURA_OPS_MATRIX_BINARY_HPP__
#define AURA_OPS_MATRIX_BINARY_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

/**
 * @defgroup ops Operators
 * @{
 *      @defgroup matrix Matrix Process
 *      @{
 *          @defgroup binary Binary
 *      @}
 * @}
 */

namespace aura
{

/**
 * @addtogroup binary
 * @{
 */

/**
 * @brief Enum class representing different binary operation types.
 */
enum class BinaryOpType
{
    MIN = 0, /*!< Represents the minimum operation */
    MAX,     /*!< Represents the maximum operation */
};

/**
 * @brief Overloaded stream insertion operator for BinaryOpType.
 *
 * This function converts a BinaryOpType to its corresponding string representation
 * and inserts it into an output stream.
 *
 * @param os The output stream.
 * @param binary_type The BinaryOpType to be inserted.
 *
 * @return A reference to the output stream.
 */
AURA_INLINE std::ostream& operator<<(std::ostream &os, BinaryOpType binary_type)
{
    switch (binary_type)
    {
        case BinaryOpType::MAX:
        {
            os << "MAX";
            break;
        }

        case BinaryOpType::MIN:
        {
            os << "MIN";
            break;
        }

        default:
        {
            os << "INVALID";
            break;
        }
    }

    return os;
}

/**
 * @brief Converts a BinaryOpType to its string representation.
 *
 * @param binary_type The BinaryOpType to be converted.
 *
 * @return The string representation of the BinaryOpType.
 */
AURA_INLINE const std::string BinaryOpTypeToString(BinaryOpType binary_type)
{
    std::ostringstream oss;
    oss << binary_type;
    return oss.str();
}

/**
 * @brief The matrix binary operation class.
 *
 * The use of this class for performing binary operations is not recommended.
 * It is recommended to use the `IMax/IMin` API, which internally call this class.
 * The only recommended scenario for using this class is when the input or output type is 'CLMem'.
 *
 * The approximate internal call within the `IMax` function is as follows:
 *
 * @code
 * Binary binary(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize().
 * return OpCall(ctx, binary, &src0, &src1, &dst, BinaryOpType::MAX);
 * @endcode
 */
class AURA_EXPORTS Binary : public Op
{
public:
    /**
     * @brief Constructor for the Binary class.
     *
     * @param ctx ctx The pointer to the Context object
     * @param target The platform on which this function runs
     */
    Binary(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Set the arguments for the matrix binary operation.
     *
     * For more details, please refer to @ref binary_imax_details or @ref binary_imin_details.
     *
     * @note If the type of src or dst is `CLMem` and is an iaura2D memory object, the row pitch
     * should be aligned to a stride, which is obtained through the `GetCLLengthAlignSize` function and is
     * platform-dependent.
     */
    Status SetArgs(const Array *src0, const Array *src1, Array *dst, BinaryOpType type);

    /**
     * @brief Generate binary opencl precompiled cache.
     *
     * @param elem_type The binary src/dst array element type.
     * @param op_type The binary opType to be converted.
     */
    static Status CLPrecompile(Context *ctx, ElemType elem_type, BinaryOpType op_type);
};

/**
 * @brief Performs element-wise maximum operation between two matrices.
 *
 * @anchor binary_imax_details
 * This function computes the element-wise maximum between src0 and src1 matrices and stores
 * the result in dst matrix. And in-place operation is supported.
 *
 * @param ctx ctx The pointer to the Context object
 * @param src0 The first source matrix.
 * @param src1 The second source matrix.
 * @param dst The destination matrix to store the element-wise maximum result.
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
AURA_EXPORTS Status IMax(Context *ctx, const Mat &src0, const Mat &src1, Mat &dst, const OpTarget &target = OpTarget::Default());


/**
 * @brief Performs element-wise minimum operation between two matrices.
 *
 * @anchor binary_imin_details
 * This function computes the element-wise minimum between src0 and src1 matrices and stores
 * the result in dst matrix. And in-place operation is supported.
 *
 * @param ctx ctx The pointer to the Context object
 * @param src0 The first source matrix.
 * @param src1 The second source matrix.
 * @param dst The destination matrix to store the element-wise minimum result.
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
AURA_EXPORTS Status IMin(Context *ctx, const Mat &src0, const Mat &src1, Mat &dst, const OpTarget &target = OpTarget::Default());

/**
 * @}
 */

}// namespace aura

#endif// AURA_OPS_MATRIX_BINARY_HPP__
