#ifndef AURA_OPS_MATRIX_ARITHMETIC_HPP__
#define AURA_OPS_MATRIX_ARITHMETIC_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

/**
 * @defgroup ops Operators
 * @{
 *      @defgroup matrix Matrix Process
 *      @{
 *          @defgroup arithmetic Arithmetic
 *      @}
 * @}
 */

namespace aura
{

/**
 * @addtogroup arithmetic
 * @{
 */

/**
 * @brief Enum class representing different arithmetic operation types.
 */
enum class ArithmOpType
{
    ADD = 0, /*!< Represents addition operation */
    SUB,     /*!< Represents subtraction operation */
    MUL,     /*!< Represents multiplication operation */
    DIV,     /*!< Represents division operation */
};

/**
 * @brief Overloaded output stream operator for ArithmOpType enumeration.
 *
 * This operator allows printing ArithmOpType enumerators to an output stream.
 *
 * @param os The output stream.
 * @param type The ArithmOpType enumerator to be printed.
 *
 * @return Color convention out stream.
 */
AURA_INLINE std::ostream& operator<<(std::ostream &os, ArithmOpType arithm_type)
{
    switch (arithm_type)
    {
        case ArithmOpType::ADD:
        {
            os << "ADD";
            break;
        }

        case ArithmOpType::SUB:
        {
            os << "SUB";
            break;
        }

        case ArithmOpType::MUL:
        {
            os << "MUL";
            break;
        }

        case ArithmOpType::DIV:
        {
            os << "DIV";
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
 * @brief Convert ArithmOpType to string representation.
 *
 * This function converts a ArithmOpType enumerator to its string representation.
 *
 * @param type The ArithmOpType enumerator.
 *
 * @return The string representation of the ArithmOpType.
 */
AURA_INLINE const std::string ArithmeOpTypeToString(ArithmOpType arithm_type)
{
    std::ostringstream oss;
    oss << arithm_type;
    return oss.str();
}

/**
 * @brief The matrix arithmetic operation class.
 *
 * The use of this class for matrix arithmetic is not recommended.
 * It is recommended to use the `IAdd\ISubtract\IMultiply\IDivide` API, which internally call this class.
 * The only recommended scenario for using this class is when the input or output type is 'CLMem'.
 *
 * The approximate internal call within the `IAdd` functions are as follows:
 *
 * @code
 * Arithmetic arithmetic(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize().
 * return OpCall(ctx, arithmetic, &src0, &src1, &dst, ArithmOpType::ADD);
 * @endcode
 */
class AURA_EXPORTS Arithmetic : public Op
{
public:
    /**
     * @brief Constructor for the Arithmetic class.
     *
     * @param ctx The pointer to the Context object
     * @param target The platform on which this function runs
     */
    Arithmetic(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Set the arguments for the matrix arithmetic operation.
     *
     * @note If the type of src or dst is `CLMem` and is an iaura2D memory object, the row pitch
     * should be aligned to a stride, which is obtained through the `GetCLLengthAlignSize` function and is
     * platform-dependent.
     */
    Status SetArgs(const Array *src0, const Array *src1, Array *dst, ArithmOpType op);

    /**
     * @brief Generate arithmetic opencl precompiled cache.
     *
     * @param src_elem_type The arithmetic src array element type.
     * @param dst_elem_type The arithmetic dst array element type.
     * @param op_type The arithmetic operation types.
     */
    static Status CLPrecompile(Context *ctx, ElemType src_elem_type, ElemType dst_elem_type, ArithmOpType op_type);
};

/**
 * @brief Interface class for dividing scalar by matrix element by element.
 *
 * The use of this class for scalar divided by matrix is not recommended.
 * It is recommended to use the `IDivide` API, which internally calls this class.
 * The only recommended scenario for using this class is when the input or output type is 'CLMem'.
 *
 * The approximate internal call within the `IDivide` functions are as follows:
 *
 * @code
 * ScalarDivide scalar_divide(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize().
 * return OpCall(ctx, scalar_divide, scalar, &src, &dst);
 * @endcode
 */
class AURA_EXPORTS ScalarDivide : public Op
{
public:
    /**
     * @brief Constructor for the ScalarDivide class.
     *
     * @param ctx The pointer to the Context object
     *
     * @param target The platform on which this function runs
     */
    ScalarDivide(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Set the arguments for the matrix scalar division operation.
     *
     * For more details, please refer to @ref scalar_divide_details
     */
    Status SetArgs(DT_F32 scalar, const Array *src, Array *dst);
};

/**
 * @brief Performs element-wise addition of two matrices.
 *
 * This function adds the elements of src0 and src1 matrices and stores the result in dst matrix.
 * And in-place operation is supported.
 *
 * @param ctx The pointer to the Context object
 * @param src0 The first source matrix.
 * @param src1 The second source matrix.
 * @param dst The destination matrix to store the result.
 * @param target The platform on which this function runs
 *
 * @return Status Status::OK if successful; otherwise, an appropriate error status.
 *
 * ### the supported data types and platforms
 * Platforms    |             Format             |       Data type
 * -------------|--------------------------------|-------------------------------
 * NONE         | src or dst, can be different   | U8Cx/S8Cx/U16Cx/S16Cx/F16Cx/U32Cx/S32Cx/F32Cx, x = N
 * NEON         | src -> dst                     | (U8->U8/U16)Cx,(S8->S8/S16)Cx,(U16->U16/U32)Cx,(S16->S16/S32)Cx,(U32->U32)Cx,(S32->S32)Cx,(F16->F16)Cx,(F32->F32)Cx, x = N
 * OpenCL       | src or dst, can be different   | U8Cx/S8Cx/U16Cx/S16Cx/F16Cx/U32Cx/S32Cx/F32Cx, x = N
 * HVX          | src -> dst                     | (U8->U8/U16)Cx,(S8->S8/S16)Cx,(U16->U16/U32)Cx,(S16->S16/S32)Cx,(U32->U32)Cx,(S32->S32)Cx, x = N
 *
 * @note N is positive integer.
 */
AURA_EXPORTS Status IAdd(Context *ctx, const Mat &src0, const Mat &src1, Mat &dst, const OpTarget &target = OpTarget::Default());

/**
 * @brief Performs element-wise subtraction of two matrices.
 *
 * This function subtracts the elements of src1 from src0 matrix and stores the result in dst matrix.
 * And in-place operation is supported.
 *
 * @param ctx The pointer to the Context object
 * @param src0 The first source matrix.
 * @param src1 The second source matrix.
 * @param dst The destination matrix to store the result.
 * @param target The platform on which this function runs
 *
 * @return Status Status::OK if successful; otherwise, an appropriate error status.
 *
 * ### the supported data types and platforms
 * Platforms    |             Format             |          Data type
 * -------------|--------------------------------|---------------------------------------------------------------
 * NONE         | src or dst, can be different   | U8Cx/S8Cx/U16Cx/S16Cx/F16Cx/U32Cx/S32Cx/F32Cx, x = N
 * NEON         | src -> dst                     | (U8->U8/S16)Cx,(S8->S8/S16)Cx,(U16->U16/S32)Cx,(S16->S16/S32)Cx,(U32->U32)Cx,(S32->S32)Cx,(F16->F16)Cx,(F32->F32)Cx, x = N
 * OpenCL       | src or dst, can be different   | U8Cx/S8Cx/U16Cx/S16Cx/F16Cx/U32Cx/S32Cx/F32Cx, x = N
 * HVX          | src -> dst                     | (U8->U8/S16)Cx,(S8->S8/S16)Cx,(U16->U16/S32)Cx,(S16->S16/S32)Cx,(U32->U32)Cx,(S32->S32)Cx, x = N
 *
 * @note N is positive integer.
 */
AURA_EXPORTS Status ISubtract(Context *ctx, const Mat &src0, const Mat &src1, Mat &dst, const OpTarget &target = OpTarget::Default());

/**
 * @brief Performs element-wise multiplication of two matrices.
 *
 * This function multiplies the elements of src0 and src1 matrices and stores the result in dst matrix.
 * And in-place operation is supported.
 *
 * @param ctx The pointer to the Context object
 * @param src0 The first source matrix.
 * @param src1 The second source matrix.
 * @param dst The destination matrix to store the result.
 * @param target The platform on which this function runs
 *
 * @return Status Status::OK if successful; otherwise, an appropriate error status.
 *
 * ### the supported data types and platforms
 * Platforms    |             Format             |            Data type
 * -------------|--------------------------------|---------------------------------------------------------------
 * NONE         | src or dst, can be different   | U8Cx/S8Cx/U16Cx/S16Cx/F16Cx/U32Cx/S32Cx/F32Cx, x = N
 * NEON         | src -> dst                     | (U8->U8/U16)Cx,(S8->S8/S16)Cx,(U16->U16/U32)Cx,(S16->S16/S32)Cx,(U32->U32)Cx,(S32->S32)Cx,(F16->F16)Cx,(F32->F32)Cx, x = N
 * OpenCL       | src or dst, can be different   | U8Cx/S8Cx/U16Cx/S16Cx/F16Cx/U32Cx/S32Cx/F32Cx, x = N
 * HVX          | src -> dst                     | (U8->U16)Cx,(S8->S16)Cx,(U16->U32)Cx,(S16->S32)Cx, x = N
 *
 * @note N is positive integer.
 */
AURA_EXPORTS Status IMultiply(Context *ctx, const Mat &src0, const Mat &src1, Mat &dst, const OpTarget &target = OpTarget::Default());

/**
 * @brief Performs element-wise division of two matrices.
 *
 * This function divides the elements of src0 matrix by elements of src1 matrix and stores the result
 * in dst matrix. And in-place operation is supported.
 *
 * @param ctx The pointer to the Context object
 * @param src0 The first source matrix.
 * @param src1 The second source matrix.
 * @param dst The destination matrix to store the result.
 * @param target The platform on which this function runs
 *
 * @return Status Status::OK if successful; otherwise, an appropriate error status.
 *
 * ### the supported data types and platforms
 * Platforms    |             Format             | Data type(src or dst)
 * -------------|--------------------------------|---------------------------------------------------------------
 * NONE         | src or dst, can be different   | U8Cx/S8Cx/U16Cx/S16Cx/F16Cx/U32Cx/S32Cx/F32Cx, x = N
 * NEON         | src -> dst                     | (F16->F16)Cx, (F32->F32)Cx, x = N
 * OpenCL       | src or dst, can be different   | U8Cx/S8Cx/U16Cx/S16Cx/F16Cx/U32Cx/S32Cx/F32Cx, x = N
 *
 * @note N is positive integer.
 */
AURA_EXPORTS Status IDivide(Context *ctx, const Mat &src0, const Mat &src1, Mat &dst, const OpTarget &target = OpTarget::Default());

/**
 * @brief Performs element-wise addition of a scalar value to a matrix.
 *
 * This function adds the scalar value to each element of the source matrix and stores the result in the
 * destination matrix. And in-place operation is supported.
 *
 * @param ctx The pointer to the Context object
 * @param src The source matrix.
 * @param scalar The scalar value to be added to the elements of the source matrix.
 * @param dst The destination matrix to store the result.
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
AURA_EXPORTS Status IAdd(Context *ctx, const Mat &src, DT_F32 scalar, Mat &dst, const OpTarget &target = OpTarget::Default());

/**
 * @brief Performs element-wise subtraction of a scalar value from a matrix.
 *
 * This function subtracts the scalar value from each element of the source matrix and stores the result
 * in the destination matrix. And in-place operation is supported.
 *
 * @param ctx The pointer to the Context object
 * @param src The source matrix.
 * @param scalar The scalar value to be subtracted from the elements of the source matrix.
 * @param dst The destination matrix to store the result.
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
AURA_EXPORTS Status ISubtract(Context *ctx, const Mat &src, DT_F32 scalar, Mat &dst, const OpTarget &target = OpTarget::Default());

/**
 * @brief Performs element-wise multiplication of a matrix by a scalar value.
 *
 * This function multiplies each element of the source matrix by the scalar value and stores the result
 * in the destination matrix. And in-place operation is supported.
 *
 * @param ctx The pointer to the Context object
 * @param src The source matrix.
 * @param scalar The scalar value to multiply with the elements of the source matrix.
 * @param dst The destination matrix to store the result.
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
AURA_EXPORTS Status IMultiply(Context *ctx, const Mat &src, DT_F32 scalar, Mat &dst, const OpTarget &target = OpTarget::Default());

/**
 * @brief Performs element-wise division of a matrix by a scalar value.
 *
 * This function divides each element of the source matrix by the scalar value and stores the result
 * in the destination matrix. And in-place operation is supported.
 *
 * @param ctx The pointer to the Context object
 * @param src The source matrix.
 * @param scalar The scalar value to divide the elements of the source matrix by.
 * @param dst The destination matrix to store the result.
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
AURA_EXPORTS Status IDivide(Context *ctx, const Mat &src, DT_F32 scalar, Mat &dst, const OpTarget &target = OpTarget::Default());

/**
 * @brief Performs element-wise subtraction of a matrix from a scalar value.
 *
 * This function subtracts each element of the source matrix from the scalar value and stores the result
 * in the destination matrix. And in-place operation is supported.
 *
 * @param ctx The pointer to the Context object
 * @param scalar The scalar value from which the elements of the source matrix will be subtracted.
 * @param src The source matrix.
 * @param dst The destination matrix to store the result.
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
AURA_EXPORTS Status ISubtract(Context *ctx, DT_F32 scalar, const Mat &src, Mat &dst, const OpTarget &target = OpTarget::Default());

/**
 * @brief Performs element-wise division of a scalar value by a matrix.
 *
 * @anchor scalar_divide_details
 * This function divides the scalar value by each element of the source matrix and stores the result
 * in the destination matrix. And in-place operation is supported.
 *
 * @param ctx The pointer to the Context object
 * @param scalar The scalar value to be divided by the elements of the source matrix.
 * @param src The source matrix.
 * @param dst The destination matrix to store the result.
 * @param target The platform on which this function runs
 *
 * @return Status Status::OK if successful; otherwise, an appropriate error status.
 *
 * ### the supported data types and platforms
 * Platforms    | Data type(src or dst)
 * -------------|---------------------------------------------------------
 * NONE         | U8Cx/S8Cx/U16Cx/S16Cx/U32Cx/S32Cx/F32Cx, x = N
 * NEON         | F32Cx, x = N
 *
 * @note N is positive integer.
 */
AURA_EXPORTS Status IDivide(Context *ctx, DT_F32 scalar, const Mat &src, Mat &dst, const OpTarget &target = OpTarget::Default());

/**
 * @}
 */

}// namespace aura

#endif // AURA_OPS_MATRIX_ARITHMETIC_HPP__
