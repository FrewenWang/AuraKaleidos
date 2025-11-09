#ifndef AURA_OPS_MATRIX_NORM_HPP__
#define AURA_OPS_MATRIX_NORM_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

/**
 * @defgroup ops Operators
 * @{
 *      @defgroup matrix Matrix Process
 *      @{
 *          @defgroup norm Norm
 *      @}
 * @}
 */

namespace aura
{

/**
 * @addtogroup norm
 * @{
 */

/**
 * @brief Enum class representing different types of norms.
 */
enum class NormType
{
    NORM_INF    = 0, /*!< Compute L_inf  Norm */
    NORM_L1,         /*!< Compute L_1    Norm */
    NORM_L2,         /*!< Compute L_2    Norm */
    NORM_L2SQR,      /*!< Compute L_2SQR Norm */
    NORM_MINMAX,     /*!< Used for Normalize  */
};

/**
 * @brief Overloaded stream insertion operator for NormType.
 *
 * This function converts a NormType to its corresponding string representation
 * and inserts it into an output stream.
 *
 * @param os The output stream.
 * @param type The NormType to be inserted.
 *
 * @return A reference to the output stream.
 */
AURA_INLINE std::ostream& operator<<(std::ostream &os, NormType type)
{
    switch(type)
    {
        case NormType::NORM_L1:
        {
            os << "NORM_L1";
            break;
        }
        case NormType::NORM_L2:
        {
            os << "NORM_L2";
            break;
        }
        case NormType::NORM_INF:
        {
            os << "NORM_INF";
            break;
        }
        case NormType::NORM_L2SQR:
        {
            os << "NORM_L2SQR";
            break;
        }
        case NormType::NORM_MINMAX:
        {
            os << "NORM_MINMAX";
            break;
        }
        default:
        {
            os << "undefined norm type";
            break;
        }
    }

    return os;
}

/**
 * @brief Converts a NormType to its string representation.
 *
 * @param type The NormType to be converted.
 *
 * @return The string representation of the NormType.
 */
AURA_INLINE const std::string NormTypeToString(NormType type)
{
    std::ostringstream ss;
    ss << type;
    return ss.str();
}

/**
 * @brief Interface class representing an operation to compute different types of norms on a matrix.
 *
 * The use of this class for the above-mentioned operation is not recommended.
 * It is recommended to use the `INorm` API, which internally calls this class.
 * The only recommended scenario for using this class is when the input or output type is 'CLMem'.
 *
 * The approximate internal call within the `INorm` function is as follows:
 *
 * @code
 * Norm norm(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize().
 * return OpCall(ctx, norm, &src, result, type);
 * @endcode
 */
class AURA_EXPORTS Norm : public Op
{
public:
    /**
     * @brief Constructor for the Norm class.
     *
     * @param ctx The pointer to the Context object
     * @param target The platform on which this function runs
     */
    Norm(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Sets the arguments for computing a specific type of norm on the array.
     *
     * For more details, please refer to @ref norm_details
     *
     * @note If the type of src or dst is `CLMem` and is an iaura2D memory object, the row pitch
     * should be aligned to a stride, which is obtained through the `GetCLLengthAlignSize` function and is
     * platform-dependent.
     */
    Status SetArgs(const Array *src, DT_F64 *result, NormType type);

    /**
     * @brief Generate norm opencl precompiled cache.
     *
     * @param src_elem_type The norm src array element type.
     * @param dst_elem_type The norm dst array element type.
     * @param type The type of norm to be computed.
     */
    static Status CLPrecompile(Context *ctx, ElemType src_elem_type, ElemType dst_elem_type, NormType type);
};

/**
 * @brief Computes a specific type of norm on the source matrix.
 *
 * @anchor norm_details
 * This function computes a specific type of norm (specified by type) on the src matrix
 * and stores the result in the provided variable (result).
 *
 * @param ctx The pointer to the Context object
 * @param src The source matrix for which the norm is computed.
 * @param result Pointer to store the computed norm result.
 * @param type The type of norm to be computed.
 * @param target The platform on which this function runs
 *
 * @return Status Status::OK if successful; otherwise, an appropriate error status.
 *
 * ### the supported data types and platforms
 * Platforms    | Src data type
 * -------------|----------------------------------------------------------------
 * NONE         | U8Cx/S8Cx/U16Cx/S16Cx/F16Cx/U32Cx/S32Cx/F32Cx, x = N
 * NEON         | U8Cx/S8Cx/U16Cx/S16Cx/F16Cx/F32Cx, x = 1, 2, 3
 * OpenCL       | U8Cx/S8Cx/U16Cx/S16Cx/F16Cx/U32Cx/S32Cx/F32Cx, x = N
 *
 * @note 1.N is positive integer. <br>
 *       2.The above implementations supported NormType(NORM_INF/NORM_L1/NORM_L2/NORM_L2SQR).
 */
AURA_EXPORTS Status INorm(Context *ctx, const Mat &src, DT_F64 *result, NormType type, const OpTarget &target = OpTarget::Default());

/**
 * @}
 */

}// namespace aura

#endif // AURA_OPS_MATRIX_NORM_HPP__
