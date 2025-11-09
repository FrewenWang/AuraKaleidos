#ifndef AURA_OPS_MATRIX_ROTATE_HPP__
#define AURA_OPS_MATRIX_ROTATE_HPP__

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
 * @brief Enum class representing different types of rotations.
 */
enum class RotateType
{
    ROTATE_90    = 0, /*!< Rotate 90  degrees clockwise */
    ROTATE_180,       /*!< Rotate 180 degrees clockwise */
    ROTATE_270,       /*!< Rotate 270 degrees clockwise */
};

/**
 * @brief Overloaded stream insertion operator for RotateType.
 *
 * This function converts a RotateType to its corresponding string representation
 * and inserts it into an output stream.
 *
 * @param os The output stream.
 * @param type The RotateType to be inserted.
 *
 * @return A reference to the output stream.
 */
AURA_INLINE std::ostream& operator<<(std::ostream &os, const RotateType &type)
{
    switch (type)
    {
        case RotateType::ROTATE_90:
        {
            os << "ROTATE_90";
            break;
        }
        case RotateType::ROTATE_180:
        {
            os << "ROTATE_180";
            break;
        }
        case RotateType::ROTATE_270:
        {
            os << "ROTATE_270";
            break;
        }
        default:
        {
            os << "Invalid";
            break;
        }
    }
    return os;
}

/**
 * @brief Converts a RotateType to its string representation.
 *
 * @param type The RotateType to be converted.
 *
 * @return The string representation of the RotateType.
 */
AURA_INLINE std::string RotateTypeToString(const RotateType &type)
{
    std::stringstream sstream;
    sstream << type;
    return sstream.str();
}

/**
 * @brief The matrix rotation operation class.
 *
 * The use of this class for the above-mentioned operation is not recommended.
 * It is recommended to use the `IRotate` API, which internally calls this class.
 * The only recommended scenario for using this class is when the input or output type is 'CLMem'.
 *
 * The approximate internal call within the `IRotate` function is as follows:
 *
 * @code
 * Rotate rotate(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize().
 * return OpCall(ctx, rotate, &src, &dst, type);
 * @endcode
 */
class AURA_EXPORTS Rotate : public Op
{
public:
    /**
     * @brief Constructor for the Rotate class.
     *
     * @param ctx The pointer to the Context object
     * @param target The platform on which this function runs
     */
    Rotate(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Sets the arguments for matrix rotation operations.
     *
     * For more details, please refer to @ref rotate_details
     *
     * @note If the type of src or dst is `CLMem` and is an iaura2D memory object, the row pitch
     * should be aligned to a stride, which is obtained through the `GetCLLengthAlignSize` function and is
     * platform-dependent.
     */
    Status SetArgs(const Array *src, Array *dst, RotateType type);

    /**
     * @brief Generate ratate opencl precompiled cache.
     *
     * @param elem_type The ratate src/dst array element type.
     * @param ochannel The ratate dst array channel.
     * @param type It represents the rotation angle for the matrix.
     */
    static Status CLPrecompile(Context *ctx, ElemType elem_type, DT_S32 ochannel, RotateType type);
};

/**
 * @brief Performs a matrix rotation operation.
 *
 * @anchor rotate_details
 * This function rotates the src matrix using the specified rotation type (type) and stores the rotated result
 * in the dst matrix based on the specified target. And in-place operation is not supported.
 *
 * @param ctx The pointer to the Context object
 * @param src The source matrix to be rotated.
 * @param dst The destination matrix to store the rotated result.
 * @param type It represents the rotation angle for the matrix.
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
 * @note 1.The above implementations support RotateType(ROTATE_90/ROTATE_180/ROTATE_270). <br>
 *       2.Src and dst must have same size and data type.
 */
AURA_EXPORTS Status IRotate(Context *ctx, const Mat &src, Mat &dst, RotateType type, const OpTarget &target = OpTarget::Default());

/**
 * @}
 */

}// namespace aura

#endif // AURA_OPS_MATRIX_ROTATE_HPP__
