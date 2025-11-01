#ifndef AURA_OPS_MATRIX_FLIP_HPP__
#define AURA_OPS_MATRIX_FLIP_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

/**
 * @defgroup ops Operators
 * @{
 *      @defgroup matrix Matrix Process
 *      @{
 *          @defgroup flip Flip
 *      @}
 * @}
 */

namespace aura
{

/**
 * @addtogroup flip
 * @{
 */

/**
 * @brief Enum class representing different types of matrix flips.
 */
enum class FlipType
{
    HORIZONTAL = 0, /*!< Flip horizontally. */
    VERTICAL,       /*!< Flip vertically. */
    BOTH,           /*!< Flip in both directions. */
};

/**
 * @brief Overloaded stream insertion operator for FlipType.
 *
 * This function converts a FlipType to its corresponding string representation
 * and inserts it into an output stream.
 *
 * @param os The output stream.
 * @param type The FlipType to be inserted.
 *
 * @return A reference to the output stream.
 */
AURA_INLINE std::ostream &operator<<(std::ostream &os, const FlipType &type)
{
    switch (type)
    {
        case FlipType::HORIZONTAL: {
            os << "Horizontal";
            break;
        }

        case FlipType::VERTICAL: {
            os << "Vertical";
            break;
        }

        case FlipType::BOTH: {
            os << "Both";
            break;
        }

        default: {
            os << "Invalid";
            break;
        }
    }

    return os;
}

/**
 * @brief Converts a FlipType to its string representation.
 *
 * @param type The FlipType to be converted.
 *
 *  @return The string representation of the FlipType.
 */
AURA_INLINE std::string FlipTypeToString(const FlipType &type)
{
    std::stringstream sstream;
    sstream << type;
    return sstream.str();
}

/**
 * @brief The matrix flipping operation class.
 *
 * The use of this class for flipping operations is not recommended.
 * It is recommended to use the `IFlip` API, which internally calls this class.
 * The only recommended scenario for using this class is when the input or output type is 'CLMem'.
 *
 * The approximate internal call within the `IFlip` function is as follows:
 *
 * @code
 * Flip flip(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize().
 * return OpCall(ctx, flip, &src, &dst, type);
 * @endcode
 */
class AURA_EXPORTS Flip : public Op
{
public:
    /**
     * @brief Constructor for the Flip class.
     *
     * @param ctx The pointer to the Context object
     * @param target The platform on which this function runs
     */
    Flip(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Set the arguments for the matrix flipping operation.
     *
     * For more details, please refer to @ref filp_details
     */
    Status SetArgs(const Array *src, Array *dst, FlipType type);
};

/**
 * @brief Performs a matrix flipping operation.
 *
 * @anchor filp_details
 * This function flips the elements of the src matrix based on the specified flip type
 * and stores the result in the dst matrix. And in-place operation is not supported.
 *
 * @param ctx The pointer to the Context object
 * @param src The source matrix to be flipped.
 * @param dst The destination matrix to store the flipped iaura.
 * @param type The type of flip operation to be performed (horizontal, vertical, or both).
 * @param target The platform on which this function runs
 *
 * @return Status Status::OK if successful; otherwise, an appropriate error status.
 *
 * ### the supported data types and platforms
 * Platforms    | Data type(src or dst)                                   | FlipType
 * -------------|---------------------------------------------------------|-------------------------------
 * NONE         | U8Cx/S8Cx/U16Cx/S16Cx/F16Cx/U32Cx/S32Cx/F32Cx, x = N    | VERTICAL/HORIZONTAL/BOTH
 *
 * @note N is positive integer.
 */
AURA_EXPORTS Status IFlip(Context *ctx, const Mat &src, Mat &dst, FlipType type, const OpTarget &target = OpTarget::Default());

/**
 * @}
 */

}// namespace aura

#endif // AURA_OPS_MATRIX_FLIP_HPP__
