#ifndef AURA_OPS_MISC_THRESHOLD_HPP__
#define AURA_OPS_MISC_THRESHOLD_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

/**
 * @defgroup ops Operators
 * @{
 *    @defgroup misc Miscellaneous Iaura Process
 * @}
 */

/**
 * @addtogroup misc
 * @{
 */

#define AURA_THRESH_BINARY                   (0)  /*!< value = value > threshold ? max_value : 0 */
#define AURA_THRESH_BINARY_INV               (1)  /*!< value = value > threshold ? 0 : max_value */
#define AURA_THRESH_TRUNC                    (2)  /*!< value = value > threshold ? threshold : max_value */
#define AURA_THRESH_TOZERO                   (3)  /*!< value = value > threshold ? max_value : 0 */
#define AURA_THRESH_TOZERO_INV               (4)  /*!< value = value > threshold ? 0 : max_value */
#define AURA_THRESH_MASK_LOW                 (7)
#define AURA_THRESH_OTSU                     (8)  /*!< use Otsu algorithm to choose the optimal threshold value; */
#define AURA_THRESH_TRIANGLE                 (16) /*!< use Triangle algorithm to choose the optimal threshold value; */
#define AURA_THRESH_MASK_HIGH                (24)

/**
 * @}
 */

namespace aura
{
/**
 * @addtogroup misc
 * @{
 */

AURA_INLINE const std::string ThresholdTypeToString(MI_S32 type)
{
    std::ostringstream ss;

    switch (type & AURA_THRESH_MASK_HIGH)
    {
        case AURA_THRESH_OTSU:
        {
            ss << "THRESH_OTSU | ";
            break;
        }

        case AURA_THRESH_TRIANGLE:
        {
            ss << "THRESH_TRIANGLE | ";
            break;
        }

        default:
        {
            break;
        }
    }

    switch (type & AURA_THRESH_MASK_LOW)
    {
        case AURA_THRESH_BINARY:
        {
            ss << "THRESH_BINARY";
            break;
        }

        case AURA_THRESH_BINARY_INV:
        {
            ss << "THRESH_BINARY_INV";
            break;
        }

        case AURA_THRESH_TRUNC:
        {
            ss << "THRESH_TRUNC";
            break;
        }

        case AURA_THRESH_TOZERO:
        {
            ss << "THRESH_TOZERO";
            break;
        }

        case AURA_THRESH_TOZERO_INV:
        {
            ss << "THRESH_TOZERO_INV";
            break;
        }

        default:
        {
            ss << "undefined type";
            break;
        }
    }

    return ss.str();
}

/**
 * @brief Class representing the Threshold operation.
 *
 * This class represents a thresholding operation, which is a type of iaura processing that convertspixel values
 * in an input iaura to specified values based on a threshold. It is recommended to use the `IThreshold` API,
 * which internally calls this class. The only recommended scenario for using this class is when the input or
 * output type is `CLMem`.
 *
 * The approximate internal call within the `IThreshold` function is as follows:
 *
 * @code
 * Threshold threshold(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize(),
 * return OpCall(ctx, threshold, &src, &dst, thresh, max_val, type); 
 * @endcode
 */
class AURA_EXPORTS Threshold : public Op
{
public:
    /**
     * @brief Constructor for Threshold class.
     *
     * @param ctx The pointer to the Context object.
     * @param target The platform on which this function runs.
     */
    Threshold(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Set arguments for the Threshold operation.
     *
     * For more details, please refer to @ref threshold_details
     */
    Status SetArgs(const Array *src, Array *dst, MI_F32 thresh, MI_F32 max_val, MI_S32 type);
};

/**
 * @brief Apply a Threshold operation to src matrix.
 *
 * @anchor threshold_details
 * This function applies a Threshold operation to src matrix. And in-place operation is not supported.
 *
 * @param ctx The pointer to the Context object.
 * @param src The source matrix.
 * @param dst The destination matrix.
 * @param thresh The threshold value.
 * @param max_val The maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV types.
 * @param type The thresholding type (THRESH_BINARY, THRESH_BINARY_INV, etc.).
 * @param target The platform on which this function runs.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 *
 * ### The Support data types and platforms
 * Platforms | Data type(src or dst)
 * ----------|------------------------------------------
 * NONE      | U8Cx/S8Cx/U16Cx/S16Cx/F16Cx/F32Cx, x = N
 * NEON      | U8Cx/S8Cx/U16Cx/S16Cx/F16Cx/F32Cx, x = N
 * HVX       | U8Cx/U16Cx/S16Cx, x = 1, 2, 3
 *
 * @note N is positive integer.
 */
AURA_EXPORTS Status IThreshold(Context *ctx, const Mat &src, Mat &dst,
                               MI_F32 thresh, MI_F32 max_val, MI_S32 type,
                               const OpTarget &target = OpTarget::Default());

/**
 * @}
 */
} // namespace aura

#endif // AURA_OPS_MISC_THRESHOLD_HPP__