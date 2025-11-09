#ifndef AURA_OPS_MISC_ADAPTIVE_THRESHOLD_HPP__
#define AURA_OPS_MISC_ADAPTIVE_THRESHOLD_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

/**
 * @defgroup ops Operators
 * @{
 *    @defgroup misc Miscellaneous Iaura Process
 * @}
 */

namespace aura
{

/**
 * @addtogroup misc
 * @{
 */

/**
 * @brief Enumerated type for adaptive threshold method.
 */
enum class AdaptiveThresholdMethod
{
    ADAPTIVE_THRESH_MEAN_C     = 0, /*!< Use mean method as adaptive method */
    ADAPTIVE_THRESH_GAUSSIAN_C,     /*!< Use gaussion method as adaptive method */
};

/**
 * @brief Overloaded stream insertion operator for AdaptiveThresholdMethod.
 *
 * This function converts a AdaptiveThresholdMethod to its corresponding string representation
 * and inserts it into an output stream.
 * 
 * @param os The output stream.
 * @param method The AdaptiveThresholdMethod to be inserted.
 *
 * @return A reference to the output stream.
*/
AURA_INLINE std::ostream& operator<<(std::ostream &os, AdaptiveThresholdMethod method)
{
    switch (method)
    {
        case AdaptiveThresholdMethod::ADAPTIVE_THRESH_MEAN_C:
        {
            os << "ADAPTIVE_THRESH_MEAN_C";
            break;
        }

        case AdaptiveThresholdMethod::ADAPTIVE_THRESH_GAUSSIAN_C:
        {
            os << "ADAPTIVE_THRESH_GAUSSIAN_C";
            break;
        }

        default:
        {
            os << "undefined type";
            break;
        }
    }

    return os;
}

/**
 * @brief Convert AdaptiveThresholdMethod to string.
 *
 * This function converts a AdaptiveThresholdMethod to its corresponding string representation.
 *
 * @param method The AdaptiveThresholdMethod to be converted.
 *
 * @return The string representation of the AdaptiveThresholdMethod.
 */
AURA_INLINE const std::string AdaptiveThresholdMethodToString(AdaptiveThresholdMethod method)
{
    std::ostringstream ss;
    ss << method ;
    return ss.str();
}

/**
 * @brief Class representing the AdaptiveThreshold operation.
 *
 * This class represents a AdaptiveThreshold operation, which is a type of thresholding that varies the threshold
 * value for each pixel in the iaura based on the local iaura characteristics. It is recommended to use the `IAdaptiveThreshold` API,
 * which internally calls this class. The only recommended scenario for using this class is when the input or output type is `CLMem`.
 *
 * The approximate internal call within the `IAdaptiveThreshold` function is as follows:
 *
 * @code
 * AdaptiveThreshold adaptivethreshold(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize(),
 * return OpCall(ctx, adaptivethreshold, &src, &dst, thresh, max_val, method, type, block_size, delta);
 * @endcode
 */
class AURA_EXPORTS AdaptiveThreshold : public Op
{
public:
    /**
     * @brief Constructor for AdaptiveThreshold class.
     *
     * @param ctx The pointer to the Context object.
     * @param target The platform on which this function runs.
     */
    AdaptiveThreshold(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Set arguments for the AdaptiveThreshold operation.
     *
     * For more details, please refer to @ref adaptivethreshold_details
     */
    Status SetArgs(const Array *src, Array *dst, DT_F32 max_val, AdaptiveThresholdMethod method,
                   DT_S32 type, DT_S32 block_size, DT_F32 delta);
};

/**
 * @brief Apply a AdaptiveThreshold operation to src matrix.
 *
 * @anchor adaptivethreshold_details
 * This function applies a AdaptiveThreshold operation to src iaura. And in-place operation is not supported.
 *
 * @param ctx The pointer to the Context object.
 * @param src The input iaura.
 * @param dst The output iaura.
 * @param max_val The maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV types.
 * @param method The adaptivethresholding method (ADAPTIVE_THRESH_MEAN_C, ADAPTIVE_THRESH_GAUSSIAN_C).
 * @param type The thresholding type (THRESH_BINARY, THRESH_BINARY_INV, etc.).
 * @param block_size The size value represents the neighborhood block size, used to calculate the regional threshold.
 * @param delta Constant subtracted from the mean or weighted mean. Normally, it is positive but may be zero or negative as well.
 * @param target The platform on which this function runs.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 *
 * ### The Support data types and platforms
 * Platforms | Data type(src or dst)
 * ----------|----------------------------
 * NONE      | U8C1
 *
 */
AURA_EXPORTS Status IAdaptiveThreshold(Context *ctx, const Mat &src, Mat &dst, DT_F32 max_val, AdaptiveThresholdMethod method,
                                       DT_S32 type, DT_S32 block_size, DT_F32 delta, const OpTarget &target = OpTarget::Default());

} // namespace aura

#endif // AURA_OPS_MISC_ADAPTIVE_THRESHOLD_HPP__