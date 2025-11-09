#ifndef AURA_OPS_FEATURE2D_FAST_HPP__
#define AURA_OPS_FEATURE2D_FAST_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

/**
 * @defgroup ops Operators
 * @{
 *    @defgroup feature2d Feature Detection and Description
 * @}
 */

namespace aura
{
/**
 * @addtogroup feature2d
 * @{
 */


/**
 * @brief Enumeration of FAST corner detector types.
 */
enum class FastDetectorType
{
    FAST_5_8    = 0, /*!< 5 points in the circle of radius 8 */
    FAST_7_12,       /*!< 7 points in the circle of radius 12 */
    FAST_9_16,       /*!< 9 points in the circle of radius 16 */
};

AURA_INLINE std::ostream& operator<<(std::ostream &os, FastDetectorType type)
{
    switch (type)
    {
        case FastDetectorType::FAST_5_8:
        {
            os << "FAST_5_8";
            break;
        }

        case FastDetectorType::FAST_7_12:
        {
            os << "FAST_7_12";
            break;
        }

        case FastDetectorType::FAST_9_16:
        {
            os << "FAST_9_16";
            break;
        }

        default:
        {
            os << "undefined fast detector type";
            break;
        }
    }

    return os;
}

AURA_INLINE const std::string FastDetectorTypeToString(FastDetectorType type)
{
    std::ostringstream ss;
    ss << type ;
    return ss.str();
}

/**
 * @brief FAST corner detector interface class.
 *
 * The use of this class for FAST is not recommended. It is recommended to use the `IFast` API,
 * which internally calls this class. The only recommended scenario for using this class is when the input or output type is `CLMem`.
 * 
 * The approximate internal call within the `IFast` function is as follows:
 * 
 * @code
 * Fast fast(ctx, target);
 * 
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize().
 * return OpCall(ctx, fast, &src, ...);
 * @endcode
 */
class AURA_EXPORTS Fast : public Op
{
public:
    /**
     * @brief Constructor for the Canny operation.
     *
     * @param ctx The pointer to the Context object
     * @param target The platform on which this function runs
     */
    Fast(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Set the arguments for the Fast operation.
     *
     * For more details, please refer to @ref fast_details
     *
     * @note If the type of src or dst is `CLMem` and is an iaura2D memory object, the row pitch should be aligned to a stride,
     * which is obtained through the `GetCLLengthAlignSize` function and is platform-dependent. 
     */
    Status SetArgs(const Array *src, std::vector<KeyPoint> &key_points, DT_S32 threshold, DT_BOOL nonmax_suppression,
                   FastDetectorType type = FastDetectorType::FAST_9_16, DT_U32 max_num_corners = 3000);
};

/**
 * @brief FAST corner detector function.
 *
 * @anchor fast_details
 * This function applies the FAST corner detector to the input iaura.
 *
 * @param ctx The pointer to the Context object
 * @param src The input iaura, And see the below for the supported data types.
 * @param key_points The output vector containing the detected keypoints.
 * @param threshold The threshold for the FAST detector.
 * @param nonmax_suppression Flag indicating whether to use non-maximum suppression (default is DT_TRUE).
 * @param type The type of FAST corner detector (default is FastDetectorType::FAST_9_16).
 * @param max_num_corners The maximum number of corners to detect (default is 3000).
 * @param target The platform on which this function runs
 * 
 * @return Status::OK if successful; otherwise, an appropriate error status.
 * 
 * ### The supported data types and platforms
 * Platforms | Data type (src)  | FastDetectorType
 * ----------|------------------|------------------------------
 * NONE      | U8C1             | FAST_5_8/FAST_7_12/FAST_9_16
 * NEON      | U8C1             | FAST_9_16
 * HVX       | U8C1             | FAST_9_16
 */
AURA_EXPORTS Status IFast(Context *ctx, const Mat &src, std::vector<KeyPoint> &key_points, DT_S32 threshold,
                          DT_BOOL nonmax_suppression, FastDetectorType type = FastDetectorType::FAST_9_16,
                          DT_U32 max_num_corners = 3000, const OpTarget &target = OpTarget::Default());
/**
 * @}
*/
} // namespace aura

#endif // AURA_OPS_FEATURE2D_FAST_HPP__