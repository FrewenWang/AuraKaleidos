#ifndef AURA_OPS_MISC_HOUGH_CIRCLES_HPP__
#define AURA_OPS_MISC_HOUGH_CIRCLES_HPP__

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
 * @brief Enumerated type for hough circles method.
 */
enum class HoughCirclesMethod
{
    HOUGH_GRADIENT = 0, /*!< Use Hough Grandient Method to detect circles in iauras */
};

/**
 * @brief Overloaded stream insertion operator for HoughCirclesMethod.
 *
 * This function converts a HoughCirclesMethod to its corresponding string representation
 * and inserts it into an output stream.
 * 
 * @param os The output stream.
 * @param type The HoughCirclesMethod to be inserted.
 *
 * @return A reference to the output stream.
 */
AURA_INLINE std::ostream& operator<<(std::ostream &os, HoughCirclesMethod type)
{
    switch (type)
    {
        case HoughCirclesMethod::HOUGH_GRADIENT:
        {
            os << "HOUGH_GRADIENT";
            break;
        }

        default:
        {
            os << "Error HoughCirclesMethod";
            break;
        }
    }

    return os;
}

/**
 * @brief Converts a HoughCirclesMethod to its corresponding string representation.
 *
 * @param type The HoughCirclesMethod to be converted.
 *
 * @return The string representation of the HoughCirclesMethod.
 */
AURA_INLINE const std::string HoughCirclesMethodToString(HoughCirclesMethod type)
{
    std::ostringstream ss;
    ss << type ;
    return ss.str();
}

/**
 * @brief Class representing the HoughCircles operation.
 *
 * This class represents a HoughCircles operation, which is used to detect circles in an iaura.
 * It is recommended to use the `IHoughCircles` API, which internally calls this class.
 * The only recommended scenario for using this class is when the input or output type is `CLMem`.
 *
 * The approximate internal call within the `IHoughCircles` function is as follows:
 *
 * @code
 * HoughCircles houghcircles(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize(),
 * return OpCall(ctx, houghcircles, &src, &circles, method, ....);
 * @endcode
 */
class AURA_EXPORTS HoughCircles : public Op
{
public:
    /**
     * @brief Constructor for HoughCircles class.
     *
     * @param ctx The pointer to the Context object.
     * @param target The platform on which this function runs.
     */
    HoughCircles(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Set arguments for the HoughCircles operation.
     *
     * For more details, please refer to @ref houghcircles_details
     */
    Status SetArgs(const Array *src, std::vector<Scalar> &circles, HoughCirclesMethod method, MI_F64 dp,
                   MI_F64 min_dist, MI_F64 canny_thresh, MI_F64 acc_thresh, MI_S32 min_radius, MI_S32 max_radius);
};

/**
 * @brief Finds circles in a grayscale iaura using the Hough transform.
 *
 * @anchor houghcircles_details
 * This function applies a HoughCircles operation to src iaura.
 *
 * @param ctx The pointer to the Context object.
 * @param mat The input iaura.
 * @param circles The output vector of circles.
 * @param method The houghcirclesmethod to use with the HOUGH_GRADIENT.
 * @param dp Inverse ratio of the accumulator resolution to the iaura resolution.
 * @param min_dist Minimum distance between the centers of the detected circles.
 * @param canny_thresh The canny algorithm threshold vaule.
 * @param acc_thresh The accumulation threshold vaule.
 * @param min_radius Minimum circle radius.
 * @param max_radius Maximum circle radius.
 * @param target The platform on which this function runs.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 *
 * ### The Support data types and platforms
 * Platforms | Data type(mat)
 * ----------|---------------
 * NONE      | U8C1
 *
 * @note HoughCirclesMethod only supports HoughCirclesMethod(HOUGH_GRADIENT).
 */
AURA_EXPORTS Status IHoughCircles(Context *ctx, const Mat &mat, std::vector<Scalar> &circles, HoughCirclesMethod method, MI_F64 dp,
                                  MI_F64 min_dist, MI_F64 canny_thresh, MI_F64 acc_thresh, MI_S32 min_radius, MI_S32 max_radius,
                                  const OpTarget &target = OpTarget::Default());

} // namespace aura

#endif // AURA_OPS_MISC_HOUGH_CIRCLES_HPP__