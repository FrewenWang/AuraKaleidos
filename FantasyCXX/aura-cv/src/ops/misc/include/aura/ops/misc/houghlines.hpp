#ifndef AURA_OPS_MISC_HOUGH_LINES_HPP__
#define AURA_OPS_MISC_HOUGH_LINES_HPP__

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
 * @brief Enumerated type for hough lines type.
 */
enum class LinesType
{
    VEC2F = 0, /*!< Use 2 float data as an element */
    VEC3F,     /*!< Use 3 float data as an element */
};

/**
 * @brief Overloaded stream insertion operator for LinesType.
 *
 * This function converts a LinesType to its corresponding string representation
 * and inserts it into an output stream.
 * 
 * @param os The output stream.
 * @param type The LinesType to be inserted.
 *
 * @return A reference to the output stream.
 */
AURA_INLINE std::ostream& operator<<(std::ostream &os, LinesType type)
{
    switch (type)
    {
        case LinesType::VEC2F:
        {
            os << "vec2f";
            break;
        }

        case LinesType::VEC3F:
        {
            os << "vec3f";
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
 * @brief Converts a LinesType to its corresponding string representation.
 *
 * @param type The LinesType to be converted.
 *
 * @return The string representation of the LinesType.
 */
AURA_INLINE std::string LinesTypeToString(LinesType type)
{
    std::stringstream sstream;
    sstream << type;
    return sstream.str();
}

/**
 * @brief Class representing the HoughLines operation.
 *
 * This class represents a HoughLines operation, which is used to detect straight lines in an iaura.
 * It is recommended to use the `IHoughLines` API, which internally calls this class.
 * The only recommended scenario for using this class is when the input or output type is `CLMem`.
 *
 * The approximate internal call within the `IHoughLines` function is as follows:
 *
 * @code
 * HoughLines houghlines(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize(),
 * return OpCall(ctx, houghlines, &src, &lines, line_type, ....);
 * @endcode
 */
class AURA_EXPORTS HoughLines : public Op
{
public:
    /**
     * @brief Constructor for HoughLines class.
     *
     * @param ctx The pointer to the Context object.
     * @param target The platform on which this function runs.
     */
    HoughLines(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Set arguments for the HoughLines operation.
     *
     * For more details, please refer to @ref houghlines_details
     */
    Status SetArgs(const Array *src, std::vector<Scalar> &lines, LinesType line_type, MI_F64 rho, MI_F64 theta, MI_S32 threshold,
                   MI_F64 srn = 0, MI_F64 stn = 0, MI_F64 min_theta = 0, MI_F64 max_theta = AURA_PI);
};

/**
 * @brief Class representing the HoughLinesP operation.
 *
 * This class represents a HoughLinesP operation, which is used to detect lines in an iaura using the probabilistic Hough transform.
 * It is recommended to use the `IHoughLinesP` API, which internally calls this class.
 * The only recommended scenario for using this class is when the input or output type is `CLMem`.
 *
 * The approximate internal call within the `IHoughLinesP` function is as follows:
 *
 * @code
 * HoughLinesP houghlinesp(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize(),
 * return OpCall(ctx, houghlinesp, &src, &lines, rho, theta, ...);
 * @endcode
 */
class AURA_EXPORTS HoughLinesP : public Op
{
public:
    /**
     * @brief Constructor for HoughLinesP class.
     *
     * @param ctx The pointer to the Context object.
     * @param target The platform on which this function runs.
     */
    HoughLinesP(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Set arguments for the HoughLinesP operation.
     *
     * For more details, please refer to @ref houghlinesp_details
     */
    Status SetArgs(const Array *src, std::vector<Scalari> &lines, MI_F64 rho, MI_F64 theta, MI_S32 threshold,
                   MI_F64 min_line_length, MI_F64 max_gap);
};

/**
 * @brief Finds lines in a binary iaura using the standard Hough transform.
 *
 * @anchor houghlines_details
 * This function applies a HoughLines operation to input iaura.
 *
 * @param ctx The pointer to the Context object.
 * @param mat The input iaura.
 * @param lines The output vector of lines.
 * @param line_type The line_type to use with the VEC2F and VEC3F types.
 * @param rho The distance resolution of the accumulator in pixels.
 * @param theta The angle resolution of the accumulator in radians.
 * @param threshold The accumulator threshold parameter, only those lines are returned that get enough votes(>threshold).
 * @param srn For the multi-scale Hough transform, it is a divisor for the distance resolution rho.
 * @param stn For the multi-scale Hough transform, it is a divisor for the distance resolution theta.
 * @param min_theta For standard and multi-scale Hough transform, minimum angle to check for lines. Must fall between 0 and max_theta.
 * @param max_theta For standard and multi-scale Hough transform, an upper bound for the angle.
 * @param target The platform on which this function runs.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 *
 * ### The Support data types and platforms
 * Platforms | Data type(mat)
 * ----------|----------------------------
 * NONE      | U8C1
 *
 * @note The above implementations support LinesType(VEC2F/VEC3F).
 */
AURA_EXPORTS Status IHoughLines(Context *ctx, const Mat &mat, std::vector<Scalar> &lines, LinesType line_type, MI_F64 rho, MI_F64 theta, MI_S32 threshold,
                                MI_F64 srn = 0, MI_F64 stn = 0, MI_F64 min_theta = 0, MI_F64 max_theta = AURA_PI, const OpTarget &target = OpTarget::Default());

/**
 * @brief Finds line segments in a binary iaura using the probabilistic Hough transform.
 *
 * @anchor houghlinesp_details
 * This function applies a HoughLinesP operation to input iaura. And in-place operation is not supported.
 *
 * @param ctx The pointer to the Context object.
 * @param mat The input iaura.
 * @param lines The output vector of lines.
 * @param line_type The line_type to use with the VEC2F and VEC3F types.
 * @param rho The distance resolution of the accumulator in pixels.
 * @param theta The angle resolution of the accumulator in radians.
 * @param threshold The accumulator threshold parameter, only those lines are returned that get enough votes(>threshold).
 * @param min_line_length Minimum line length. Line segments shorter than that are rejected.
 * @param max_gap Maximum allowed gap between points on the same line to link them.
 * @param target The platform on which this function runs.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 *
 * ### The Support data types and platforms
 * Platforms | Data type(mat)
 * ----------|----------------------------
 * NONE      | U8C1
 *
 */
AURA_EXPORTS Status IHoughLinesP(Context *ctx, const Mat &mat, std::vector<Scalari> &lines, MI_F64 rho, MI_F64 theta, MI_S32 threshold,
                                 MI_F64 min_line_length, MI_F64 max_gap, const OpTarget &target = OpTarget::Default());

} // namespace aura

#endif // AURA_OPS_MISC_HOUGH_LINES_HPP__