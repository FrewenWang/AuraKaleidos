#ifndef AURA_OPS_MISC_FIND_CONTOURS_HPP__
#define AURA_OPS_MISC_FIND_CONTOURS_HPP__

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
namespace aura
{
/**
 * @addtogroup misc
 * @{
 */

/**
 * @brief Enum class representing different contours detection methods.
 */
enum class ContoursMethod
{
    CHAIN_APPROX_NONE,     /*!< get all contour waypoints */
    CHAIN_APPROX_SIMPLE,   /*!< get only contour extreme points */
};

/**
 * @brief Enum class representing different contours detection modes.
 */
enum class ContoursMode
{
    RETR_EXTERNAL,         /*!< retrieves only the extreme outer contours */
    RETR_LIST,             /*!< retrieves all contours without establishing any hierarchical relationships. NOT SUPPORTED YET*/
    // RETR_CCOMP,            /*!< retrieves all of the contours and organizes them into a two-level hierarchy */
    // RETR_TREE,             /*!< retrieves all of the contours and reconstructs a full hierarchy of nested contours */
};

/**
 * @brief Overloaded stream insertion operator for ContoursMethod.
 *
 * This function converts a ContoursMethod to its corresponding string representation
 * and inserts it into an output stream.
 *
 * @param os The output stream.
 * @param binary_type The ContoursMethod to be inserted.
 *
 * @return A reference to the output stream.
 */
AURA_INLINE std::ostream& operator<<(std::ostream &os, ContoursMethod method)
{
    switch (method)
    {
        case ContoursMethod::CHAIN_APPROX_NONE:
        {
            os << "CHAIN_APPROX_NONE";
            break;
        }

        case ContoursMethod::CHAIN_APPROX_SIMPLE:
        {
            os << "CHAIN_APPROX_SIMPLE";
            break;
        }

        default:
        {
            os << "undefined ContoursMethod type";
            break;
        }
    }

    return os;
}

/**
 * @brief Overloaded stream insertion operator for ContoursMode.
 *
 * This function converts a ContoursMode to its corresponding string representation
 * and inserts it into an output stream.
 *
 * @param os The output stream.
 * @param binary_type The ContoursMode to be inserted.
 *
 * @return A reference to the output stream.
 */
AURA_INLINE std::ostream& operator<<(std::ostream &os, ContoursMode mode)
{
    switch (mode)
    {
        case ContoursMode::RETR_EXTERNAL:
        {
            os << "RETR_EXTERNAL";
            break;
        }

        case ContoursMode::RETR_LIST:
        {
            os << "RETR_LIST";
            break;
        }

        default:
        {
            os << "undefined ContoursMode type";
            break;
        }
    }

    return os;
}

/**
 * @brief Converts a ContoursMethod to its string representation.
 *
 * @param method The ContoursMethod to be converted.
 *
 * @return The string representation of the ContoursMethod.
 */
AURA_INLINE std::string FindContoursMethodToString(ContoursMethod method)
{
    std::ostringstream ss;
    ss << method;
    return ss.str();
}

/**
 * @brief Converts a ContoursMode to its string representation.
 *
 * @param mode The ContoursMode to be converted.
 *
 * @return The string representation of the ContoursMode.
 */
AURA_INLINE std::string FindContoursModeToString(ContoursMode mode)
{
    std::ostringstream ss;
    ss << mode;
    return ss.str();
}

/**
 * @brief Class representing the FindContours operation.
 *
 * This class represents a contour detection operation.
 * It is recommended to use the `IFindContours` API, which internally calls this class.
 *
 * The approximate internal call within the `IFindContours` function is as follows:
 *
 * @code
 * FindContours find_contours(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize(),
 * return OpCall(ctx, find_contours, &src, contours, hierarchy, mode, method, offset); 
 * @endcode
 */
class AURA_EXPORTS FindContours : public Op
{
public:
    /**
     * @brief Constructor for FindContours class.
     *
     * @param ctx The pointer to the Context object.
     * @param target The platform on which this function runs.
     */
    FindContours(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Set arguments for the Threshold operation.
     *
     * For more details, please refer to @ref find_contours_details
     */
    Status SetArgs(const Array *src, std::vector<std::vector<Point2i>> &contours, std::vector<Scalari> &hierarchy, 
                   ContoursMode mode = ContoursMode::RETR_EXTERNAL, ContoursMethod method = ContoursMethod::CHAIN_APPROX_SIMPLE, 
                   Point2i offset = Point2i());
};

/**
 * @brief Apply a contour detection operation of src mat.
 *
 * @anchor find_contours_details
 * This function applies a contour detection operation to the source matrix. (based on the Suzuki algorithm)
 *
 * @param ctx The pointer to the Context object.
 * @param src The source matrix.
 * @param contours The 2-dimension vector to save contours result.
 * @param hierarchy The 1-dimension vector to save the relationship between contours.
 * @param ContoursMode The contours detection mode(RETR_EXTERNAL, RETR_LIST).
 * @param ContoursMethod The contours detection method (CHAIN_APPROX_NONE, CHAIN_APPROX_SIMPLE, etc.).
 * @param offset The beginning detect offset point, Default is (0, 0).
 * @param target The platform on which this function runs.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 *
 * ### The Support data types and platforms
 * Platforms | Data type(src)
 * ----------|------------------------------------------
 * NONE      | U8C1
 *
 * @note ContoursMode only support RETR_EXTERNAL for now.
 */
AURA_EXPORTS Status IFindContours(Context *ctx, const Mat &src, std::vector<std::vector<Point2i>> &contours,
                                  std::vector<Scalari> &hierarchy, ContoursMode mode = ContoursMode::RETR_EXTERNAL,
                                  ContoursMethod method = ContoursMethod::CHAIN_APPROX_SIMPLE,
                                  Point2i offset = Point2i(), const OpTarget &target = OpTarget::Default());

/**
 * @}
 */

} // namespace aura

#endif // AURA_OPS_MISC_FIND_CONTOURS_HPP__