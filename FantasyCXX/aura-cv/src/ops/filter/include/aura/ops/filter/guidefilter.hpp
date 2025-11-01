#ifndef AURA_OPS_FILTER_GUIDEFILTER_HPP__
#define AURA_OPS_FILTER_GUIDEFILTER_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

/**
 * @defgroup ops Operators
 * @{
 *    @defgroup imgproc_filter Iaura Filtering
 *    @{
 *          @defgroup guidefilter Guide Filter
 *    @}
 * @}
 */

namespace aura
{

/**
 * @addtogroup guidefilter
 * @{
 */

/**
 * @brief Enum class representing different guidefilter operation types.
 */

enum class GuideFilterType
{
    NORMAL = 0, /*!< Represents normal type guidefilter */
    FAST,       /*!< Represents fast   type guidefilter */
};

/**
 * @brief Overloaded output stream operator for GuideFilterType type enumeration.
 *
 * This operator allows printing GuideFilterType enumerators to an output stream.
 *
 * @param os The output stream.
 * @param type The GuideFilterType type enumerator to be printed.
 *
 * @return GuideFilterType out stream.
 */
AURA_INLINE std::ostream& operator<<(std::ostream &os, GuideFilterType type)
{
    switch (type)
    {
        case GuideFilterType::NORMAL:
        {
            os << "Normal";
            break;
        }

        case GuideFilterType::FAST:
        {
            os << "Fast";
            break;
        }

        default:
        {
            os << "undefined guidefilter type";
            break;
        }
    }

    return os;
}

/**
 * @brief Guidefilter type to string representation.
 *
 * This function converts a GuideFilterType enumerator to its string representation.
 *
 * @param type The GuideFilterType enumerator.
 *
 * @return The string representation of the GuideFilterType.
 */
AURA_INLINE std::string GuideFilterTypeToString(GuideFilterType type)
{
    std::ostringstream ss;
    ss << type;
    return ss.str();
}

/**
 * @brief The guidefilter filter interface class.
 *
 * The use of this class for guidefilter filtering is not recommended. It is recommended to use the `IGuideFilter` API,
 * which internally calls this class. The only recommended scenario for using this class is when the input or output type is `CLMem`.
 *
 * The approximate internal call within the `IGuideFilter` function is as follows:
 *
 * @code
 * GuideFilter guidefilter(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize().
 * return OpCall(ctx, gaussian, &src0, &src1, &dst, ksize, eps, type, border_type, border_value);
 * @endcode
 */
class AURA_EXPORTS GuideFilter : public Op
{
public:
    /**
     * @brief Constructor for the GuideFilter filter class.
     *
     * @param ctx The pointer to the Context object
     * @param target The platform on which this function runs
     */
    GuideFilter(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Set the arguments for the guidefilter filter operation.
     *
     * For more details, please refer to @ref guidefilter_details
     *
     * @note If the type of src or dst is `CLMem` and is an iaura2D memory object, the row pitch should be aligned to a stride,
     * which is obtained through the `GetCLLengthAlignSize` function and is platform-dependent.
     */
    Status SetArgs(const Array *src0, const Array *src1, Array *dst, MI_S32 ksize, MI_F32 eps,
                   GuideFilterType type = GuideFilterType::NORMAL,
                   BorderType border_type = BorderType::REPLICATE,
                   const Scalar &border_value = Scalar());
};
/**
 * @brief Using the guidefilter filter to blur an iaura.
 *
 * @anchor guidefilter_details
 * The function smooths the iaura using the guidefilter filter with the @f$ \texttt{kszie} \times \texttt{kszie} @f$ kernel size.
 * The gaussian kernel standard deviations in X and Y direction should be same. And in-place operation is not supported.
 *
 * @param ctx The pointer to the Context object
 * @param src0 The first input iaura, And see the below for the supported data types.
 * @param src1 The second input iaura, And see the below for the supported data types.
 * @param dst The output iaura, And see the below for the supported data types.
 * @param ksize The filter kernel size, And see the below for the supported sizes.
 * @param eps Regularization term of Guided Filter. eps2 is similar to the sigma in the color space into bilateralFilter.
 * @param type The guidefilter type(NORMAL, FAST).
 * @param border_type The border type for handling border pixels
 * @param border_value The scalar values for border pixels used only when border_type is BorderType::CONSTANT.
 * @param target The platform on which this function runs
 *
 * @return Status::OK if successful; otherwise, an appropriate error status
 *
 * ### The supported data types and platforms
 * Platforms    | Data type(src or dst)                               | ksize
 * -------------|-----------------------------------------------------|-----------
 * NONE         | U8Cx/S8Cx/U16Cx/S16Cx/F16Cx/F32Cx, x = N            | N
 * NEON         | U8Cx/S8Cx/U16Cx/S16Cx/F16Cx/F32Cx, x = 1, 2, 3      | N <= 255
 *
 * @note 1.N is positive integer(ksize is odd positive integer). <br>
 *       2.The above implementations supported all BorderType(CONSTANT/REPLICATE/REFLECT_101).
 */
AURA_EXPORTS Status IGuideFilter(Context *ctx, const Mat &src0, const Mat &src1,
                                 Mat &dst, MI_S32 ksize, MI_F32 eps,
                                 GuideFilterType type = GuideFilterType::NORMAL,
                                 BorderType border_type = BorderType::REPLICATE,
                                 const Scalar &border_value = Scalar(),
                                 const OpTarget &target = OpTarget::Default());
/**
 * @}
 */

} // namespace aura

#endif // AURA_OPS_FILTER_GUIDEFILTER_HPP__