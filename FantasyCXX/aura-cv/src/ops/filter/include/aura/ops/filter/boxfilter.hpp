#ifndef AURA_OPS_FILTER_BOXFILTER_HPP__
#define AURA_OPS_FILTER_BOXFILTER_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

/**
 * @defgroup ops Operators
 * @{
 *    @defgroup imgproc_filter Iaura Filtering
 *    @{
 *          @defgroup boxfilter BoxFilter
 *    @}
 * @}
 */

namespace aura
{
/**
 * @addtogroup boxfilter
 * @{
 */

/**
 * @brief The BoxFilter interface class.
 *
 * The use of this class for BoxFilter filtering is not recommended. It is recommended to use the `IBoxfilter` API,
 * which internally calls this class. The only recommended scenario for using this class is when the input or output type is `CLMem`.
 *
 * The approximate internal call within the `IBoxfilter` function is as follows:
 *
 * @code
 * BoxFilter boxfilter(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize().
 * return OpCall(ctx, boxfilter, &src, &dst, ksize, sigma, border_type, border_value);
 * @endcode
 */
class AURA_EXPORTS BoxFilter : public Op
{
public:
    /**
     * @brief Constructor for the BoxFilter filter class.
     *
     * @param ctx The pointer to the Context object
     * @param target The platform on which this function runs
     */
    BoxFilter(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Set the arguments for the BoxFilter filter operation.
     *
     * For more details, please refer to @ref boxfilter_details
     *
     * @note If the type of src or dst is `CLMem` and is an iaura2D memory object, the row pitch should be aligned to a stride,
     * which is obtained through the `GetCLLengthAlignSize` function and is platform-dependent.
     */
    Status SetArgs(const Array *src, Array *dst, DT_S32 ksize,
                   BorderType border_type = BorderType::REFLECT_101,
                   const Scalar &border_value = Scalar());

    /**
     * @brief Generate boxfilter opencl precompiled cache.
     *
     * @param elem_type The element type of the src/dst array.
     * @param channel The channel of the src/dst array.
     * @param ksize The size of the boxfilter kernel.
     * @param border_type The border type for handling border pixels.
     */
    static Status CLPrecompile(Context *ctx, ElemType elem_type, DT_S32 channel, DT_S32 ksize, BorderType border_type);
};

/**
 * @brief Using the boxfilter filter to blur an iaura.
 *
 * @anchor boxfilter_details
 * The function smooths the iaura using the boxfilter filter with the @f$ \texttt{kszie} \times \texttt{kszie} @f$ kernel size.
 *
 * @param ctx The pointer to the Context object
 * @param src The input iaura, And see the below for the supported data types.
 * @param dst The output iaura, And see the below for the supported data types.
 * @param ksize The filter kernel size(must be odd number), And see the below for the supported sizes.
 * @param border_type The border type for handling border pixels
 * @param border_value The scalar values for border pixels used only when border_type is BorderType::CONSTANT.
 * @param target The platform on which this function runs
 *
 * @return Status::OK if successful; otherwise, an appropriate error status
 *
 * ### The supported data types and platforms
 * Platforms    | Data type(src or dst)                                    | ksize
 * -------------|----------------------------------------------------------|-----------
 * NONE         | U8Cx/S8Cx/U16Cx/S16Cx/F16Cx/U32Cx/S32Cx/F32Cx, x = N     | N, odd
 * NEON         | U8Cx/S8Cx/U16Cx/S16Cx, x = 1, 2, 3                       | N < 256, odd
 * NEON         | F16Cx/F32Cx, x = 1, 2, 3                                 | N, odd
 * OpenCL       | U8Cx/U16Cx/S16Cx/F16Cx/F32Cx, x = 1, 2, 3                | 3, 5
 * OpenCL       | U8Cx/U16Cx/S16Cx/F16Cx/F32Cx, x = 1,                     | 7
 * HVX          | U8Cx,        x = 1, 2, 3                                 | N < 128, odd
 * HVX          | U16Cx/S16Cx, x = 1, 2, 3                                 | N < 64, odd
 *
 * @note 1.N is positive integer(ksize is odd positive integer). <br>
 *       2.The above implementations supported all BorderType(CONSTANT/REPLICATE/REFLECT_101).
 */
AURA_EXPORTS Status IBoxfilter(Context *ctx, const Mat &src, Mat &dst, DT_S32 ksize,
                               BorderType border_type = BorderType::REFLECT_101,
                               const Scalar &border_value = Scalar(),
                               const OpTarget &target = OpTarget::Default());
} // namespace aura

#endif // AURA_OPS_FILTER_BOXFILTER_HPP__