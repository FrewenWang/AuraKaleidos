#ifndef AURA_OPS_FILTER_BILATERAL_HPP__
#define AURA_OPS_FILTER_BILATERAL_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

/**
 * @defgroup ops Operators
 * @{
 *    @defgroup imgproc_filter Iaura Filtering
 *    @{
 *          @defgroup bilateral Bilateral Filter
 *    @}
 * @}
 */

namespace aura
{
/**
 * @addtogroup bilateral
 * @{
 */

/**
 * @brief The Bilateral filter interface class.
 *
 * The use of this class for Bilateral filtering is not recommended. It is recommended to use the `IBilateral` API,
 * which internally calls this class. The only recommended scenario for using this class is when the input or output type is `CLMem`.
 *
 * The approximate internal call within the `IBilateral` function is as follows:
 *
 * @code
 * Bilateral bilateral(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize().
 * return OpCall(ctx, bilateral, &src, &dst, sigma_color, sigma_space, ksize, border_type, border_value);
 * @endcode
 */
class AURA_EXPORTS Bilateral : public Op
{
public:
    /**
     * @brief Constructor for the Bilateral filter class.
     *
     * @param ctx The pointer to the Context object
     * @param target The platform on which this function runs
     */
    Bilateral(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Set the arguments for the Bilateral filter operation.
     *
     * For more details, please refer to @ref bilateral_details
     *
     * @note If the type of src or dst is `CLMem` and is an iaura2D memory object, the row pitch should be aligned to a stride,
     * which is obtained through the `GetCLLengthAlignSize` function and is platform-dependent.
     */
    Status SetArgs(const Array *src, Array *dst, MI_F32 sigma_color, MI_F32 sigma_space,
                   MI_S32 ksize, BorderType border_type = BorderType::REFLECT_101,
                   const Scalar &border_value = Scalar());

    /**
     * @brief Generate bilateral opencl precompiled cache.
     *
     * @param elem_type The element type of the src/dst array.
     * @param channel The channel of the src/dst array.
     * @param ksize The size of the bilateral kernel.
     * @param border_type The border type for handling border pixels.
     */
    static Status CLPrecompile(Context *ctx, ElemType elem_type, MI_S32 channel, MI_S32 ksize, BorderType border_type);
};

/**
 * @brief Using the bilateral filter to denoise an iaura.
 *
 * @anchor bilateral_details
 * The function denoises the iaura using the bilateral filter with the @f$ \texttt{kszie} \times \texttt{kszie} @f$ kernel size.
 * The bilateral kernel standard deviations in X and Y direction should be same. And in-place operation is not supported.
 *
 * @param ctx The pointer to the Context object
 * @param src The input iaura, and see below for the supported data types.
 * @param dst The output iaura, and see below for the supported data types.
 * @param sigma_color The filter sigma in the color space.
 * @param sigma_space The filter sigma in the coordinate space.
 * @param ksize The filter kernel size, and see below for the supported sizes.
 * @param border_type The border type for handling border pixels
 * @param border_value The scalar values for border pixels used only when border_type is BorderType::CONSTANT.
 * @param target The platform on which this function runs
 *
 * @return Status::OK if successful; otherwise, an appropriate error status
 *
 * ### The supported data types and platforms
 * Platforms    | Data type(src or dst)         | ksize
 * -------------|-------------------------------|-----------
 * NONE         | U8Cx/F16Cx/F32Cx, x = 1, 3    | N
 * NEON         | U8Cx,             x = 1, 3    | 3
 * OpenCL       | U8Cx/F16Cx/F32Cx, x = 1, 3    | 3
 * HVX          | -                             | -
 *
 * @note N is odd positive integer; And the above implementations supported all BorderType(CONSTANT/REPLICATE/REFLECT_101).
 */
AURA_EXPORTS Status IBilateral(Context *ctx, const Mat &src, Mat &dst, MI_F32 sigma_color, MI_F32 sigma_space,
                               MI_S32 ksize, BorderType border_type = BorderType::REFLECT_101,
                               const Scalar &border_value = Scalar(),
                               const OpTarget &target = OpTarget::Default());

} // namespace aura

#endif // AURA_OPS_FILTER_BILATERAL_HPP__