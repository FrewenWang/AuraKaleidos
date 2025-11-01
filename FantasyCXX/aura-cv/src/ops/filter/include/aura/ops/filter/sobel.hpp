#ifndef AURA_OPS_FILTER_SOBEL_HPP__
#define AURA_OPS_FILTER_SOBEL_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"
/**
 * @defgroup ops Operators
 * @{
 *    @defgroup imgproc_filter Iaura Filtering
 *    @{
 *          @defgroup sobel Sobel Filter
 *    @}
 * @}
 */

namespace aura
{
/**
 * @addtogroup sobel
 * @{
 */

/**
 * @brief Sobel filter class.
 *
 * The use of this class for Sobel filtering is not recommended. It is recommended to use the `ISobel` API,
 * which internally calls this class. The only recommended scenario for using this class is when the input or output type is `CLMem`.
 *
 * The approximate internal call within the `ISobel` function is as follows:
 *
 * @code
 * Sobel sobel(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize().
 * return OpCall(ctx, sobel, &src, &dst, ...);
 * @endcode
 */
class AURA_EXPORTS Sobel : public Op
{
public:
    /**
     * @brief Constructor for Sobel class.
     *
     * @param ctx The pointer to the Context object
     * @param target The platform on which this function runs
     */
    Sobel(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Set the arguments for the Sobel filter operation.
     *
     * For more details, please refer to @ref sobel_details
     *
     * @note If the type of src or dst is `CLMem` and is an iaura2D memory object, the row pitch should be aligned to a stride,
     * which is obtained through the `GetCLLengthAlignSize` function and is platform-dependent.
     */
    Status SetArgs(const Array *src, Array *dst, MI_S32 dx, MI_S32 dy, MI_S32 ksize, MI_F32 scale = 1.f,
                   BorderType border_type = BorderType::REFLECT_101, const Scalar &border_value = Scalar());

    /**
     * @brief Generate sobel opencl precompiled cache.
     *
     * @param dx The order of the derivative in x.
     * @param dy The order of the derivative in y.
     * @param ksize The size of the sobel kernel.
     * @param scale Optional scale factor.
     * @param border_type The border type for handling border pixels.
     * @param channel The channel of the dst array.
     * @param src_elem_type The element type of the src array.
     * @param dst_elem_type The element type of the dst array.
     */
    static Status CLPrecompile(Context *ctx, MI_S32 dx, MI_S32 dy, MI_S32 ksize, MI_F32 scale, BorderType border_type,
                               MI_S32 channel, ElemType src_elem_type, ElemType dst_elem_type);
};

/**
 * @brief Sobel filter function.
 *
 * @anchor sobel_details
 * This function applies a Sobel filter to the source matrix and stores the result in the destination matrix.
 * And in-place operation is not supported.
 *
 * @param ctx The context in which the Sobel filter operates.
 * @param src The source matrix. And see the below for the supported data types.
 * @param dst The destination matrix. And see the below for the supported data types.
 * @param dx The order of the derivative in x.
 * @param dy The order of the derivative in y.
 * @param ksize The size of the extended Sobel kernel.
 * @param scale Optional scale factor (default is 1.f).
 * @param border_type The border type for handling border pixels
 * @param border_value The scalar values for border pixels used only when border_type is BorderType::CONSTANT.
 * @param target The platform on which this function runs.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status
 *
 * ### The supported data types and platforms
 * Platforms    | Data type(src:dst)                                                                           | channel       | ksize
 *--------------|----------------------------------------------------------------------------------------------|---------------|-----------
 * NONE         | U8:U8, U8:U16, U8:S16, U8:F32, U16:U16, U16:F32, S16:S16, S16:F32, F16:F16, F16:F32, F32:F32 | N             | N
 * NEON         | U8:S16, U8:F32, F32:F32                                                                      | 1, 2, 3       | 1, 3, 5
 * OPENCL       | U8:S16, U8:F32, U16:U16, S16:S16, F16:F16, F32:F32                                           | 1, 2, 3       | 1, 3, 5
 * HVX          | U8:S16,                                                                                      | 1, 2, 3       | 1, 3, 5
 *
 * @note 1.N is positive integer(ksize is odd positive integer). <br>
 *       2.The above implementations supported all BorderType(CONSTANT/REPLICATE/REFLECT_101).
 */
AURA_EXPORTS Status ISobel(Context *ctx, const Mat &src, Mat &dst, MI_S32 dx, MI_S32 dy, MI_S32 ksize, MI_F32 scale = 1.f,
                           BorderType border_type = BorderType::REFLECT_101, const Scalar &border_value = Scalar(),
                           const OpTarget &target = OpTarget::Default());
/**
 * @}
 */
} // namespace aura

#endif // AURA_OPS_FILTER_SOBEL_HPP__