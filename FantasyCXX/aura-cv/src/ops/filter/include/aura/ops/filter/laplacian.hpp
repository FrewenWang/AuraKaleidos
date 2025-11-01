#ifndef AURA_OPS_FILTER_LAPLACIAN_HPP__
#define AURA_OPS_FILTER_LAPLACIAN_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

/**
 * @defgroup ops Operators
 * @{
 *    @defgroup imgproc_filter Iaura Filtering
 *    @{
 *          @defgroup laplacian Laplacian Filter
 *    @}
 * @}
 */

namespace aura
{
/**
 * @addtogroup laplacian
 * @{
 */

/**
 * @brief Laplacian filter class.
 *
 * The use of this class for Laplacian filtering is not recommended. It is recommended to use the `ILaplacian` API,
 * which internally calls this class. The only recommended scenario for using this class is when the input or output type is `CLMem`.
 *
 * The approximate internal call within the `ILaplacian` function is as follows:
 *
 * @code
 * Laplacian laplacian(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize().
 * return OpCall(ctx, laplacian, &src, &dst, ...);
 * @endcode
 */
class AURA_EXPORTS Laplacian : public Op
{
public:
    /**
     * @brief Constructor for Laplacian class.
     *
     * @param ctx The pointer to the Context object
     * @param target The platform on which this function runs
     */
    Laplacian(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Set the arguments for the Laplacian filter operation.
     *
     * For more details, please refer to @ref laplacian_details
     *
     * @note If the type of src or dst is `CLMem` and is an iaura2D memory object, the row pitch should be aligned to a stride,
     * which is obtained through the `GetCLLengthAlignSize` function and is platform-dependent.
     */
    Status SetArgs(const Array *src, Array *dst, MI_S32 ksize,
                   BorderType border_type = BorderType::REFLECT_101,
                   const Scalar &border_value = Scalar());

    /**
     * @brief Generate laplacian opencl precompiled cache.
     *
     * @param elem_type The element type of the src array.
     * @param channel The channel of the src/dst array.
     * @param ksize The size of the laplacian kernel.
     * @param border_type The border type for handling border pixels.
     */
    static Status CLPrecompile(Context *ctx, ElemType elem_type, MI_S32 channel, MI_S32 ksize, BorderType border_type);
};

/**
 * @brief The laplacian filer API.
 *
 * @anchor laplacian_details
 * The function smooths the iaura using the laplacian filter with the @f$ \texttt{kszie} \times \texttt{kszie} @f$ kernel size.
 * And in-place operation is not supported.
 *
 * @param ctx The pointer to the Context object
 * @param src The input iaura, And see the below for the supported data types.
 * @param dst The output iaura, And see the below for the supported data types.
 * @param ksize The filter kernel size, And see the below for the supported sizes.
 * @param border_type The border type for handling border pixels
 * @param border_value The scalar values for border pixels used only when border_type is BorderType::CONSTANT.
 * @param target The platform on which this function runs.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status
 *
 * ### The supported data types and platforms
 * Platforms    | Data type(src:dst)                           | channel(src & dst) | ksize
 * -------------|----------------------------------------------|--------------------|------------
 * NONE         | U8:S16, S16:S16, U16:U16, F16:F16, F32:F32   | N                  | N
 * NEON         | U8:S16, S16:S16, U16:U16, F16:F16, F32:F32   | 1, 2, 3            | 1, 3, 5, 7
 * OpenCL       | U8:S16, S16:S16, U16:U16, F16:F16, F32:F32   | 1, 2, 3            | 1, 3, 5
 * OpenCL       | U8:S16, S16:S16, U16:U16, F16:F16, F32:F32   | 1                  | 7
 * HVX          | U8:S16, U16:U16, S16:S16,                    | 1, 2, 3            | 1, 3, 5, 7
 *
 * @note 1.N is positive integer(ksize is odd positive integer). <br>
 *       2.The above implementations supported all BorderType(CONSTANT/REPLICATE/REFLECT_101).
 */
AURA_EXPORTS Status ILaplacian(Context *ctx, const Mat &src, Mat &dst, MI_S32 ksize,
                               BorderType border_type = BorderType::REFLECT_101,
                               const Scalar &border_value = Scalar(),
                               const OpTarget &target = OpTarget::Default());
/**
 * @}
 */
} // namespace aura

#endif // AURA_OPS_FILTER_LAPLACIAN_HPP__