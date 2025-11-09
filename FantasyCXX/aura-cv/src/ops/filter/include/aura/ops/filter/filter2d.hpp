#ifndef AURA_OPS_FILTER_FILTER2D_HPP__
#define AURA_OPS_FILTER_FILTER2D_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

/**
 * @defgroup ops Operators
 * @{
 *    @defgroup imgproc_filter Iaura Filtering
 *    @{
 *          @defgroup filter2d Filter2d Filter
 *    @}
 * @}
 */

namespace aura
{
/**
 * @addtogroup filter2d
 * @{
 */

/**
 * @brief 2D filter class.
 *
 * The use of this class for 2D filter is not recommended. It is recommended to use the `IFilter2d` API,
 * which internally calls this class. The only recommended scenario for using this class is when the input or output type is `CLMem`.
 *
 * The approximate internal call within the `IFilter2d` function is as follows:
 *
 * @code
 * Filter2d filter2d(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize().
 * return OpCall(ctx, filter2d, &src, ...);
 * @endcode
 */
class AURA_EXPORTS Filter2d : public Op
{
public:
    /**
     * @brief Constructor for Filter2d class.
     *
     * @param ctx The pointer to the Context object
     * @param target The platform on which this function runs
     */
    Filter2d(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Set the arguments for the Filter2d operation.
     *
     * For more details, please refer to @ref filter2d_details
     *
     * @note If the type of src or dst is `CLMem` and is an iaura2D memory object, the row pitch should be aligned to a stride,
     * which is obtained through the `GetCLLengthAlignSize` function and is platform-dependent.
     */
    Status SetArgs(const Array *src, Array *dst, const Array *kmat,
                   BorderType border_type = BorderType::REFLECT_101,
                   const Scalar &border_value = Scalar());

    /**
     * @brief Generate filter2d opencl precompiled cache.
     *
     * @param elem_type The element type of the src/dst array.
     * @param channel The channel of the src/dst array.
     * @param ksize The size of the filter2d kernel.
     * @param border_type The border type for handling border pixels.
     */
    static Status CLPrecompile(Context *ctx, ElemType elem_type, DT_S32 channel, DT_S32 ksize, BorderType border_type);
};

/**
 * @brief The 2D filer API.
 *
 * @anchor filter2d_details
 * The function smooths the iaura using the 2D filter. And in-place operation is not supported.
 *
 * @param ctx The pointer to the Context object
 * @param src The input iaura, And see the below for the supported data types.
 * @param dst The output iaura, And see the below for the supported data types.
 * @param kmat The kernel matrix for the 2D filter, and the elem type is F32C1. The width and height of the kmat must also be same and odd.
 * @param border_type The border type for handling border pixels
 * @param border_value The scalar values for border pixels used only when border_type is BorderType::CONSTANT.
 * @param target The platform on which this function runs
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 *
 * ### The supported data types and platforms
 * Platforms    | Data type(src or dst)                            | ksize
 * -------------|--------------------------------------------------|-----------
 * NONE         | U8Cx/U16Cx/S16Cx/F16Cx/F32Cx, x = N              | N
 * NEON         | U8Cx/U16Cx/S16Cx/F16Cx/F32Cx, x = 1, 2, 3        | 3, 5, 7
 * OpenCL       | U8Cx/U16Cx/S16Cx/F16Cx/F32Cx, x = 1, 2, 3        | 3, 5
 * OpenCL       | U8Cx/U16Cx/S16Cx/F16Cx/F32Cx, x = 1              | 7
 * HVX          | U8Cx/U16Cx/S16Cx, x = 1, 2, 3                    | 3, 5
 * HVX          | U8Cx/U16Cx/S16Cx, x = 1, 2                       | 7
 *
 * @note 1.N is positive integer(ksize is odd positive integer). <br>
 *       2.The above implementations supported all BorderType(CONSTANT/REPLICATE/REFLECT_101).
 */
AURA_EXPORTS Status IFilter2d(Context *ctx, const Mat &src, Mat &dst, const Mat &kmat,
                              BorderType border_type = BorderType::REFLECT_101,
                              const Scalar &border_value = Scalar(),
                              const OpTarget &target = OpTarget::Default());
/**
 * @}
 */

}
#endif // AURA_OPS_FILTER_FILTER2D_HPP__