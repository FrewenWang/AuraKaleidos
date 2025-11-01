#ifndef AURA_OPS_FEATURE2D_HARRIS_HPP__
#define AURA_OPS_FEATURE2D_HARRIS_HPP__

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
 * @brief Harris corner detector class.
 *
 * The use of this class for Harris is not recommended. It is recommended to use the `IHarris` API,
 * which internally calls this class. The only recommended scenario for using this class is when the input or output type is `CLMem`.
 * 
 * The approximate internal call within the `IHarris` function is as follows:
 * 
 * @code
 * Harris harris(ctx, target);
 * 
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize().
 * return OpCall(ctx, harris, &src, ...);
 * @endcode
 */
class AURA_EXPORTS Harris : public Op
{
public:
    /**
     * @brief Constructor for Harris class.
     *
     * @param ctx The pointer to the Context object
     * @param target The platform on which this function runs
     */
    Harris(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Sets the arguments for the Harris corner detector operation.
     *
     * For more details, please refer to @ref harris_details
     *
     * @note If the type of src or dst is `CLMem` and is an iaura2D memory object, the row pitch should be aligned to a stride,
     * which is obtained through the `GetCLLengthAlignSize` function and is platform-dependent. 
     */
    Status SetArgs(const Array *src, Array *dst, MI_S32 block_size, MI_S32 ksize, MI_F64 k,
                   BorderType border_type = BorderType::REFLECT_101,
                   const Scalar &border_value = Scalar());
};

/**
 * @brief Harris corner detector function.
 *
 * @anchor harris_details
 * This function applies the Harris corner detector to the input iaura.
 *
 * @param ctx The pointer to the Context object
 * @param src The input iaura, And see the below for the supported data types.
 * @param dst The output array containing the Harris corner response.
 * @param block_size The neighborhood size for the Harris detector.
 * @param ksize Aperture parameter for the Sobel operator.
 * @param k The Harris detector free parameter.
 * @param border_type The border type for handling border pixels
 * @param border_value The scalar values for border pixels used only when border_type is BorderType::CONSTANT.
 * @param target The platform on which this function runs.
 * 
 * @return Status::OK if successful; otherwise, an appropriate error status.
 * 
 * ### The supported data types and platforms
 * Platforms    | Data type(src)  | Data type(dst)
 * -------------|-----------------|--------------------
 * NONE         | U8C1/F32C1      | F32C1
 * NEON         | U8C1/F32C1      | F32C1
 *
 * @note The above implementations supported all BorderType(CONSTANT/REPLICATE/REFLECT_101).
 */
AURA_EXPORTS Status IHarris(Context *ctx, const Mat &src, Mat &dst, MI_S32 block_size, MI_S32 ksize, MI_F64 k,
                            BorderType border_type = BorderType::REFLECT_101, const Scalar &border_value = Scalar(),
                            const OpTarget &target = OpTarget::Default());
/**
 * @}
*/
} // namespace aura

#endif // AURA_OPS_FEATURE2D_HARRIS_HPP__