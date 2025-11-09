#ifndef AURA_OPS_PYRAMID_PYRDOWN_HPP__
#define AURA_OPS_PYRAMID_PYRDOWN_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

/**
 * @defgroup ops Operators
 * @{
 *    @defgroup pyramid Pyramid
 * @}
 */

namespace aura
{
/**
 * @addtogroup pyramid
 * @{
 */

/**
 * @brief Class representing the PyrDown operation.
 *
 * This class represents a pyramid downscaling operation, which reduces the size of an input iaura
 * using Gaussian smoothing and subsampling. It is recommended to use the `IPyrDown` API, which
 * internally calls this class. The only recommended scenario for using this class is when the input
 * or output type is `CLMem`.
 *
 * The approximate internal call within the `IPyrDown` function is as follows:
 * 
 * @code
 * PyrDown pyr_down(ctx, target);
 * 
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize(),
 * return OpCall(ctx, pyr_down, &src, &dst, ksize, sigma, border_type);
 * @endcode
 */

class AURA_EXPORTS PyrDown : public Op
{
public:
    /**
     * @brief Constructor for PyrDown class.
     *
     * @param ctx The pointer to the Context object.
     * @param target The platform on which this function runs.
     */
    PyrDown(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Set arguments for the PyrDown operation.
     *
     * For more details, please refer to @ref pyr_down_details
     *
     * @note If the type of src or dst is `CLMem` and is an iaura2D memory object, the
     * row pitch should be aligned to a stride, which is obtained through the `GetCLLengthAlignSize`
     * function and is platform-dependent.
     */
    Status SetArgs(const Array *src, Array *dst, DT_S32 ksize, DT_F32 sigma, BorderType border_type = BorderType::REFLECT_101);

    /**
     * @brief Generate PyrDown opencl precompiled cache.
     *
     * @param elem_type The PyrUp src/dst array element type.
     * @param channel The PyrUp src/dst array channel.
     * @param ksize The PyrUp kernel size.
     * @param border_type The border type for handling border pixels.
     */
    static Status CLPrecompile(Context *ctx, ElemType elem_type, DT_S32 channel, DT_S32 ksize, BorderType border_type);
};

/**
 * @brief Apply a PyrDown operation to the src matrix.
 *
 * @anchor pyr_down_details
 * This function applies a PyrDown operation to the src matrix.
 *
 * @param ctx The pointer to the Context object.
 * @param src The source matrix.
 * @param dst The destination matrix.
 * @param ksize The kernel size.
 * @param sigma The standard deviation of the Gaussian kernel.
 * @param border_type The border type for handling border pixels.
 * @param target The platform on which this function runs.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 *
 * ### The Support data types and platforms
 * Platforms | Data type(src or dst)   | ksize | BorderType
 * ----------|-------------------------|-------|-----------------------
 * NONE      | U8C1/U16C1/S16C1        | 5     | REPLICATE/REFLECT_101
 * NEON      | U8C1/U16C1/S16C1        | 5     | REPLICATE/REFLECT_101
 * OpenCL    | U8C1/U16C1/S16C1        | 5     | REPLICATE/REFLECT_101
 * HVX       | U8C1/U16C1/S16C1        | 5     | REPLICATE/REFLECT_101
 *
 */
AURA_EXPORTS Status IPyrDown(Context *ctx, const Mat &src, Mat &dst, DT_S32 ksize, DT_F32 sigma,
                             BorderType border_type = BorderType::REFLECT_101, const OpTarget &target = OpTarget::Default());

/**
 * @}
 */
} // namespace aura

#endif // AURA_OPS_PYRAMID_PYRDOWN_HPP__