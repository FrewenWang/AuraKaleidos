#ifndef AURA_OPS_RESIZE_RESIZE_HPP__
#define AURA_OPS_RESIZE_RESIZE_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

/**
 * @defgroup ops Operators
 * @{
 *    @defgroup resize Resize
 * @}
 */

namespace aura
{
/**
 * @addtogroup resize
 * @{
 */

/**
 * @brief Class representing the Resize operation.
 *
 * This class represents an iaura resizing operation, which changes the size of an input iaura
 * using various interpolation method. It is recommended to use the `IResize` API, which
 * internally calls this class. The only recommended scenario for using this class is when
 * the input or output type is `CLMem`.
 * 
 * The approximate internal call within the `IResize` function is as follows:
 * 
 * @code
 * Resize resize(ctx, target);
 * 
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize(),
 * return OpCall(ctx, resize, &src, &dst, type);
 * @endcode
 */
class AURA_EXPORTS Resize : public Op
{
public:
    /**
     * @brief Constructor for the resize operation.
     *
     * @param ctx The pointer to the Context object.
     * @param target The platform on which this function runs.
     */
    Resize(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Set arguments for the Resize operation.
     *
     * For more details, please refer to @ref resize_details
     *
     * @note If the type of src or dst is `CLMem` and is an iaura2D memory object, the
     * row pitch should be aligned to a stride, which is obtained through the `GetCLLengthAlignSize`
     * function and is platform-dependent.
     */
    Status SetArgs(const Array *src, Array *dst, InterpType type);

    /**
     * @brief Generate resize opencl precompiled cache.
     *
     * @param elem_type The src/dst array element type.
     * @param channel The src/dst array channel.
     * @param iwidth The width of the src.
     * @param iheight The height of the src.
     * @param owidth The width of the dst.
     * @param oheight The height of the dst.
     * @param type The interpolation method.
     */
    static Status CLPrecompile(Context *ctx, ElemType elem_type, DT_S32 channel, DT_S32 iwidth, DT_S32 iheight, DT_S32 owidth, DT_S32 oheight,
                               InterpType interp_type);
};

/**
 * @brief Resize operation function.
 *
 * @anchor resize_details
 * This function resizes the source matrix to the specified dimensions. And in-place operation is not supported.
 *
 * @param ctx The pointer to the Context object.
 * @param src The source matrix.
 * @param dst The destination matrix.
 * @param type The interpolation method.
 * @param target The platform on which this function runs.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 *
 * ### The Support data types and platforms
 * Platforms | Data type(src or dst)                          | InterpType
 * ----------|------------------------------------------------|---------------------------
 * NONE      | U8Cx/S8Cx/U16Cx/S16Cx/F16Cx/F32Cx, x = N       | NEAREST/LINEAR/CUBIC/AREA
 * NEON      | U8Cx/S8Cx/U16Cx/S16Cx/F16Cx/F32Cx, x = 1, 2, 3 | NEAREST/LINEAR/CUBIC/AREA
 * OpenCL    | U8Cx/S8Cx/U16Cx/S16Cx/F16Cx/F32Cx, x = 1, 2, 3 | NEAREST/LINEAR/CUBIC/AREA
 * HVX       | U8Cx/S8Cx/U16Cx/S16Cx, x = 1, 2, 3             | NEAREST/LINEAR/CUBIC/AREA
 *
 * @note N is positive integer.
 */
AURA_EXPORTS Status IResize(Context *ctx, const Mat &src, Mat &dst, InterpType type,
                            const OpTarget &target = OpTarget::Default());

/**
 * @}
 */
} // namespace aura

#endif // AURA_OPS_RESIZE_RESIZE_HPP__