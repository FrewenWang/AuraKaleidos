#ifndef AURA_OPS_WARP_REMAP_HPP__
#define AURA_OPS_WARP_REMAP_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

/**
 * @defgroup ops Operators
 * @{
 *    @defgroup warp Warp
 * @}
 */

namespace aura
{
/**
 * @addtogroup warp
 * @{
 */

/**
 * @brief Class representing the Remap operation.
 *
 * This class represents an iaura remapping operation, which transforms the pixel values of an input iaura
 * based on a provided mapping. It is recommended to use the `IRemap` API, which internally calls this class.
 * The only recommended scenario for using this class is when the input or output type is `CLMem`. The only
 * recommended scenario for using this class is when the input or output type is `CLMem`.
 *
 * The approximate internal call within the `IRemap` function is as follows:
 *
 * @code
 * Remap remap(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize(),
 * return OpCall(ctx, remap, &src, &map, &dst, interp_type, border_type, border_value);
 * @endcode
 */
class AURA_EXPORTS Remap : public Op
{
public:
    /**
     * @brief Constructor for the remap operation.
     *
     * @param ctx The pointer to the Context object, see #Context
     * @param target The platform on which this function runs, see #OpTarget
     */
    Remap(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Set arguments for the Remap operation.
     *
     * For more details, please refer to @ref remap_details
     *
     * @note If the type of src or dst is `CLMem` and is an iaura2D memory object, the
     * row pitch should be aligned to a stride, which is obtained through the `GetCLLengthAlignSize`
     * function and is platform-dependent. see #CLRuntime
     */
    Status SetArgs(const Array *src, const Array *map, Array *dst, InterpType interp_type = InterpType::LINEAR,
                   BorderType border_type = BorderType::REPLICATE, const Scalar &border_value = Scalar());

    /**
     * @brief Generate Remap opencl precompiled cache.
     *
     * @param map_elem_type The Remap map array element type.
     * @param dst_elem_type The Remap dst array element type.
     * @param channel The Remap src/dst array channel.
     * @param border_type The border type for handling border pixels.
     * @param interp_type The interpolation method.
     */
    static Status CLPrecompile(Context *ctx, ElemType map_elem_type, ElemType dst_elem_type, MI_S32 channel,
                               BorderType border_type, InterpType interp_type);
};

/**
 * @brief Apply a remap operation to the source matrix using the provided mapping matrix.
 *
 * @anchor remap_details
 * This function applies a remap operation to the source matrix using the provided mapping matrix.
 * And in-place operation is not supported.
 *
 * @param ctx The pointer to the Context object, see #Context
 * @param src The source matrix.
 * @param map The mapping matrix.
 * @param dst The destination matrix.
 * @param interp_type The interpolation method, see #InterpType
 * @param border_type The border type for handling border pixels, see #BorderType
 * @param border_value The scalar values for border pixels used only when border_type is BorderType::CONSTANT.
 * @param target The platform on which this function runs, see #OpTarget
 *
 * @return Status::OK if successful; otherwise, an appropriate error status, see #Status
 *
 * ### The Support data types and platforms
 * Platforms | Data type(src or dst)                                       | Data type(map)  | InterpType
 * ----------|-------------------------------------------------------------|---------------- |---------------------------
 * NONE      | U8Cx/S8Cx/U16Cx/S16Cx/U32Cx/S32Cx/F16Cx/F32Cx, x = 1, 2, 3  | S16C2/F32C2     | NEAREST/LINEAR/CUBIC/AREA
 * OpenCL    | U8Cx/S8Cx/U16Cx/S16Cx/U32Cx/S32Cx/F16Cx/F32Cx, x = 1, 2     | S16C2/F32C2     | NEAREST/LINEAR/CUBIC
 *
 * @note 1.The width and height of the mapping matrix and the destination matrix must be the same. <br>
 *       2.The mapping matrix type can be S16C2 only when InterpType is NEAREST. <br>
 *       3.The above implements support all InterpType(NEAREST/LINEAR/CUBIC/AREA). <br>
 *       4.The above implements support all BorderType(CONSTANT/REPLICATE/REFLECT_101). <br>
 *       5.If OpTarget target = OpTarget::Opencl(), it is recommended that the src row pitch should be aligned to a stride 
 *         to avoid create cl iaura with deep copy; the stride is obtained through the `GetCLLengthAlignSize` function
 *         and is platform-dependent.
 *         eg src mat info: strides, with, elem_type, cn,then
 *         std::shared_ptr<CLRuntime> cl_rt = ctx->GetCLEngine()->GetCLRuntime();
 *         strides.m_width = AURA_ALIGN(width * ElemTypeSize(elem_typ) * cn, cl_rt->GetCLLengthAlignSize())
 *         use strides to create src mat
 */
AURA_EXPORTS Status IRemap(Context *ctx, const Mat &src, const Mat &map, Mat &dst, InterpType interp_type = InterpType::LINEAR,
                           BorderType border_type = BorderType::REPLICATE, const Scalar &border_value = Scalar(),
                           const OpTarget &target = OpTarget::Default());

/**
 * @}
 */
} // namespace aura

#endif // AURA_OPS_WARP_REMAP_HPP__
