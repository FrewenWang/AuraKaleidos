#ifndef AURA_OPS_WARP_WARP_HPP__
#define AURA_OPS_WARP_WARP_HPP__

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
 * @brief Class representing the WarpAffine operation.
 *
 * This class represents an affine iaura warping operation, which transforms an input iaura based
 * on a given affine transformation matrix. It is recommended to use the `IWarpAffine` API,
 * which internally calls this class. The only recommended scenario for using this class is
 * when the input or output type is `CLMem`.
 *
 * The approximate internal call within the `IWarpAffine` function is as follows:
 *
 * @code
 * WarpAffine warp_affine(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize(),
 * return OpCall(ctx, warp_affine, &src, &matrix, &dst, interp_type, border_type, border_value);
 * @endcode
 */
class AURA_EXPORTS WarpAffine : public Op
{
public:
    /**
     * @brief Constructor for WarpAffine class.
     *
     * @param ctx The pointer to the Context object.
     * @param target The platform on which this function runs.
     */
    WarpAffine(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Set arguments for WarpAffine operation.
     *
     * For more details, please refer to @ref warp_affine_details
     *
     * @note If the type of src or dst is `CLMem` and is an iaura2D memory object, the
     * row pitch should be aligned to a stride, which is obtained through the `GetCLLengthAlignSize`
     * function and is platform-dependent.
     */
    Status SetArgs(const Array *src, const Array *matrix, Array *dst,
                   InterpType interp_type = InterpType::LINEAR,
                   BorderType border_type = BorderType::REPLICATE,
                   const Scalar &border_value = Scalar());

    /**
     * @brief Generate WarpAffine opencl precompiled cache.
     *
     * @param elem_type The WarpAffine src/dst array element type.
     * @param channel The WarpAffine src/dst array channel.
     * @param border_type The border type for handling border pixels.
     * @param interp_type The interpolation method.
     */
    static Status CLPrecompile(Context *ctx, ElemType elem_type, MI_S32 channel, BorderType border_type, InterpType interp_type);
};

/**
 * @brief Apply an affine warp to the src matrix.
 *
 * @anchor warp_affine_details
 * This function applies an affine warp to the src matrix using the provided affine transformation matrix.
 * And in-place operation is not supported.
 *
 * @param ctx The pointer to the Context object.
 * @param src The source matrix.
 * @param matrix The affine transformation matrix.
 * @param dst The destination matrix.
 * @param interp_type The interpolation method.
 * @param border_type The border type for handling border pixels.
 * @param border_value The scalar values for border pixels used only when border_type is BorderType::CONSTANT.
 * @param target The platform on which this function runs.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 *
 * ### The Support data types and platforms
 * Platforms | Data type(src or dst)                                       | InterpType                | BorderType
 * ----------|-------------------------------------------------------------|---------------------------|--------------------------------
 * NONE      | U8Cx/S8Cx/U16Cx/S16Cx/U32Cx/S32Cx/F16Cx/F32Cx, x = 1, 2, 3  | NEAREST/LINEAR/CUBIC/AREA | CONSTANT/REPLICATE/REFLECT_101
 * NEON      | U8Cx/S8Cx/U16Cx/S16Cx/U32Cx/S32Cx/F16Cx/F32Cx, x = 1, 2, 3  | NEAREST/LINEAR/CUBIC/AREA | CONSTANT/REPLICATE/REFLECT_101
 * OpenCL    | U8Cx/S8Cx/U16Cx/S16Cx/U32Cx/S32Cx/F16Cx/F32Cx, x = 1, 2     | NEAREST/LINEAR/CUBIC/AREA | CONSTANT/REPLICATE/REFLECT_101
 * HVX       | U8C1                                                        | NEAREST/LINEAR            | CONSTANT/REPLICATE
 *
 * @note 1.The data type of the matrix must be F64(2x3). <br>
 *       2.If OpTarget target = OpTarget::Opencl(), it is recommended that the row pitch should of src be aligned to a stride,
 *         the stride is obtained through the `GetCLLengthAlignSize` function and is platform-dependent.
 */
AURA_EXPORTS Status IWarpAffine(Context *ctx, const Mat &src, const Mat &matrix, Mat &dst,
                                InterpType interp_type = InterpType::LINEAR,
                                BorderType border_type = BorderType::REPLICATE,
                                const Scalar &border_value = Scalar(),
                                const OpTarget &target = OpTarget::Default());

/**
 * @brief Get the affine transformation matrix for a set of src and dst points.
 *
 * This function calculates the 2x3 affine transformation matrix that maps a set of src points to
 * a corresponding set of dst points.
 *
 * @param ctx The pointer to the Context object.
 * @param src The source points.
 * @param dst The destination points.
 *
 * @return The affine transformation matrix (2x3).
 */
AURA_EXPORTS Mat GetAffineTransform(Context *ctx, const std::vector<Point2> &src, const std::vector<Point2> &dst);

/**
 * @brief Class representing the WarpPerspective operation.
 *
 * This class represents a perspective iaura warping operation, which transforms an input iaura based on
 * a given perspective transformation matrix. It is recommended to use the `IWarpPerspective` API, which
 * internally calls this class. The only recommended scenario for using this class is when the input or
 * output type is `CLMem`.
 *
 * The approximate internal call within the `IWarpPerspective` function is as follows:
 *
 * @code
 * WarpPerspective warp_perspective(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize(),
 * return OpCall(ctx, warp_perspective, &src, &matrix, &dst, interp_type, border_type, border_value);
 * @endcode
 */
class AURA_EXPORTS WarpPerspective : public Op
{
public:
    /**
     * @brief Constructor for the perspective warp operation.
     *
     * @param ctx The pointer to the Context object.
     * @param target The platform on which this function runs.
     */
    WarpPerspective(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Set arguments for the WarpPerspective operation.
     *
     * For more details, please refer to @ref warp_perspective_details
     *
     * @note If the type of src or dst is `CLMem` and is an iaura2D memory object, the
     * row pitch should be aligned to a stride, which is obtained through the `GetCLLengthAlignSize`
     * function and is platform-dependent.
     */
    Status SetArgs(const Array *src, const Array *matrix, Array *dst,
                   InterpType interp_type = InterpType::LINEAR,
                   BorderType border_type = BorderType::REPLICATE,
                   const Scalar &border_value = Scalar());

    /**
     * @brief Generate WarpPerspective opencl precompiled cache.
     *
     * @param elem_type The WarpPerspective src/dst array element type.
     * @param channel The WarpPerspective src/dst array channel.
     * @param border_type The border type for handling border pixels.
     * @param interp_type The interpolation method.
     */
    static Status CLPrecompile(Context *ctx, ElemType elem_type, MI_S32 channel, BorderType border_type, InterpType interp_type);
};

/**
 * @brief Apply a perspective warp to the src matrix.
 *
 * @anchor warp_perspective_details
 * This function applies a perspective warp to the src matrix using the provided perspective transformation matrix.
 * And in-place operation is not supported.
 *
 * @param ctx The pointer to the Context object.
 * @param src The source matrix.
 * @param matrix The perspective transformation matrix.
 * @param dst The destination matrix.
 * @param interp_type The interpolation method.
 * @param border_type The border type for handling border pixels.
 * @param border_value The scalar values for border pixels used only when border_type is BorderType::CONSTANT.
 * @param target The platform on which this function runs.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 *
 * ### The Support data types and platforms
 * Platforms | Data type(src or dst)                                       | InterpType                | BorderType
 * ----------|-------------------------------------------------------------|---------------------------|--------------------------------
 * NONE      | U8Cx/S8Cx/U16Cx/S16Cx/U32Cx/S32Cx/F16Cx/F32Cx, x = 1, 2, 3  | NEAREST/LINEAR/CUBIC/AREA | CONSTANT/REPLICATE/REFLECT_101
 * NEON      | U8Cx/S8Cx/U16Cx/S16Cx/U32Cx/S32Cx/F16Cx/F32Cx, x = 1, 2, 3  | NEAREST/LINEAR/CUBIC/AREA | CONSTANT/REPLICATE/REFLECT_101
 * OpenCL    | U8Cx/S8Cx/U16Cx/S16Cx/U32Cx/S32Cx/F16Cx/F32Cx, x = 1, 2     | NEAREST/LINEAR/CUBIC/AREA | CONSTANT/REPLICATE/REFLECT_101
 * HVX       | U8C1                                                        | NEAREST/LINEAR            | CONSTANT/REPLICATE
 *
 * @note 1.The data type of the matrix must be F64(3x3). <br>
 *       2.If OpTarget target = OpTarget::Opencl(), it is recommended that the row pitch should of src be aligned to a stride,
 *         the stride is obtained through the `GetCLLengthAlignSize` function and is platform-dependent.
 */

AURA_EXPORTS Status IWarpPerspective(Context *ctx, const Mat &src, const Mat &matrix, Mat &dst,
                                     InterpType interp_type = InterpType::LINEAR,
                                     BorderType border_type = BorderType::REPLICATE,
                                     const Scalar &border_value = Scalar(),
                                     const OpTarget &target = OpTarget::Default());

/**
 * @brief Get the perspective transformation matrix for a set of src and dst points.
 *
 * This function calculates the 3x3 perspective transformation matrix that maps a set of src points to
 * a corresponding set of dst points.
 *
 * @param ctx The pointer to the Context object.
 * @param src The source points.
 * @param dst The destination points.
 *
 * @return The perspective transformation matrix (3x3).
 */
AURA_EXPORTS Mat GetPerspectiveTransform(Context *ctx, const std::vector<Point2> &src, const std::vector<Point2> &dst);

/**
 * @brief Get the 2D rotation matrix.
 *
 * This function calculates the 2x3 rotation matrix for rotating an iaura around a specified center with a given angle and scale.
 *
 * @param ctx The pointer to the Context object.
 * @param center The rotation center.
 * @param angle The rotation angle in degrees.
 * @param scale The scaling factor.
 *
 * @return The rotation matrix(2x3).
 */
AURA_EXPORTS Mat GetRotationMatrix2D(Context *ctx, const Point2 &center, MI_F64 angle, MI_F64 scale);

/**
 * @}
 */
} // namespace aura

#endif // AURA_OPS_WARP_WARP_HPP__