#ifndef AURA_OPS_FILTER_GAUSSIAN_HPP__
#define AURA_OPS_FILTER_GAUSSIAN_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

/**
 * @defgroup ops Operators
 * @{
 *    @defgroup imgproc_filter Iaura Filtering
 *    @{
 *          @defgroup gaussian Gaussian Filter
 *    @}
 * @}
 */

namespace aura
{
/**
 * @addtogroup gaussian
 * @{
 */

/**
 * @brief Get Gaussian Kernel
 *
 * This function computes and returns a vector representing a Gaussian kernel
 * with the specified kernel size (`ksize`) and standard deviation (`sigma`).
 *
 * @param ksize The size of the Gaussian kernel (must be an odd number).
 * @param sigma The standard deviation of the Gaussian kernel.
 *
 * @return A vector of `DT_F32` type representing the Gaussian kernel.
 *         The kernel is centered at index `ksize / 2`.
 */
AURA_EXPORTS std::vector<DT_F32> GetGaussianKernel(DT_S32 ksize, DT_F32 sigma);

/**
 * @brief The Gaussian filter interface class.
 *
 * The use of this class for Gaussian filtering is not recommended. It is recommended to use the `IGaussian` API,
 * which internally calls this class. The only recommended scenario for using this class is when the input or output type is `CLMem`.
 *
 * The approximate internal call within the `IGaussian` function is as follows:
 *
 * @code
 * Gaussian gaussian(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize().
 * return OpCall(ctx, gaussian, &src, &dst, ksize, sigma, border_type, border_value);
 * @endcode
 */
class AURA_EXPORTS Gaussian : public Op
{
public:
    /**
     * @brief Constructor for the Gaussian filter class.
     *
     * @param ctx The pointer to the Context object
     * @param target The platform on which this function runs
     */
    Gaussian(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Set the arguments for the Gaussian filter operation.
     *
     * For more details, please refer to @ref gaussian_details
     *
     * @note If the type of src or dst is `CLMem` and is an iaura2D memory object, the row pitch should be aligned to a stride,
     * which is obtained through the `GetCLLengthAlignSize` function and is platform-dependent.
     */
    Status SetArgs(const Array *src, Array *dst, DT_S32 ksize, DT_F32 sigma,
                   BorderType border_type = BorderType::REFLECT_101,
                   const Scalar &border_value = Scalar());

    /**
     * @brief Generate gaussian opencl precompiled cache.
     *
     * @param elem_type The element type of the src/dst array.
     * @param channel The channel of the src/dst array.
     * @param ksize The size of the gaussian kernel.
     * @param border_type The border type for handling border pixels.
     */
    static Status CLPrecompile(Context *ctx, ElemType elem_type, DT_S32 channel, DT_S32 ksize, BorderType border_type);
};

/**
 * @brief Using the gaussian filter to blur an iaura.  使用高斯滤波器对图像进行模糊处理。
 * 外部调用高斯函数的唯一入口
 * @anchor gaussian_details
 * The function smooths the iaura using the gaussain filter with the @f$ \texttt{kszie} \times \texttt{kszie} @f$ kernel size.
 * The gaussian kernel standard deviations in X and Y direction should be same. And in-place operation is not supported.
 *
 * @param ctx The pointer to the Context object
 * @param src The input iaura, And see the below for the supported data types.
 * @param dst The output iaura, And see the below for the supported data types.
 * @param ksize The filter kernel size, And see the below for the supported sizes.
 * @param sigma The standard deviation in X and Y direction.
 * @param border_type The border type for handling border pixels
 * @param border_value The scalar values for border pixels used only when border_type is BorderType::CONSTANT.
 * @param target The platform on which this function runs
 *
 * @return Status::OK if successful; otherwise, an appropriate error status
 *
 * ### The supported data types and platforms
 * Platforms    | Data type(src or dst)                               | ksize
 * -------------|-----------------------------------------------------|-----------
 * NONE         | U8Cx/U16Cx/S16Cx/F16Cx/U32Cx/S32Cx/F32Cx, x = N     | N
 * NEON         | U8Cx/U16Cx/S16Cx/F16Cx/F32Cx, x = 1, 2, 3           | 3, 5, 7
 * OpenCL       | U8Cx/U16Cx/S16Cx/F16Cx/F32Cx, x = 1, 2, 3           | 3, 5
 * OpenCL       | U8Cx/U16Cx/S16Cx/F16Cx/F32Cx, x = 1                 | 7
 * HVX          | U8Cx/U16Cx/S16Cx/U32Cx/S32cX, x = 1, 2, 3           | 3, 5, 7, 9
 *
 * @note 1.N is positive integer(ksize is odd positive integer). <br>
 *       2.The above implementations supported all BorderType(CONSTANT/REPLICATE/REFLECT_101).
 */
AURA_EXPORTS Status IGaussian(Context *ctx, const Mat &src, Mat &dst, DT_S32 ksize, DT_F32 sigma,
                              BorderType border_type     = BorderType::REFLECT_101,
                              const Scalar &border_value = Scalar(),
                              const OpTarget &target     = OpTarget::Default());
/**
 * @}
 */

} // namespace aura

#endif // AURA_OPS_FILTER_GAUSSIAN_HPP__