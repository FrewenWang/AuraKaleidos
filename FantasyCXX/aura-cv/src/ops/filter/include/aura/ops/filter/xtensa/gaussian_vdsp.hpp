#ifndef AURA_OPS_FILTER_GAUSSIAN_VDSP_HPP__
#define AURA_OPS_FILTER_GAUSSIAN_VDSP_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/xtensa.h"

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
namespace xtensa
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
 * @return A vector of `MI_F32` type representing the Gaussian kernel.
 *         The kernel is centered at index `ksize / 2`.
 */
vector<MI_F32> GetGaussianKernel(MI_S32 ksize, MI_F32 sigma);

/**
 * @brief Enumerated type for Gaussian blur filter.
 *
 * This enumeration defines types for Gaussian blur filter operations.
 * The types include different sizes and standard deviations for the Gaussian kernel.
 */
class GaussianVdsp : public VdspOp
{
public:
    /**
     * @brief Constructor for the GaussianVdsp filter class.
     *
     * @param xv_tm The pointer to the TileManager object
     * @param mode  The execute mode.
     */
    GaussianVdsp(TileManager tm, ExecuteMode mode = ExecuteMode::FRAME);

    /**
     * @brief Set the arguments for the GaussianVdsp filter operation.
     *
     * @param src The source Mat.
     * @param dst  The destination Mat.
     * @param ksize  The filter kernel size.
     * @param sigma  The filter kernel sigma.
     * @param border_type  The border type for handling border pixels.
     * @param border_value  The scalar values for border pixels used only when border_type is BorderType::CONSTANT.
     *
     * @return Status::OK if the operation is successful; otherwise, an appropriate error status.
     */
    Status SetArgs(const Mat *src, Mat *dst, MI_S32 ksize, MI_F32 sigma,
                   BorderType border_type = BorderType::REFLECT_101,
                   const Scalar &border_value = Scalar());

    /**
     * @brief Set the arguments for the GaussianVdsp filter operation.
     *
     * @param src The source TileWrapper.
     * @param dst  The destination TileWrapper.
     * @param ksize  The filter kernel size.
     * @param sigma  The filter kernel sigma.
     *
     * @return Status::OK if the operation is successful; otherwise, an appropriate error status.
     */
    Status SetArgs(const TileWrapper *src, TileWrapper *dst, MI_S32 ksize, MI_F32 sigma);

    AURA_VDSP_OP_HPP();
};

/**
 * @}
 */
} // namespace xtensa
} // namespace aura

#endif // AURA_OPS_FILTER_BOXFILTER_VDSP_HPP__