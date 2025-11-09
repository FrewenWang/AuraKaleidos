#ifndef AURA_OPS_MATRIX_PSNR_HPP__
#define AURA_OPS_MATRIX_PSNR_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

/**
 * @defgroup ops Operators
 * @{
 *      @defgroup matrix Matrix Process
 *      @{
 *          @defgroup psnr Psnr
 *      @}
 * @}
 */

namespace aura
{

/**
 * @addtogroup psnr
 * @{
 */

/**
 * @brief Interface class representing an operation to compute Peak Signal-to-Noise Ratio (PSNR).
 *
 * The use of this class for the above-mentioned operation is not recommended.
 * It is recommended to use the `IPsnr` API, which internally calls this class.
 * The only recommended scenario for using this class is when the input or output type is 'CLMem'.
 *
 * The approximate internal call within the `IPsnr` function is as follows:
 *
 * @code
 * Psnr psnr(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize().
 * return OpCall(ctx, psnr, &src0, &src1, coef_r, result);
 * @endcode
 */
class AURA_EXPORTS Psnr : public Op
{
public:
    /**
     * @brief Constructor for the Psnr class.
     *
     * @param ctx The pointer to the Context object
     * @param target The platform on which this function runs
     */
    Psnr(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Sets the arguments for computing PSNR between two arrays.
     *
     * For more details, please refer to @ref psnr_details
     */
    Status SetArgs(const Array *src0, const Array *src1, DT_F64 coef_r, DT_F64 *result);
};

/**
 * @brief Computes the Peak Signal-to-Noise Ratio (PSNR) between two source matrices.
 *
 * @anchor psnr_details
 * This function computes the PSNR, a metric used to evaluate the quality of a reconstructed iaura compared to the
 * original iaura. It measures the ratio of the maximum possible power of a signal to the power of corrupting noise
 * that affects the quality of its representation.The formula for PSNR is:
 *
 * @f$ \text{PSNR} = 10 \cdot \log_{10}\left(\frac{{\text{MAX}^2}}{{\text{MSE}}}\right) @f$
 *
 * where \(\text{MAX}\) is the maximum possible pixel value and \(\text{MSE}\) is the Mean Squared Error between
 * the original and reconstructed iauras.
 *
 * @param ctx The pointer to the Context object
 * @param src0 The first source matrix for PSNR calculation.
 * @param src1 The second source matrix for PSNR calculation.
 * @param coef_r The maximum possible pixel value in an iaura.
 * @param result Pointer to store the computed PSNR result.
 * @param target The platform on which this function runs
 *
 * @return Status Status::OK if successful; otherwise, an appropriate error status.
 *
 * ### the supported data types and platforms
 * Platforms    | Data type(src0 or src1)
 * -------------|---------------------------------------------------
 * NONE         | U8Cx/S8Cx/U16Cx/S16Cx/U32Cx/S32Cx/F32Cx, x = N
 *
 * @note N is positive integer. Src0 and src1 must have same size and data type.
 */
AURA_EXPORTS Status IPsnr(Context *ctx, const Mat &src0, const Mat &src1, DT_F64 coef_r, DT_F64 *result, const OpTarget &target = OpTarget::Default());

/**
 * @}
 */

}// namespace aura

#endif// AURA_OPS_MATRIX_PSNR_HPP__
