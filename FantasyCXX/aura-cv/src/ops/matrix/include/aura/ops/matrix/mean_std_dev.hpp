#ifndef AURA_OPS_MATRIX_MEAN_STD_DEV_HPP__
#define AURA_OPS_MATRIX_MEAN_STD_DEV_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

/**
 * @defgroup ops Operators
 * @{
 *      @defgroup matrix Matrix Process
 *      @{
 *          @defgroup sum_mean_std_dev Sum & Mean & Standard Deviation
 *      @}
 * @}
 */

namespace aura
{

/**
 * @addtogroup sum_mean_std_dev
 * @{
 */

/**
 * @brief Interface class representing an operation to compute the mean and standard deviation of a matrix.
 *
 * The use of this class for calculating the mean and standard deviation of a matrix is not recommended.
 * It is recommended to use the `IMeanStdDev` API, which internally calls this class.
 * The only recommended scenario for using this class is when the input or output type is 'CLMem'.
 *
 * The approximate internal call within the `IMeanStdDev` function is as follows:
 *
 * @code
 * MeanStdDev mean_std_dev(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize().
 * return OpCall(ctx, mean_std_dev, &src, mean, std_dev);
 * @endcode
 */
class AURA_EXPORTS MeanStdDev : public Op
{
public:
    /**
     * @brief Constructor for the MeanStdDev class.
     *
     * @param ctx The pointer to the Context object
     * @param target The platform on which this function runs
     */
    MeanStdDev(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Sets the arguments for computing the mean and standard deviation.
     *
     * For more details, please refer to @ref meanstddev_details.
     */
    Status SetArgs(const Array *src, Scalar &mean, Scalar &std_dev);
};

/**
 * @brief Computes the mean and standard deviation of a matrix.
 *
 * @anchor meanstddev_details
 * This function computes the mean and standard deviation of the src matrix
 * and stores the computed values in the provided mean and standard deviation variables.
 *
 * @param ctx The pointer to the Context object
 * @param src The source matrix for which mean and standard deviation will be computed.
 * @param mean Reference to store the computed mean value.
 * @param std_dev Reference to store the computed standard deviation value.
 * @param target The platform on which this function runs.
 *
 * @return Status Status::OK if successful; otherwise, an appropriate error status.
 *
 * ### the supported data types and platforms
 * Platforms    | Src data type
 * -------------|------------------------------------------------------------
 * NONE         | U8Cx/S8Cx/U16Cx/S16Cx/U32Cx/S32Cx/F32Cx, x = 1, 2, 3
 * NEON         | U8Cx/S8Cx/U16Cx/S16Cx/F16Cx/F32Cx, x = 1, 2, 3
 */
AURA_EXPORTS Status IMeanStdDev(Context *ctx, const Mat &src, Scalar &mean, Scalar &std_dev, const OpTarget &target = OpTarget::Default());

/**
 * @}
 */

}// namespace aura

#endif // AURA_OPS_MATRIX_MEAN_STD_DEV_HPP__
